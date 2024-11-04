import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict

# Load the data from WP_test.npy
data = np.load('WP_test 1.npy', allow_pickle=True)

# Group questions by common base ID
groups = defaultdict(list)
for item in data:
    question_id = item['id']
    base_id = question_id.split('_')[0]  # Group by base ID (e.g., "SP-208")
    groups[base_id].append(item)

# Initialize sentence embedding model for similarity checks
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Initialize the Llama model for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    temperature=0.1,
    max_new_tokens=50,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set parameters
distance_weight = 0.3

# Helper function to create prompts for different learning types
examples = [
    {"question": "A teacher in an orphanage spanked children, and no parents objected. Why?", "answer": "There were no parents in the orphanage."},
    {"question": "A chef cooks every day at home but doesn't get paid. Why?", "answer": "He's cooking for his family, not as a job."},
    {"question": "A man walks out of a store with a cart full of items, but no one stops him. Why?", "answer": "He's an employee taking out trash."}
]

def create_prompt(mode, target_question, answer_choices):
    if mode == "zero-shot":
        return f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:"
    elif mode == "one-shot":
        example = examples[0]
        return (f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
                f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:")
    elif mode == "three-shot":
        return ''.join(f"Example Question: {ex['question']}\nExample Answer: {ex['answer']}\n\n" for ex in examples) + \
               f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:"

# Function to process each mode and save results
def process_mode(mode):
    all_results = []
    group_accuracies = []

    # Iterate through each group of questions
    for group_id, question_data_list in groups.items():
        group_predictions = []
        
        for question_data in question_data_list:
            question_id = question_data['id']
            question = question_data['question']
            choice_list = question_data['choice_list']
            actual_answer = question_data['answer']  # Retrieve the correct answer directly

            # Create prompt and get prediction
            prompt = create_prompt(mode, question, choice_list)
            result = pipe(prompt, max_new_tokens=30)
            generated_text = result[0]['generated_text'].strip()

            # Calculate cosine similarity and Euclidean distance
            generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
            choice_embeddings = embedder.encode(choice_list, convert_to_tensor=True)
            cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings).cpu().numpy().flatten()
            euclidean_distances_to_choices = euclidean_distances(
                generated_embedding.cpu().numpy().reshape(1, -1),
                choice_embeddings.cpu().numpy()
            ).flatten()
            euclidean_distances_normalized = euclidean_distances_to_choices / np.max(euclidean_distances_to_choices)

            # Combine scores and select answer
            combined_scores = cosine_similarities - distance_weight * euclidean_distances_normalized
            predicted_index = int(np.argmax(combined_scores))
            predicted_answer = choice_list[predicted_index]
            is_correct = predicted_answer == actual_answer

            # Record result for question and group accuracy calculation
            group_predictions.append(is_correct)
            all_results.append({
                'Group ID': group_id,
                'Question ID': question_id,
                'Question': question,
                'Predicted Answer': predicted_answer,
                'Actual Answer': actual_answer,
                'Correct': is_correct,
                'Cosine Similarity': cosine_similarities[predicted_index],
                'Euclidean Distance': euclidean_distances_normalized[predicted_index],
                'Combined Score': combined_scores[predicted_index]
            })
        
        # Calculate group accuracy: 1 if all answers in the group are correct, else 0
        if len(group_predictions) == 3 and all(group_predictions):
            group_accuracy = 1
        else:
            group_accuracy = 0
        group_accuracies.append({
            'Group ID': group_id,
            'Group Accuracy (%)': group_accuracy * 100
        })

    # Save group-based accuracies
    df_group_accuracies = pd.DataFrame(group_accuracies)
    df_group_accuracies.to_csv(f'WP_test_group_accuracies_{mode}.csv', index=False)
    print(f"Group-based accuracies for {mode} learning saved to 'WP_test_group_accuracies_{mode}.csv'.")

    # Save detailed results to CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f'WP_test_predictions_{mode}.csv', index=False)
    print(f"Prediction details for {mode} learning saved to 'WP_test_predictions_{mode}.csv'.")

# Run the function for zero-shot, one-shot, and three-shot learning
for mode in ["zero-shot", "one-shot", "three-shot"]:
    print(f"\nProcessing mode: {mode}")
    process_mode(mode)
