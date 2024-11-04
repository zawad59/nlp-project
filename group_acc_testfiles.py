import numpy as np
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# Load training and test data
train_data = np.load('SP_train 1.npy', allow_pickle=True)
test_data = np.load('SP_test 1.npy', allow_pickle=True)

# Group questions by common base ID
groups = defaultdict(list)
for item in test_data:
    question_id = item['id']
    base_id = question_id.split('_')[0]  # Group by base ID (e.g., "SP-1")
    groups[base_id].append(item)

# Initialize the SentenceTransformer model for embedding similarity calculations
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Initialize the Llama model and tokenizer for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# One-shot example
one_shot_example = {
    "question": "A teacher in an orphanage spanked children, and no parents objected. Why?",
    "answer": "There were no parents in the orphanage."
}

# Helper function to create prompts
def create_prompt(mode, target_question, answer_choices):
    if mode == "zero-shot":
        return f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:"
    elif mode == "one-shot":
        example = one_shot_example
        return (f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
                f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:")

# Function to process each mode and save results
def process_mode(mode):
    all_results = []
    group_accuracies = []

    # Iterate through each group of questions
    for group_id, question_data_list in groups.items():
        group_predictions = []
        group_detail = []  # Store detailed info for this group
        
        for question_data in question_data_list:
            question_id = question_data['id']
            question = question_data['question']
            choice_list = question_data['choice_list']
            actual_answer = question_data['answer']  # Retrieve the correct answer directly

            # Create prompt and get prediction
            prompt = create_prompt(mode, question, choice_list)
            result = pipe(prompt, max_new_tokens=50)
            generated_text = result[0]['generated_text'].strip()

            # Calculate cosine similarity for selecting the best answer
            generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
            choice_embeddings = embedder.encode(choice_list, convert_to_tensor=True)
            cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings).cpu().numpy().flatten()
            predicted_index = int(np.argmax(cosine_similarities))
            predicted_answer = choice_list[predicted_index]
            is_correct = predicted_answer == actual_answer

            # Record result for question and group accuracy calculation
            group_predictions.append(is_correct)
            group_detail.append({
                'Group ID': group_id,
                'Question ID': question_id,
                'Question': question,
                'Predicted Answer': predicted_answer,
                'Actual Answer': actual_answer,
                'Correct': is_correct,
                'Cosine Similarity': cosine_similarities[predicted_index]
            })
        
        # Calculate group accuracy: 1 if all answers in the group are correct, else 0
        group_accuracy = 1 if all(group_predictions) else 0
        group_accuracies.append({
            'Group ID': group_id,
            'Group Accuracy (%)': group_accuracy * 100
        })
        
        # Append each question's result in the group to all_results
        all_results.extend(group_detail)

        # Print verification output for each group
        print(f"\nGroup ID: {group_id}")
        for detail in group_detail:
            print(f"Question ID: {detail['Question ID']}")
            print(f"  - Predicted Answer: {detail['Predicted Answer']}")
            print(f"  - Actual Answer: {detail['Actual Answer']}")
            print(f"  - Correct: {detail['Correct']}")
        print(f"Group Accuracy: {group_accuracy * 100}%\n")

    # Save group-based accuracies
    df_group_accuracies = pd.DataFrame(group_accuracies)
    df_group_accuracies.to_csv(f'SP_test_group_accuracies_{mode}.csv', index=False)
    print(f"Group-based accuracies for {mode} learning saved to 'SP11_test_group_accuracies_{mode}.csv'.")

    # Save detailed results to CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f'SP_test_predictions_{mode}.csv', index=False)
    print(f"Prediction details for {mode} learning saved to 'SP11_test_predictions_{mode}.csv'.")

# Run the function for zero-shot and one-shot learning
for mode in ["zero-shot", "one-shot"]:
    print(f"\nProcessing mode: {mode}")
    process_mode(mode)
