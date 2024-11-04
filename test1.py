import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import euclidean_distances

# Load the data from word_test.npy and word_test_answers.npy
data = np.load('word_test.npy', allow_pickle=True)
answers_data = np.load('word_test_answers.npy', allow_pickle=True)

# Extract question IDs and correct answer indices
question_ids = [item[0] for item in answers_data]
correct_answer_indices = [int(item[1]) for item in answers_data]

# Initialize sentence embedding model for similarity checks
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Initialize the Llama model for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    temperature=0.1,  # Lower temperature for more deterministic responses
    max_new_tokens=50,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set parameters
distance_weight = 0.3  # Adjust this weight to tune the influence of Euclidean distance

# Predefined examples for one-shot and three-shot prompts
examples = [
    {"question": "A teacher in an orphanage spanked children, and no parents objected. Why?", "answer": "There were no parents in the orphanage."},
    {"question": "A chef cooks every day at home but doesn't get paid. Why?", "answer": "He's cooking for his family, not as a job."},
    {"question": "A man walks out of a store with a cart full of items, but no one stops him. Why?", "answer": "He's an employee taking out trash."}
]

# Helper function to create prompts for different learning types
def create_prompt(mode, target_question, answer_choices):
    if mode == "zero-shot":
        prompt = f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:"
    elif mode == "one-shot":
        example = examples[0]
        prompt = (f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
                  f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:")
    elif mode == "three-shot":
        prompt = ""
        for example in examples:
            prompt += f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
        prompt += f"Question: {target_question}\nSelect the answer that best fits the question:\n{', '.join(answer_choices)}\nAnswer:"
    return prompt

# Function to process each mode and save results
def process_mode(mode):
    all_results = []
    
    # Process each question and make predictions
    for idx, item in enumerate(data):
        question = item['question']
        choice_list = item['choice_list']
        
        # Use the index to get the correct question ID and answer index
        question_id = question_ids[idx]
        correct_answer_index = correct_answer_indices[idx]
        actual_answer = choice_list[correct_answer_index]

        # Create prompt based on learning mode
        prompt = create_prompt(mode, question, choice_list)

        # Generate response
        result = pipe(prompt, max_new_tokens=30)
        generated_text = result[0]['generated_text'].strip()

        # Calculate similarity and distance between generated text and each choice
        generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
        choice_embeddings = embedder.encode(choice_list, convert_to_tensor=True)

        # Calculate cosine similarities and Euclidean distances
        cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings).cpu().numpy().flatten()
        euclidean_distances_to_choices = euclidean_distances(
            generated_embedding.cpu().numpy().reshape(1, -1),
            choice_embeddings.cpu().numpy()
        ).flatten()
        euclidean_distances_normalized = euclidean_distances_to_choices / np.max(euclidean_distances_to_choices)

        # Combine cosine similarity and inverse Euclidean distance with weights
        combined_scores = cosine_similarities - distance_weight * euclidean_distances_normalized

        # Select the answer with the highest combined score
        predicted_index = int(np.argmax(combined_scores))
        predicted_answer = choice_list[predicted_index]
        is_correct = predicted_answer == actual_answer

        # Explanation of prediction
        explanation = (f"The predicted answer was chosen due to the highest cosine similarity of "
                       f"{cosine_similarities[predicted_index]:.4f} with the generated text embedding.")

        # Append the result
        all_results.append({
            'Question ID': question_id,
            'Question': question,
            'Predicted Answer': predicted_answer,
            'Actual Answer': actual_answer,
            'Correct': is_correct,
            'Explanation of Prediction': explanation
        })

    # Save detailed results to CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f'word_test_predictions_{mode}.csv', index=False) 
        print(f"Prediction details for {mode} learning saved to 'word_test_predictions_{mode}.csv'.")
    else:
        print("No results to save. Ensure data processing is working correctly.")

# Run the function for zero-shot, one-shot, and three-shot learning
for mode in ["zero-shot", "one-shot", "three-shot"]:
    print(f"\nProcessing mode: {mode}")
    process_mode(mode)
