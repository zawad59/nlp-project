import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

# Load the data from SP_dev.npy
data = np.load('WP_dev.npy', allow_pickle=True)

# Group questions by prefix
groups = {}
for item in data:
    group_id = item['id'].split('_')[0]  # Extract the group prefix
    if group_id not in groups:
        groups[group_id] = []
    groups[group_id].append(item)

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
similarity_threshold = 0.85
distance_weight = 0.3  # Adjust this weight to tune the influence of Euclidean distance
batch_size = 5  # Number of groups per time interval

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
    interval_accuracies = []
    all_results = []
    
    # Process each interval
    for start in range(0, len(groups), batch_size):
        interval_correct_predictions = []
        interval_results = []
        
        # Get the groups for this interval
        interval_groups = dict(list(groups.items())[start:start + batch_size])
        
        # Process each group in the interval
        for group_id, items in interval_groups.items():
            # Generate sentence embeddings for the group's questions
            questions = [item['question'] for item in items]
            answers = [item['answer'] for item in items]
            embeddings = embedder.encode(questions)
            predictions = []  # Store all predicted answers in this group

            # Calculate pairwise cosine similarities and process predictions
            for i in range(len(questions)):
                for j in range(i + 1, len(questions)):
                    # Calculate cosine similarity
                    similarity = util.cos_sim(embeddings[i], embeddings[j]).item()

                    # If questions are semantically similar, check if predictions align
                    if similarity > similarity_threshold:
                        prompt = create_prompt(mode, questions[i], items[i]['choice_list'])
                        
                        # Generate response
                        result = pipe(prompt, max_new_tokens=30)
                        generated_text = result[0]['generated_text'].strip()

                        # Calculate similarity and distance between generated text and each choice
                        generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
                        choice_embeddings = embedder.encode(items[i]['choice_list'], convert_to_tensor=True)
                        
                        # Calculate cosine similarities and check dimensions
                        cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings).cpu().numpy().flatten()
                        if cosine_similarities.size != len(items[i]['choice_list']):
                            print(f"Warning: Dimension mismatch in cosine similarity calculation for group {group_id}")
                            continue

                        # Calculate Euclidean distances and check dimensions
                        euclidean_distances_to_choices = euclidean_distances(
                            generated_embedding.cpu().numpy().reshape(1, -1),
                            choice_embeddings.cpu().numpy()
                        ).flatten()
                        if euclidean_distances_to_choices.size != len(items[i]['choice_list']):
                            print(f"Warning: Dimension mismatch in Euclidean distance calculation for group {group_id}")
                            continue

                        # Normalize Euclidean distances for better scaling
                        euclidean_distances_normalized = euclidean_distances_to_choices / np.max(euclidean_distances_to_choices)
                        
                        # Combine cosine similarity and inverse Euclidean distance with weights
                        combined_scores = cosine_similarities - distance_weight * euclidean_distances_normalized

                        # Select the answer with the highest combined score
                        predicted_index = int(np.argmax(combined_scores))
                        predicted_answer = items[i]['choice_list'][predicted_index]
                        predictions.append(predicted_answer)

                        interval_results.append({
                            'Interval': start // batch_size + 1,
                            'Group ID': group_id,
                            'Question 1': questions[i],
                            'Question 2': questions[j],
                            'Cosine Similarity': similarity,
                            'Euclidean Distance': euclidean_distances_normalized[predicted_index],
                            'Combined Score': combined_scores[predicted_index],
                            'Predicted Answer': predicted_answer,
                            'Actual Answer': answers[i]
                        })

            # Majority voting for group-based answer
            if predictions:
                final_prediction = Counter(predictions).most_common(1)[0][0]  # Get the most common answer
                correct_prediction = final_prediction == answers[0]  # Check with the first answer as reference
                interval_correct_predictions.append(correct_prediction)

        # Calculate accuracy for this interval
        interval_accuracy = sum(interval_correct_predictions) / len(interval_correct_predictions) if interval_correct_predictions else 0
        interval_accuracies.append({
            'Interval': start // batch_size + 1,
            'Accuracy': interval_accuracy * 100
        })
        all_results.extend(interval_results)

    # Save interval accuracies to CSV
    df_interval_accuracies = pd.DataFrame(interval_accuracies)
    df_interval_accuracies.to_csv(f'interval_accuracies_{mode}.csv', index=False)
    print(f"Interval accuracies for {mode} learning saved to 'interval_accuracies_{mode}.csv'.")

    # Save detailed results to CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f'interval_predictions_{mode}.csv', index=False)
    print(f"Prediction details with intervals for {mode} learning saved to 'interval_predictions_{mode}.csv'.")

# Run the function for zero-shot, one-shot, and three-shot learning
for mode in ["zero-shot", "one-shot", "three-shot"]:
    print(f"\nProcessing mode: {mode}")
    process_mode(mode)
