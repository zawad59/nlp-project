import numpy as np
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score

# Load training and test data for WP dataset
train_data = np.load('WP_train 1.npy', allow_pickle=True)
test_data = np.load('WP_test 1.npy', allow_pickle=True)

# Prepare test data for WP
test_texts = []
test_labels = []
test_ids = []
for item in test_data:
    question = item['question']
    choice_list = item['choice_list']
    context = f"{question} Choices: {', '.join(choice_list)}"
    test_texts.append(context)
    test_labels.append(item['label'])  # Using the label directly as correct answer index
    test_ids.append(item['id'])  # ID for each question

# Initialize SentenceTransformer for similarity calculations
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Initialize the tokenizer and Llama model for text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Predefined examples for one-shot and three-shot prompts
examples = [
    {"question": "A teacher in an orphanage spanked children, and no parents objected. Why?", "answer": "There were no parents in the orphanage."},
    {"question": "A chef cooks every day at home but doesn't get paid. Why?", "answer": "He's cooking for his family, not as a job."},
    {"question": "A man walks out of a store with a cart full of items, but no one stops him. Why?", "answer": "He's an employee taking out trash."}
]

# Function to create prompts for zero-shot, one-shot, and three-shot learning
def create_prompt(mode, question, choices):
    if mode == "zero-shot":
        prompt = f"Question: {question}\nSelect the answer that best fits the question:\n{', '.join(choices)}\nAnswer:"
    elif mode == "one-shot":
        example = examples[0]
        prompt = (f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
                  f"Question: {question}\nSelect the answer that best fits the question:\n{', '.join(choices)}\nAnswer:")
    elif mode == "three-shot":
        prompt = ""
        for example in examples:
            prompt += f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
        prompt += f"Question: {question}\nSelect the answer that best fits the question:\n{', '.join(choices)}\nAnswer:"
    return prompt

# Function to generate predictions, calculate cosine similarity, and provide explanations
def generate_predictions_and_evaluate(texts, true_indices, mode):
    predicted_labels = []
    explanations = []

    for i, context in enumerate(texts):
        question_part = context.split(" Choices: ")[0]
        choices = context.split("Choices: ")[1].split(", ")
        
        # Create prompt based on learning mode
        prompt = create_prompt(mode, question_part, choices)

        # Generate response
        output = pipe(prompt, max_new_tokens=50)
        generated_text = output[0]['generated_text'].strip()

        # Calculate embeddings for generated answer and choices
        generated_embedding = embedder.encode(generated_text, convert_to_tensor=True)
        choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
        
        # Calculate cosine similarities for each choice
        cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings).cpu().numpy().flatten()
        predicted_index = int(np.argmax(cosine_similarities))  # Select the choice with the highest similarity
        predicted_labels.append(predicted_index)
        
        # Explanation based on similarity
        explanation = f"The predicted answer '{choices[predicted_index]}' was selected due to the highest cosine similarity of {cosine_similarities[predicted_index]:.4f} with the generated response."
        explanations.append(explanation)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(true_indices, predicted_labels)
    f1 = f1_score(true_indices, predicted_labels, average='weighted')
    print(f"Mode: {mode} | Accuracy: {accuracy} | F1 Score: {f1}")

    return predicted_labels, explanations

# Run for each mode
modes = ["zero-shot", "one-shot", "three-shot"]
for mode in modes:
    print(f"\nProcessing mode: {mode}")
    predicted_labels, explanations = generate_predictions_and_evaluate(test_texts, test_labels, mode)
    
    # Save predictions to CSV with explanations
    df_predictions = pd.DataFrame({
        'Question ID': test_ids,
        'Question': test_texts,
        'True Label': test_labels,
        'Predicted Label': predicted_labels,
        'Explanation': explanations
    })
    df_predictions.to_csv(f'WP_test_predictions_{mode}_with_explanations.csv', index=False)
    print(f"Predicted labels with explanations for {mode} saved to WP_test_predictions_{mode}_with_explanations.csv.")
