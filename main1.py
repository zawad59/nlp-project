import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test data
train_data = np.load('SP_train.npy', allow_pickle=True)
test_data = np.load('SP_test.npy', allow_pickle=True)
test_answers = np.load('SP_test_answer.npy', allow_pickle=True)

# Extract only the correct answer indices from test_answers
test_answer_indices = [int(answer[1]) for answer in test_answers]  # Second element is the correct answer index

# Prepare training data
train_texts = []
train_labels = []
for item in train_data:
    question = item['question']
    choice_list = item['choice_list']
    correct_index = item['label']  # Index of the correct answer
    context = f"{question} Choices: {', '.join(choice_list)}"
    
    train_texts.append(context)
    train_labels.append(correct_index)

# Prepare test data
test_texts = []
for item in test_data:
    question = item['question']
    choice_list = item['choice_list']
    context = f"{question} Choices: {', '.join(choice_list)}"
    
    test_texts.append(context)

# Initialize the tokenizer and model for text generation with Llama
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Function to generate predictions and compare with true labels
def generate_predictions_and_evaluate(texts, true_indices):
    predicted_labels = []
    for i, context in enumerate(texts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
        ]
        
        # Generate response
        output = pipe(messages, max_new_tokens=50)
        
        # Extract generated text safely
        generated_text = ""
        if isinstance(output, list) and len(output) > 0:
            generated_content = output[0].get("generated_text", "")
            if isinstance(generated_content, list):
                generated_text = " ".join([str(part) for part in generated_content])
            elif isinstance(generated_content, str):
                generated_text = generated_content
        
        # Ensure generated_text is a single string
        generated_text = str(generated_text)

        # Extract predicted answer index by matching choices
        choices = context.split("Choices: ")[1].split(", ")
        predicted_label = -1  # Default to -1 if no match is found
        for idx, choice in enumerate(choices):
            if choice.lower() in generated_text.lower():
                predicted_label = idx
                break
        predicted_labels.append(predicted_label)

    # Calculate accuracy and F1 score based on predicted vs true labels
    accuracy = accuracy_score(true_indices, predicted_labels)
    f1 = f1_score(true_indices, predicted_labels, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    return predicted_labels

# Generate predictions and evaluate on test data
predicted_labels = generate_predictions_and_evaluate(test_texts, test_answer_indices)

# Save predictions to a CSV file for review
df_predictions = pd.DataFrame({
    'Question': test_texts,
    'True Label': test_answer_indices,
    'Predicted Label': predicted_labels
})
df_predictions.to_csv('predictions.csv', index=False)
print("Predicted labels saved to predictions.csv.")
