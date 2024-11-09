import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Ensure the pad token is set (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Function to prepare training data
def preprocess_data(data):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    processed_data = []
    for item in data:
        text = item['text'].lower()
        words = word_tokenize(text)
        filtered_words = [w for w in words if w not in stop_words]
        stemmed_words = [stemmer.stem(w) for w in filtered_words]
        processed_text = ' '.join(stemmed_words)
        processed_data.append({
            'text': processed_text,
            'label': item['label']
        })
    return processed_data

# Function to generate predictions and evaluate results
def generate_predictions_and_evaluate(texts, true_indices, true_answers, test_ids):
    results_data = []
    
    for i, context in enumerate(texts):
        question_part = context.split(" Choices: ")[0]
        choices = context.split("Choices: ")[1].split(", ")
        
        # Generate prediction
        prompt = f"Question: {question_part}\nChoices: {', '.join(choices)}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs['input_ids'].to(device),
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = generated_text.split("Answer:")[-1].strip()
        
        predicted_index = choices.index(model_answer) if model_answer in choices else -1
        predicted_answer = choices[predicted_index] if predicted_index != -1 else "None"
        
        is_correct = predicted_index == true_indices[i]
        
        result = {
            'Question_ID': test_ids[i],
            'Question_Text': question_part,
            'Correct_Answer': true_answers[i],
            'Predicted_Answer': predicted_answer,
            'Is_Correct': "Yes" if is_correct else "No"
        }
        results_data.append(result)
    
    predictions = [item['Predicted_Answer'] for item in results_data]
    accuracy = accuracy_score(true_indices, [choices.index(ans) if ans in choices else -1 for ans in predictions])
    f1 = f1_score(true_indices, [choices.index(ans) if ans in choices else -1 for ans in predictions], average='weighted')
    
    return {
        'results_data': results_data,
        'accuracy': accuracy,
        'f1_score': f1
    }

# Function to evaluate and save results
def evaluate_and_save_results(test_texts, test_labels, actual_answers, test_ids):
    os.makedirs('results', exist_ok=True)
    
    print("Running evaluation...")
    results = generate_predictions_and_evaluate(test_texts, test_labels, actual_answers, test_ids)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results['results_data'])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'results/evaluation_results_{timestamp}.csv', index=False)
    
    print(f"\nResults Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    return results

# Function to analyze saved CSV results
def analyze_csv_results(csv_path):
    df = pd.read_csv(csv_path)
    print(f"\nAnalysis of {csv_path}")
    print("-" * 50)
    print(f"Total questions: {len(df)}")
    print(f"Correct predictions: {sum(df['Is_Correct'] == 'Yes')}")
    print(f"Incorrect predictions: {sum(df['Is_Correct'] == 'No')}")
    print(f"Accuracy: {(sum(df['Is_Correct'] == 'Yes') / len(df)):.4f}")
    
    print("\nSample of incorrect predictions:")
    incorrect_samples = df[df['Is_Correct'] == 'No'].head(3)
    for _, row in incorrect_samples.iterrows():
        print("\nQuestion:", row['Question_Text'])
        print("Correct Answer:", row['Correct_Answer'])
        print("Predicted Answer:", row['Predicted_Answer'])

# Load test data
def load_test_data():
    test_texts = [
        "What is the capital of France? Choices: Paris, London, Berlin, Madrid",
        "Who wrote '1984'? Choices: Orwell, Huxley, Dickens, Tolkien"
    ]
    test_labels = [0, 0]
    actual_answers = ["Paris", "Orwell"]
    test_ids = [1, 2]
    return test_texts, test_labels, actual_answers, test_ids

if __name__ == "__main__":
    # Load test data
    test_texts, test_labels, actual_answers, test_ids = load_test_data()
    
    # Evaluate and save results
    results = evaluate_and_save_results(test_texts, test_labels, actual_answers, test_ids)
    
    # Analyze the most recent results
    results_dir = 'results'
    latest_file = max([f for f in os.listdir(results_dir) if f.startswith('evaluation_results')],
                      key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    
    analyze_csv_results(os.path.join(results_dir, latest_file))
