import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from sentence_transformers import SentenceTransformer, util
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

# Function to preprocess training data
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

# Custom Dataset for training
class QADataset(Dataset):
    def __init__(self, processed_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for item in processed_data:
            question = item['question']
            choices = item['choices']
            correct_answer = choices[item['label']]
            training_text = (
                f"Question: {question}\n"
                f"Choices: {', '.join(choices)}\n"
                f"Answer: {correct_answer}\n\n"
            )
            encoded = tokenizer(
                training_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': encoded['input_ids'].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Function to prepare data
def prepare_training_data(train_data):
    processed_data = []
    for item in train_data:
        text = item['text']
        label = item['label']
        question = text.split('?')[0] + '?'
        processed_item = {
            'question': question,
            'choices': ['choice1', 'choice2', 'choice3', 'choice4'],
            'label': label
        }
        processed_data.append(processed_item)
    return processed_data

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_qa_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_dir='./logs',
    learning_rate=5e-5,
    fp16=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training function
def train_model(train_data, model, tokenizer):
    processed_data = prepare_training_data(train_data)
    train_dataset = QADataset(processed_data, tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model("./gpt2_qa_model_final")
    return trainer.model

# Evaluation function
def validate_training(model, tokenizer, test_question):
    prompt = f"Question: {test_question['question']}\nChoices: {', '.join(test_question['choices'])}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'].to(device),
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load and prepare test data
def load_test_data():
    """
    Load or define test data here.
    Replace the below list with actual data loading logic.
    """
    test_texts = [
        "What is the capital of France? Choices: Paris, London, Berlin, Madrid",
        "Who wrote '1984'? Choices: Orwell, Huxley, Dickens, Tolkien"
    ]
    test_labels = [0, 0]  # Indices of correct answers
    actual_answers = ["Paris", "Orwell"]
    test_ids = [1, 2]
    return test_texts, test_labels, actual_answers, test_ids

if __name__ == "__main__":
    # Load test data
    test_texts, test_labels, actual_answers, test_ids = load_test_data()

    # Run evaluation and save results
    results = evaluate_and_save_results(test_texts, test_labels, actual_answers, test_ids)

    # Analyze results
    results_dir = 'results'
    latest_file = max([f for f in os.listdir(results_dir) if f.startswith('zero_shot_results')], 
                     key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    analyze_csv_results(os.path.join(results_dir, latest_file))
