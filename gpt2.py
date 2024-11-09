import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Load the training and test data
train_file_path = '/mnt/data/WP_train 1.npy'
dev_file_path = '/mnt/data/WP_dev 1.npy'

train_data = np.load(train_file_path, allow_pickle=True)
dev_data = np.load(dev_file_path, allow_pickle=True)

# Initialize PorterStemmer and stopwords set
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Step 1: Preprocess Training Data
def preprocess_gpt2_data(train_data):
    """
    Preprocess the training data:
    - Sentence tokenization
    - Stopword removal
    - Porter stemming
    """
    processed_data = []
    
    for item in train_data:
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]
        
        # Sentence tokenize the question
        sentences = sent_tokenize(question)
        cleaned_sentences = []
        
        for sentence in sentences:
            # Word tokenize, remove stopwords, and apply stemming
            words = word_tokenize(sentence.lower())
            filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)
        
        cleaned_question = ' '.join(cleaned_sentences)
        
        # Construct training text
        training_text = (
            f"Question: {cleaned_question}\n"
            f"Choices: {', '.join(choices)}\n"
            f"Answer: {correct_answer}\n\n"
        )
        processed_data.append(training_text)
    
    return processed_data

# Preprocess the training data
processed_train_data = preprocess_gpt2_data(train_data)

# Convert to Hugging Face Dataset
train_dataset = HFDataset.from_dict({"text": processed_train_data})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 2: Fine-tune GPT-2
training_args = TrainingArguments(
    output_dir="./gpt2_qa_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    logging_steps=500,
    learning_rate=5e-5,
    evaluation_strategy="no",
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")
trainer.save_model("./gpt2_qa_finetuned_model")

# Step 3: Generate Answers for the Test Data
def generate_answers(dev_data):
    results = []
    for item in dev_data:
        question = item['question']
        choices = item['choice_list']
        prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs['input_ids'],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = generated_text.split("Answer:")[-1].strip()
        predicted_index = choices.index(model_answer) if model_answer in choices else -1
        predicted_answer = choices[predicted_index] if predicted_index != -1 else "None"
        
        is_correct = (predicted_index == item['label'])
        
        results.append({
            'Question_ID': item['id'],
            'Question_Text': question,
            'Correct_Answer': item['answer'],
            'Predicted_Answer': predicted_answer,
            'Is_Correct': "Yes" if is_correct else "No"
        })
    
    return results

# Step 4: Evaluate Model Performance
def evaluate_model(dev_data):
    results = generate_answers(dev_data)
    predictions = [res['Predicted_Answer'] for res in results]
    true_answers = [item['answer'] for item in dev_data]
    true_labels = [item['label'] for item in dev_data]

    accuracy = accuracy_score(true_labels, [item['choice_list'].index(pred) if pred in item['choice_list'] else -1 for item, pred in zip(dev_data, predictions)])
    f1 = f1_score(true_labels, [item['choice_list'].index(pred) if pred in item['choice_list'] else -1 for item, pred in zip(dev_data, predictions)], average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'results/evaluation_results_{timestamp}.csv', index=False)
    print(f"Results saved to: results/evaluation_results_{timestamp}.csv")

# Step 5: Run Evaluation
os.makedirs('results', exist_ok=True)
evaluate_model(dev_data)
