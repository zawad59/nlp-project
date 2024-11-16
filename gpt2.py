import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import os
from datasets import Dataset as HFDataset

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load training and development datasets
train_file_path = 'WP_train 1.npy'
data = np.load(train_file_path, allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess the training data
def preprocess_gpt2_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]

        sentences = sent_tokenize(question)
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)

        cleaned_question = ' '.join(cleaned_sentences)
        training_text = (
            f"Question: {cleaned_question}\n"
            f"Choices: {', '.join(choices)}\n"
            f"Answer: {correct_answer}\n\n"
        )
        processed_data.append({'text': training_text, 'label': item['label'], 'choices': choices})
    return processed_data

processed_data = preprocess_gpt2_data(data)

# Split data into train, validation, and test sets
train_data, test_data = train_test_split(processed_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = HFDataset.from_list(train_data)
val_dataset = HFDataset.from_list(val_data)
test_dataset = HFDataset.from_list(test_data)

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the text and assign the input_ids to the labels key
    tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Set labels as input_ids
    return tokens


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# LoRA fine-tuning configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_lora_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=3e-5,
    weight_decay=0.001,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)


# Fine-tune the model
print("Starting LoRA fine-tuning...")
trainer.train()
trainer.save_model("./gpt2_lora_best_model")
tokenizer.save_pretrained("./gpt2_lora_best_model")

# Load the fine-tuned model and tokenizer
def load_finetuned_model():
    model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_best_model").to(device)
    return model

def load_finetuned_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./gpt2_lora_best_model")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

model = load_finetuned_model()
tokenizer = load_finetuned_tokenizer()

# Function to generate answers
def generate_answer(question, choices):
    prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Answer:")[-1].strip()

# Evaluate the model on the test set
def evaluate_model(test_data):
    correct_predictions = 0
    total_predictions = len(test_data)

    for item in test_data:
        question = item['text']
        choices = item['choices']
        true_label = item['label']

        generated_answer = generate_answer(question, choices)
        if generated_answer in choices:
            predicted_label = choices.index(generated_answer)
        else:
            predicted_label = -1

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.4f}")

# Run evaluation
print("Evaluating model on test set...")
evaluate_model(test_data)
