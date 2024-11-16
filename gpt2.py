#random
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
dev_file_path = 'WP_test 1.npy'
train_data = np.load(train_file_path, allow_pickle=True)
dev_data = np.load(dev_file_path, allow_pickle=True)

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
        processed_data.append(training_text)
    return processed_data

# Preprocess training data
processed_train_data = preprocess_gpt2_data(train_data)

# Convert to Hugging Face Dataset
train_dataset = HFDataset.from_dict({"text": processed_train_data})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
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
    num_train_epochs=100,
    per_device_train_batch_size=8,
    save_steps=500,
    logging_steps=200,
    learning_rate=3e-5,
    weight_decay=0.001,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator
)

print("Starting LoRA fine-tuning...")
trainer.train()
trainer.save_model("./gpt2_lora_finetuned_model")
tokenizer.save_pretrained("./gpt2_lora_finetuned_model")

# Load the fine-tuned model and tokenizer for evaluation
def load_finetuned_model():
    model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_finetuned_model").to(device)
    return model

def load_finetuned_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./gpt2_lora_finetuned_model")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Reload model and tokenizer
model = load_finetuned_model()
tokenizer = load_finetuned_tokenizer()


# Function to generate answer with adjusted sampling and attention mask
def generate_answer(question, choices):
    prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate the output with sampling enabled and attention mask set
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Set the attention mask
        max_length=200,
        temperature=0.7,
        do_sample=True,  # Enable sampling for temperature to have an effect
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Answer:")[-1].strip()


# Function to refine prediction using cosine similarity
def refine_prediction_with_embeddings(question, choices, generated_answer):
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]

# Evaluation function
def evaluate_model(dev_data):
    correct_predictions = 0
    total_predictions = len(dev_data)

    for item in dev_data:
        question = item['question']
        choices = item['choice_list']
        true_answer = item['answer']

        generated_answer = generate_answer(question, choices)
        refined_answer = refine_prediction_with_embeddings(question, choices, generated_answer)

        if refined_answer == true_answer:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Refined Accuracy: {accuracy:.4f}")

# Run evaluation
print("Evaluating model...")
evaluate_model(dev_data)
