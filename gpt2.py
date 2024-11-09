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
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
#import porterStemmer
from nltk.stem import PorterStemmer
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load training and dev data
train_file_path = 'WP_train 1.npy'
dev_file_path = 'WP_test 1.npy'
train_data = np.load(train_file_path, allow_pickle=True)
dev_data = np.load(dev_file_path, allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Step 1: Preprocess Training Data
def preprocess_gpt2_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]
        
        # Sentence tokenization, stopword removal, and stemming
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
from datasets import Dataset as HFDataset
train_dataset = HFDataset.from_dict({"text": processed_train_data})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
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
trainer.save_model("./gpt2_qa_finetuned_model")

# Step 3: Generate and Refine Answers
def generate_answer(question, choices):
    prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=200, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Answer:")[-1].strip()

def refine_prediction_with_embeddings(question, choices, generated_answer):
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(similarities).item()
    best_answer = choices[best_index]
    return best_answer, similarities

# Step 4: Evaluate on Dev Data
def evaluate_model(dev_data):
    correct_predictions = 0
    total_predictions = len(dev_data)
    refined_results = []
    
    for item in dev_data:
        question = item['question']
        choices = item['choice_list']
        true_answer = item['answer']
        
        # Generate answer using GPT-2
        generated_answer = generate_answer(question, choices)
        
        # Refine using sentence embeddings
        refined_answer, similarities = refine_prediction_with_embeddings(question, choices, generated_answer)
        is_correct = (refined_answer == true_answer)
        correct_predictions += is_correct
        
        refined_results.append({
            'Question_ID': item['id'],
            'Question_Text': question,
            'Generated_Answer': generated_answer,
            'Refined_Answer': refined_answer,
            'Correct_Answer': true_answer,
            'Is_Correct': "Yes" if is_correct else "No",
            'Cosine_Similarities': similarities.tolist()
        })
    
    accuracy = correct_predictions / total_predictions
    print(f"Refined Accuracy: {accuracy:.4f}")
    
    # Save results to CSV
    refined_results_df = pd.DataFrame(refined_results)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    refined_results_df.to_csv(f'results/refined_evaluation_{timestamp}_onTest.csv', index=False)

os.makedirs('results', exist_ok=True)
evaluate_model(dev_data)
