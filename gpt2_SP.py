import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from datasets import Dataset as HFDataset
import csv

# Download NLTK data
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

# Load the SP dataset
train_data = np.load('SP_train 1.npy', allow_pickle=True)
dev_data = np.load('SP_dev 1.npy', allow_pickle=True)
test_data = np.load('SP_test 1.npy', allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess SP data
def preprocess_sp_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        # Choices are embedded within the question itself in SP data
        correct_answer = item['choice_list'][item['label']]

        sentences = sent_tokenize(question)
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)

        cleaned_question = ' '.join(cleaned_sentences)
        training_text = f"Question: {cleaned_question}\nAnswer:"
        processed_data.append({'text': training_text, 'correct_answer': correct_answer})
    return processed_data

# Preprocess datasets
processed_train_data = preprocess_sp_data(train_data)
processed_dev_data = preprocess_sp_data(dev_data)
processed_test_data = preprocess_sp_data(test_data)

# Convert to Hugging Face Dataset
train_dataset = HFDataset.from_list(processed_train_data)
dev_dataset = HFDataset.from_list(processed_dev_data)
test_dataset = HFDataset.from_list(processed_test_data)

# Tokenize datasets
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "correct_answer"])
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["text", "correct_answer"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# LoRA fine-tuning configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Custom Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override the compute_loss method to handle the custom logic.
        The **kwargs ensures compatibility with any unexpected arguments.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)
        
        # Calculate the cross-entropy loss
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_lora_finetuned_SP",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.001,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

# Initialize the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Start training
print("Starting training...")
trainer.train()
trainer.save_model("./gpt2_lora_best_model_SP")


# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_best_model_SP").to(device)

def generate_answer(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = generated_text.split("Answer:")[-1].strip()
    return predicted_answer

def evaluate_on_test(test_data):
    correct_predictions = 0
    results = []
    for idx, item in enumerate(test_data):
        question = item['text']
        correct_answer = item['correct_answer']

        predicted_answer = generate_answer(question)
        is_correct = "yes" if predicted_answer == correct_answer else "no"
        results.append({
            "Question ID": idx + 1,
            "Question Text": question,
            "Predicted Answer": predicted_answer,
            "Correct Answer": correct_answer,
            "Correct?": is_correct
        })
        if is_correct == "yes":
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")
    return results

def save_predictions_to_csv(results):
    with open("prediction_results_SP_gpt2.csv", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question Text", "Predicted Answer", "Correct Answer", "Correct?"])
        writer.writeheader()
        writer.writerows(results)

results = evaluate_on_test(processed_test_data)
save_predictions_to_csv(results)
