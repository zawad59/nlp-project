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
        correct_answer = item['choice_list'][item['label']]
        cleaned_sentences = []

        sentences = sent_tokenize(question)
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

# Custom Trainer class to handle the compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override the compute_loss method to handle unexpected arguments.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)
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
    load_best_model_at_end=True,
    report_to="none"
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

# Fine-tune the model
print("Starting training...")
trainer.train()
trainer.save_model("./gpt2_lora_best_model_SP")

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_best_model_SP").to(device)

# Function to generate answers using the fine-tuned model
def generate_answer(question):
    """
    Generate an answer using the fine-tuned model.
    The question already contains the choices embedded in SP data.
    """
    # Format the prompt with the question only
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the predicted answer
    if "Answer:" in generated_text:
        predicted_answer = generated_text.split("Answer:")[-1].strip()
    else:
        predicted_answer = generated_text.strip()

    # Clean up the predicted answer by removing any choices or extra text
    predicted_answer = predicted_answer.split('\n')[0].strip()

    # Handle edge cases where the model generates extra text
    if len(predicted_answer.split()) > 5:  # Assuming answers are typically short
        predicted_answer = predicted_answer.split('.')[0].strip()  # Keep the first sentence

    return predicted_answer


# Evaluate on the test set and save predictions to CSV
def evaluate_on_test(test_data):
    predictions = []
    correct_predictions = 0
    for idx, item in enumerate(test_data):
        question = item['text']
        correct_answer = item['correct_answer']

        predicted_answer = generate_answer(question)
        is_correct = "yes" if predicted_answer == correct_answer else "no"
        predictions.append({
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
    return predictions

def save_predictions_to_csv(results, filename="prediction_results_SP_gpt2.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question Text", "Predicted Answer", "Correct Answer", "Correct?"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to {filename}")

# Run evaluation and save results to CSV
results = evaluate_on_test(processed_test_data)
save_predictions_to_csv(results)
