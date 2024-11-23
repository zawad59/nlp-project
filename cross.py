import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from datasets import Dataset as HFDataset
import nltk
import csv

# Download required NLTK data
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_data = np.load('SP_train 1.npy', allow_pickle=True)
dev_data = np.load('SP_dev 1.npy', allow_pickle=True)
test_data = np.load('SP_test 1.npy', allow_pickle=True)


# Preprocess the data
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
            filtered_words = [word for word in words if word.isalpha()]
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)

        cleaned_question = ' '.join(cleaned_sentences)
        training_text = (
            f"Question: {cleaned_question}\n"
            f"Choices: {', '.join(choices)}\n"
            f"Answer: {correct_answer}\n\n"
        )
        processed_data.append({'text': training_text, 'choices': choices, 'label': item['label']})
    return processed_data


# Preprocess datasets
processed_train_data = preprocess_gpt2_data(train_data)

# Convert to Hugging Face Dataset
train_dataset = HFDataset.from_list(processed_train_data)


# Tokenize and ensure correct labels field
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Set the correct 'labels' key
    return tokens


# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                            remove_columns=["text", "choices", "label"])

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

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_lora_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_steps=100,
    learning_rate=3e-5,
    weight_decay=0.001,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Fine-tune the model
print("Starting training...")
trainer.train()
trainer.save_model("./gpt2_lora_best_model")


# Function to compute probabilities for the training set
def get_training_probabilities(train_data, model, tokenizer, device):
    """
    Generate probabilities for all answer choices for all questions in the training set.
    """
    all_probabilities = []
    model.eval()  # Ensure the model is in evaluation mode

    for idx, item in enumerate(train_data):
        question = item['text']
        choices = item['choices']

        # Format the input prompt for the model
        prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # Get logits from the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # Only consider logits for the last token

            # Calculate probabilities for each choice
            choice_probabilities = []
            for choice in choices:
                choice_input = tokenizer(choice, return_tensors="pt").input_ids.to(device)
                choice_logits = logits[:, choice_input.squeeze()]
                probabilities = torch.nn.functional.softmax(choice_logits, dim=-1)
                choice_probabilities.append(probabilities.mean().item())

            # Normalize probabilities to sum to 1
            normalized_probabilities = [p / sum(choice_probabilities) for p in choice_probabilities]

        # Append results for the current question
        all_probabilities.append({
            "Question ID": idx + 1,
            "Question": question,
            "Choices": choices,
            "Probabilities": normalized_probabilities
        })

    return all_probabilities


# Compute probabilities for the training set
training_probabilities = get_training_probabilities(processed_train_data, model, tokenizer, device)

# Print probabilities for verification
for prob in training_probabilities:
    print(f"Question ID: {prob['Question ID']}")
    print(f"Probabilities: {prob['Probabilities']}\n")


# Save probabilities to a CSV file
def save_probabilities_to_csv(probabilities, filename="training_probabilities.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question", "Choices", "Probabilities"])
        writer.writeheader()
        for prob in probabilities:
            writer.writerow({
                "Question ID": prob["Question ID"],
                "Question": prob["Question"],
                "Choices": ', '.join(prob["Choices"]),
                "Probabilities": ', '.join(map(str, prob["Probabilities"]))
            })
    print(f"Probabilities saved to {filename}")


save_probabilities_to_csv(training_probabilities)
