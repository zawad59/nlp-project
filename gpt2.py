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

# Load datasets
train_data = np.load('WP_train 1.npy', allow_pickle=True)
dev_data = np.load('WP_dev 1.npy', allow_pickle=True)
test_data = np.load('WP_test 1.npy', allow_pickle=True)

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

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
            filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
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
processed_dev_data = preprocess_gpt2_data(dev_data)
processed_test_data = preprocess_gpt2_data(test_data)

# Convert to Hugging Face Dataset
train_dataset = HFDataset.from_list(processed_train_data)
dev_dataset = HFDataset.from_list(processed_dev_data)
test_dataset = HFDataset.from_list(processed_test_data)

# Tokenize and ensure correct labels field
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Debug: Print a sample to verify fields
print("Sample tokenized data:", tokenized_train_dataset[0])

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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom compute_loss method that handles additional keyword arguments.
        """
        # Ensure labels are correctly passed
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Flatten the logits and labels for computing cross-entropy loss
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)

        # Compute the loss using cross-entropy
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


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
trainer.save_model("./gpt2_lora_best_model")

# Load the best model for testing
model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_best_model").to(device)

# Evaluate on the test set
def evaluate_on_test(test_data):
    correct_predictions = 0
    for item in test_data:
        question = item['text']
        choices = item['choices']
        true_label = item['label']
        generated_answer = generate_answer(question, choices)

        choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
        generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
        cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
        best_index = torch.argmax(cosine_similarities).item()

        if best_index == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")

# Run evaluation on test set
evaluate_on_test(processed_test_data)
