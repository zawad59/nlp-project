import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
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

# Initialize Phi-1_5 Model
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Ensure pad token is set
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load datasets
train_data = np.load('SP_train 1.npy', allow_pickle=True)
dev_data = np.load('SP_dev 1.npy', allow_pickle=True)
test_data = np.load('SP_test 1.npy', allow_pickle=True)


# Preprocess the SP dataset
def preprocess_sp_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        choices = item['choice_list']
        correct_answer = choices[item['label']]
        cleaned_question = question.lower()  # Preserve original context

        # Create training text
        training_text = (
                f"Question: {cleaned_question}\n"
                f"Choices:\n" + "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:"
        )
        processed_data.append({'text': training_text, 'choices': choices, 'label': item['label']})
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


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                            remove_columns=["text", "choices", "label"])
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["text", "choices", "label"])

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


# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training arguments
training_args = TrainingArguments(
    output_dir="./phi1_5_finetuned_SP",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train and save the best model
trainer.train()
trainer.save_model("./phi1_5_best_model_SP")
model = AutoModelForCausalLM.from_pretrained("./phi1_5_best_model_SP").to(device)


# Generate answers
def generate_answer(question, choices):
    choices_text = "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"Question: {question}\nChoices:\n{choices_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Answer:")[-1].strip()


def refine_prediction_with_similarity(generated_answer, choices):
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]


def evaluate_on_test(test_data):
    predictions = []
    for idx, item in enumerate(test_data):
        question = item['text']
        choices = item['choices']
        true_label = item['label']
        correct_answer = choices[true_label]

        generated_answer = generate_answer(question, choices)
        refined_answer = refine_prediction_with_similarity(generated_answer, choices)

        predictions.append({
            "Question ID": idx + 1,
            "Question": question,
            "Choices": ', '.join(choices),
            "Predicted Answer": refined_answer,
            "Correct Answer": correct_answer,
            "Predicted == Correct": "yes" if refined_answer == correct_answer else "no"
        })
    return predictions


def save_predictions_to_csv(predictions, filename="phi1_5_predictions.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question", "Choices", "Predicted Answer",
                                                  "Correct Answer", "Predicted == Correct"])
        writer.writeheader()
        writer.writerows(predictions)


# Evaluate and save results
predictions = evaluate_on_test(processed_test_data)
save_predictions_to_csv(predictions)
print("Results saved to phi1_5_predictions.csv")
