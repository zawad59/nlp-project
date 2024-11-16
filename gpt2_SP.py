import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
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
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to be the same as the eos token

# Load the GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Ensure that the model's pad_token_id is set correctly
model.config.pad_token_id = tokenizer.pad_token_id


# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load the SP dataset
train_data = np.load('SP_train 1.npy', allow_pickle=True)
dev_data = np.load('SP_dev 1.npy', allow_pickle=True)
test_data = np.load('SP_test 1.npy', allow_pickle=True)


# Preprocess the SP data
def preprocess_sp_data(data):
    processed_data = []
    for item in data:
        question = item['question']
        correct_answer = item['answer']
        choices = item['choice_list']
        label = item['label']

        choices_text = "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
        training_text = f"Question: {question}\nChoices:\n{choices_text}\nAnswer:"

        processed_data.append({
            'text': training_text,
            'choices': choices,
            'correct_answer': correct_answer,
            'label': label
        })
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
    tokens = tokenizer(
        examples["text"], padding='max_length', truncation=True, max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    tokens["attention_mask"] = tokens["attention_mask"]
    return tokens


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                            remove_columns=["text", "choices", "correct_answer", "label"])
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True,
                                        remove_columns=["text", "choices", "correct_answer", "label"])

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
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_lora_finetuned_SP",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.001,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"
)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
print("Starting training...")
trainer.train()
trainer.save_model("./gpt2_lora_best_model_SP")

# Load the best model for evaluation
model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_best_model_SP").to(device)


def generate_answer(question):
    """
    Generate an answer using the fine-tuned model.
    This function takes care of padding and attention masks.
    """
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate text with more deterministic settings
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=20,  # Restrict the length of the generated answer
        temperature=0.3,
        do_sample=False,  # Use deterministic generation for more focused output
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = generated_text.split("Answer:")[-1].strip()
    print(f"Debug - Raw Generated Answer: {predicted_answer}")  # Debugging statement
    return predicted_answer


def refine_prediction_with_embeddings(generated_answer, choices):
    """
    Use cosine similarity to refine the generated answer.
    """
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]

    # Print debug information for analysis
    print("\nGenerated Answer:", generated_answer)
    print("Cosine Similarities:", cosine_similarities.tolist())
    print("Choices:", choices)

    best_index = torch.argmax(cosine_similarities).item()

    # Check if "None of the above" has a lower similarity threshold
    if choices[best_index].lower() == "none of above" and cosine_similarities[best_index] < 0.6:
        # If "None of the above" is chosen but has low similarity, select the next best option
        second_best_index = torch.topk(cosine_similarities, 2)[1][1].item()
        best_index = second_best_index

    return choices[best_index]


def evaluate_on_test(test_data):
    """
    Evaluate the model on the test set and refine answers using cosine similarity.
    """
    results = []
    correct_predictions = 0
    for idx, item in enumerate(test_data):
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']

        # Generate initial answer
        generated_answer = generate_answer(question)

        # Debugging: Check what the model is generating
        print(f"\nQuestion: {question}")
        print(f"Generated Answer: {generated_answer}")

        # Refine prediction using cosine similarity
        refined_answer = refine_prediction_with_embeddings(generated_answer, choices)
        print(f"Refined Answer: {refined_answer}")

        is_correct = "yes" if refined_answer == correct_answer else "no"

        results.append({
            "Question ID": idx + 1,
            "Question Text": question,
            "Generated Answer": generated_answer,
            "Refined Answer": refined_answer,
            "Correct Answer": correct_answer,
            "Correct?": is_correct
        })

        if is_correct == "yes":
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")
    return results


def save_predictions_to_csv(results, filename="prediction_results_SP_gpt2_fixed.csv"):
    """
    Save the prediction results to a CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question Text", "Generated Answer", "Refined Answer",
                                                  "Correct Answer", "Correct?"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to {filename}")


# Run evaluation and save predictions
results = evaluate_on_test(processed_test_data)
save_predictions_to_csv(results)

