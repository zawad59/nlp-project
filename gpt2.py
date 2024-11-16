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

# Initialize models
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Load SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load datasets
train_data = np.load('SP_train 1.npy', allow_pickle=True)
dev_data = np.load('SP_dev 1.npy', allow_pickle=True)
test_data = np.load('SP_test 1.npy', allow_pickle=True)

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
    tokens["labels"] = tokens["input_ids"].copy()  # Set the correct 'labels' key
    return tokens


# Tokenize datasets
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


# Custom Trainer class to handle the compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = labels.view(-1)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
trainer.save_model("./gpt2_lora_best_model")

# Load best model for testing
model = AutoModelForCausalLM.from_pretrained("./gpt2_lora_best_model").to(device)


# Function to generate answers using the fine-tuned model
# Function to generate answers using the fine-tuned model
def generate_answer(question, choices):
    """
    Generate an answer using the fine-tuned model.
    The prompt is structured to be clear for the model to generate the correct answer.
    """
    # Format the prompt to clearly separate the question and the choices
    prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"

    # Generate response from the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,  # Limit the response length
        temperature=0.7,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the predicted answer by splitting at "Answer:"
    if "Answer:" in generated_text:
        predicted_answer = generated_text.split("Answer:")[-1].strip()
    else:
        predicted_answer = generated_text.strip()

    # Ensure the predicted answer is one of the given choices
    predicted_answer = predicted_answer.split('\n')[0]  # Only take the first line of the answer
    predicted_answer = predicted_answer.strip()

    # If the predicted answer is not exactly one of the choices, refine it
    if predicted_answer not in choices:
        predicted_answer = refine_prediction_with_similarity(predicted_answer, choices)

    return predicted_answer


def refine_prediction_with_similarity(generated_answer, choices):
    """
    Refine the predicted answer using cosine similarity with the choices.
    This ensures that even if the model output is slightly different, we select the closest match.
    """
    # Generate embeddings for the choices and the generated answer
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]

    # Select the choice with the highest similarity score
    best_index = torch.argmax(cosine_similarities).item()
    return choices[best_index]


# Function to evaluate on the test set
def evaluate_on_test(test_data):
    predictions = []
    correct_predictions = 0
    for idx, item in enumerate(test_data):
        question = item['text']
        choices = item['choices']
        true_label = item['label']
        correct_answer = choices[true_label]

        # Generate the predicted answer
        predicted_answer = generate_answer(question, choices)

        # Check if predicted answer is correct
        is_correct = "yes" if predicted_answer == correct_answer else "no"
        predictions.append({
            "Question ID": idx + 1,
            "Actual Question Text": question,
            "Choices": ', '.join(choices),
            "Predicted Answer": predicted_answer,
            "Correct Answer": correct_answer,
            "Predicted == Correct": is_correct
        })
        if is_correct == "yes":
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Test Accuracy: {accuracy:.4f}")
    return predictions


def save_predictions_to_csv(predictions, filename="prediction_results_WP_gpt2.csv"):
    """
    Save the predictions to a CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Actual Question Text", "Choices",
                                                  "Predicted Answer", "Correct Answer", "Predicted == Correct"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions saved to {filename}")


# Run evaluation and save results to CSV
predictions = evaluate_on_test(processed_test_data)
save_predictions_to_csv(predictions)
