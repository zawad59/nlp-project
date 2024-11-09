import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from datasets import Dataset as HFDataset

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess training data
def preprocess_data(data):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    processed_data = []
    
    for item in data:
        text = item['text'].lower()
        words = word_tokenize(text)
        filtered_words = [w for w in words if w not in stop_words]
        stemmed_words = [stemmer.stem(w) for w in filtered_words]
        processed_text = ' '.join(stemmed_words)
        processed_data.append({
            'text': processed_text,
            'label': item['label']
        })
    
    return processed_data

# Custom Dataset for training
class QADataset(Dataset):
    def __init__(self, processed_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in processed_data:
            # Create training example format
            text = f"Text: {item['text']}\nLabel: {item['label']}\n\n"
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Load and prepare data
print("Loading and preprocessing data...")
train_data = np.load('WP_train 1.npy', allow_pickle=True)
test_data = np.load('WP_test 1.npy', allow_pickle=True)

# Preprocess training data
processed_train_data = preprocess_data(train_data)

# Prepare test data
test_texts = []
test_labels = []
test_ids = []
actual_answers = []

for item in test_data:
    question = item['question']
    choice_list = item['choice_list']
    context = f"{question} Choices: {', '.join(choice_list)}"
    test_texts.append(context)
    test_labels.append(item['label'])
    test_ids.append(item['id'])
    actual_answers.append(choice_list[item['label']])

# Initialize models and tokenizer
print("Initializing models...")
model_id = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Create dataset for training
train_dataset = QADataset(processed_train_data, tokenizer)

# Convert to HuggingFace dataset format
hf_train_dataset = HFDataset.from_dict({
    'input_ids': [example['input_ids'].numpy() for example in train_dataset],
    'attention_mask': [example['attention_mask'].numpy() for example in train_dataset]
})

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_qa_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
)

# Initialize trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train_dataset,
    data_collator=data_collator,
)

# Train the model
print("Training model...")
trainer.train()

# Save the trained model
print("Saving trained model...")
trainer.save_model("./gpt2_qa_model_final")

# Initialize SentenceTransformer for similarity calculations
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Initialize pipeline with trained model
pipe = pipeline(
    "text-generation",
    model="./gpt2_qa_model_final",
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Example for one-shot prompt
examples = [
    {"question": "A teacher in an orphanage spanked children, and no parents objected. Why?", 
     "answer": "There were no parents in the orphanage."}
]

def create_prompt(mode, question, choices):
    if mode == "zero-shot":
        prompt = f"Question: {question}\nSelect the answer that best fits the question:\n{', '.join(choices)}\nAnswer:"
    elif mode == "one-shot":
        example = examples[0]
        prompt = (f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
                 f"Question: {question}\nSelect the answer that best fits the question:\n{', '.join(choices)}\nAnswer:")
    return prompt

def generate_predictions_and_evaluate(texts, true_indices, true_answers, mode):
    predicted_answers = []
    explanations = []
    match_results = []
    
    for i, context in enumerate(texts):
        question_part = context.split(" Choices: ")[0]
        choices = context.split("Choices: ")[1].split(", ")
        
        prompt = create_prompt(mode, question_part, choices)
        output = pipe(prompt, max_new_tokens=100, num_return_sequences=1)[0]
        generated_text = output['generated_text']
        
        model_answer = generated_text.split("Answer:")[-1].strip()
        
        answer_embedding = embedder.encode(model_answer, convert_to_tensor=True)
        choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(answer_embedding, choice_embeddings)[0]
        predicted_index = torch.argmax(similarities).item()
        predicted_answer = choices[predicted_index]
        
        predicted_answers.append(predicted_answer)
        
        explanation = f"""
Question: {question_part}
Model's raw answer: {model_answer}
Available choices: {', '.join(choices)}
Selected answer: {predicted_answer}
Confidence scores: {[f"{sim:.4f}" for sim in similarities]}
Correct answer: {true_answers[i]}
"""
        explanations.append(explanation)
        match_results.append(predicted_index == true_indices[i])
    
    accuracy = accuracy_score(true_indices, [choices.index(ans) for ans in predicted_answers])
    f1 = f1_score(true_indices, [choices.index(ans) for ans in predicted_answers], average='weighted')
    
    return {
        'predicted_answers': predicted_answers,
        'explanations': explanations,
        'match_results': match_results,
        'accuracy': accuracy,
        'f1_score': f1
    }

def evaluate_both_approaches(test_texts, test_labels, actual_answers):
    print("Running Zero-shot evaluation...")
    zero_shot_results = generate_predictions_and_evaluate(
        test_texts, test_labels, actual_answers, mode="zero-shot"
    )
    
    print("Running One-shot evaluation...")
    one_shot_results = generate_predictions_and_evaluate(
        test_texts, test_labels, actual_answers, mode="one-shot"
    )
    
    print("\nResults Summary:")
    print(f"Zero-shot Accuracy: {zero_shot_results['accuracy']:.4f}")
    print(f"Zero-shot F1 Score: {zero_shot_results['f1_score']:.4f}")
    print(f"One-shot Accuracy: {one_shot_results['accuracy']:.4f}")
    print(f"One-shot F1 Score: {one_shot_results['f1_score']:.4f}")
    
    return {
        'zero_shot': zero_shot_results,
        'one_shot': one_shot_results
    }

def analyze_results(results):
    for approach in ['zero_shot', 'one_shot']:
        print(f"\nDetailed Analysis for {approach}:")
        
        for i in range(min(5, len(results[approach]['explanations']))):
            print(f"\nExample {i+1}:")
            print(results[approach]['explanations'][i])
        
        incorrect_predictions = [i for i, match in enumerate(results[approach]['match_results']) if not match]
        print(f"\nNumber of incorrect predictions: {len(incorrect_predictions)}")
        
        if incorrect_predictions:
            print("\nSample of incorrect predictions:")
            for i in incorrect_predictions[:3]:
                print(results[approach]['explanations'][i])

# Run evaluation and analysis
print("Running evaluation...")
results = evaluate_both_approaches(test_texts, test_labels, actual_answers)
analyze_results(results)