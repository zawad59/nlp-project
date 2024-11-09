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
import os

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
            # Format training example as a QA pair
            question = item['question']
            choices = item['choices']
            correct_answer = choices[item['label']]
            
            training_text = (
                f"Question: {question}\n"
                f"Choices: {', '.join(choices)}\n"
                f"Answer: {correct_answer}\n\n"
            )
            
            encoded = tokenizer(
                training_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': encoded['input_ids'].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def prepare_training_data(train_data):
    """
    Prepare training data in the correct format for QA
    """
    processed_data = []
    
    for item in train_data:
        # Extract question and answer from the text
        text = item['text']
        label = item['label']
        
        # Split into question and choices if needed
        question_parts = text.split('?')[0] + '?'
        
        # Create a format matching your test data
        processed_item = {
            'question': question_parts,
            'choices': ['choice1', 'choice2', 'choice3', 'choice4'],  # Replace with actual choices
            'label': label
        }
        
        processed_data.append(processed_item)
    
    return processed_data

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
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    fp16=True,
    prediction_loss_only=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

def train_model(train_data, model, tokenizer):
    # Prepare data
    processed_data = prepare_training_data(train_data)
    train_dataset = QADataset(processed_data, tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the trained model
    print("Saving model...")
    trainer.save_model("./gpt2_qa_model_final")
    
    return trainer.model

def validate_training(model, tokenizer, test_question):
    """
    Validate the model's training by testing on a sample question
    """
    prompt = f"Question: {test_question['question']}\nChoices: {', '.join(test_question['choices'])}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_predictions_and_evaluate(texts, true_indices, true_answers, test_ids, mode):
    results_data = []
    
    for i, context in enumerate(texts):
        question_part = context.split(" Choices: ")[0]
        choices = context.split("Choices: ")[1].split(", ")
        
        # Generate prediction
        prompt = create_prompt(mode, question_part, choices)
        output = pipe(prompt, max_new_tokens=100, num_return_sequences=1)[0]
        generated_text = output['generated_text']
        
        model_answer = generated_text.split("Answer:")[-1].strip()
        
        answer_embedding = embedder.encode(model_answer, convert_to_tensor=True)
        choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(answer_embedding, choice_embeddings)[0]
        
        predicted_index = torch.argmax(similarities).item()
        predicted_answer = choices[predicted_index]
        
        is_correct = predicted_index == true_indices[i]
        
        explanation = f"Model generated answer: '{model_answer}'\n"
        explanation += f"Similarity scores with choices:\n"
        for j, (choice, sim) in enumerate(zip(choices, similarities)):
            explanation += f"- {choice}: {sim:.4f}\n"
        explanation += f"\nSelected '{predicted_answer}' due to highest similarity score: {max(similarities):.4f}"
        
        result = {
            'Question_ID': test_ids[i],
            'Question_Text': question_part,
            'Correct_Answer': true_answers[i],
            'Predicted_Answer': predicted_answer,
            'Is_Correct': "Yes" if is_correct else "No",
            'Explanation': explanation
        }
        results_data.append(result)
    
    predictions = [item['Predicted_Answer'] for item in results_data]
    accuracy = accuracy_score(true_indices, [choices.index(ans) for ans in predictions])
    f1 = f1_score(true_indices, [choices.index(ans) for ans in predictions], average='weighted')
    
    return {
        'results_data': results_data,
        'accuracy': accuracy,
        'f1_score': f1
    }

def evaluate_and_save_results(test_texts, test_labels, actual_answers, test_ids):
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run evaluations
    print("Running Zero-shot evaluation...")
    zero_shot_results = generate_predictions_and_evaluate(
        test_texts, test_labels, actual_answers, test_ids, mode="zero-shot"
    )
    
    print("Running One-shot evaluation...")
    one_shot_results = generate_predictions_and_evaluate(
        test_texts, test_labels, actual_answers, test_ids, mode="one-shot"
    )
    
    # Create DataFrames and save to CSV
    zero_shot_df = pd.DataFrame(zero_shot_results['results_data'])
    one_shot_df = pd.DataFrame(one_shot_results['results_data'])
    
    # Save to CSV files
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    zero_shot_df.to_csv(f'results/zero_shot_results_{timestamp}.csv', index=False)
    one_shot_df.to_csv(f'results/one_shot_results_{timestamp}.csv', index=False)
    
    # Print summary metrics
    print("\nResults Summary:")
    print(f"Zero-shot Accuracy: {zero_shot_results['accuracy']:.4f}")
    print(f"Zero-shot F1 Score: {zero_shot_results['f1_score']:.4f}")
    print(f"One-shot Accuracy: {one_shot_results['accuracy']:.4f}")
    print(f"One-shot F1 Score: {one_shot_results['f1_score']:.4f}")
    
    # Create and save summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score'],
        'Zero-shot': [zero_shot_results['accuracy'], zero_shot_results['f1_score']],
        'One-shot': [one_shot_results['accuracy'], one_shot_results['f1_score']]
    })
    summary_stats.to_csv(f'results/summary_statistics_{timestamp}.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"- results/zero_shot_results_{timestamp}.csv")
    print(f"- results/one_shot_results_{timestamp}.csv")
    print(f"- results/summary_statistics_{timestamp}.csv")
    
    return {
        'zero_shot': zero_shot_results,
        'one_shot': one_shot_results
    }

def analyze_csv_results(csv_path):
    """
    Analyze results from a saved CSV file
    """
    df = pd.read_csv(csv_path)
    
    # Print basic statistics
    print(f"\nAnalysis of {csv_path}")
    print("-" * 50)
    print(f"Total questions: {len(df)}")
    print(f"Correct predictions: {sum(df['Is_Correct'] == 'Yes')}")
    print(f"Incorrect predictions: {sum(df['Is_Correct'] == 'No')}")
    print(f"Accuracy: {(sum(df['Is_Correct'] == 'Yes') / len(df)):.4f}")
    
    # Sample of incorrect predictions
    print("\nSample of incorrect predictions:")
    incorrect_samples = df[df['Is_Correct'] == 'No'].head(3)
    for _, row in incorrect_samples.iterrows():
        print("\nQuestion:", row['Question_Text'])
        print("Correct Answer:", row['Correct_Answer'])
        print("Predicted Answer:", row['Predicted_Answer'])
        print("Explanation:", row['Explanation'])
    
    return df

if __name__ == "__main__":
    # Assuming test_texts, test_labels, actual_answers, and test_ids are already defined
    
    # Run evaluation and save results
    results = evaluate_and_save_results(test_texts, test_labels, actual_answers, test_ids)
    
    # Get the most recent results files
    results_dir = 'results'
    latest_file = max([f for f in os.listdir(results_dir) if f.startswith('zero_shot_results')], 
                     key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    
    # Analyze the results
    results_df = analyze_csv_results(os.path.join(results_dir, latest_file))
