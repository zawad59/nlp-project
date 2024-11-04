import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test data
train_data = np.load('SP_train.npy', allow_pickle=True)
test_data = np.load('SP_test.npy', allow_pickle=True)

# Assuming the arrays are structured as tuples (input, label)
train_texts, train_labels = zip(*train_data)
test_texts, test_labels = zip(*test_data)

# Load a pretrained tokenizer and model
model_name = "Llama-3.2-3B"  # Example LLM; change as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels)))

# Tokenize the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Convert labels to tensor format
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Create a PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    evaluation_strategy="epoch",     
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    num_train_epochs=3,              
    weight_decay=0.01,
)

# Define evaluation metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=test_dataset,           
    compute_metrics=compute_metrics      
)

# Train the model
trainer.train()

# Predict on the test set
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Compare predictions with actual test labels
accuracy = accuracy_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels, average='weighted')

print(f"Test Accuracy: {accuracy}")
print(f"Test F1 Score: {f1}")
'''import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

print(pipe("The key to life is"))'''

'''import numpy as np
data = np.random.normal(0, 1, 100)
np.save('SP_test.npy', data)
data = np.load('data.npy')
print(data)'''
