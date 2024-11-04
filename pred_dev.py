import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# Load the training and dev data
train_data = np.load('SP_train.npy', allow_pickle=True)
dev_data = np.load('SP_dev.npy', allow_pickle=True)

# Prepare dev data for prediction
dev_question_ids = [item['id'] for item in dev_data]
dev_texts = []
dev_choices = []
dev_actual_answers = []

for item in dev_data:
    question = item['question']
    choice_list = item['choice_list']
    correct_answer = choice_list[item['label']]  # Actual answer
    context = f"{question} Choices: {', '.join(choice_list)}"

    dev_texts.append(context)
    dev_choices.append(choice_list)
    dev_actual_answers.append(correct_answer)

# Initialize the tokenizer and model for text generation with LLama
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize the text generation pipeline with bfloat16 for reduced memory usage
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Initialize a sentence transformer model for semantic similarity on CPU
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Function to generate predictions with explanations using semantic similarity
def generate_predictions_and_evaluate(texts, choices):
    predicted_answers = []

    for i, (context, choice_list) in enumerate(zip(texts, choices)):
        prompt = f"Question: {context}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        # Generate response
        output = pipe(messages, max_new_tokens=50)
        
        # Extract generated text safely
        generated_text = ""
        if isinstance(output, list) and len(output) > 0:
            generated_content = output[0].get("generated_text", "")
            if isinstance(generated_content, list):
                generated_text = " ".join([str(part) for part in generated_content])
            elif isinstance(generated_content, str):
                generated_text = generated_content

        # Compute similarity between the generated text and each choice
        generated_embedding = similarity_model.encode(generated_text, convert_to_tensor=True)
        choice_embeddings = similarity_model.encode(choice_list, convert_to_tensor=True)
        similarities = util.cos_sim(generated_embedding, choice_embeddings)[0].cpu().numpy()

        # Select the choice with the highest similarity score
        predicted_answer = choice_list[int(np.argmax(similarities))]
        predicted_answers.append(predicted_answer)

    return predicted_answers

# Generate predictions on dev data
predicted_answers = generate_predictions_and_evaluate(dev_texts, dev_choices)

# Save predictions to a CSV file for review
df_predictions = pd.DataFrame({
    'Question ID': dev_question_ids,
    'Question': dev_texts,
    'Actual Correct Answer': dev_actual_answers,
    'Predicted Answer': predicted_answers
})
df_predictions.to_csv('predictions_dev.csv', index=False)
print("Predictions on dev data saved to predictions_dev.csv.")
