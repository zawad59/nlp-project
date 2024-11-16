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

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=50,
        temperature=0.5,  # Lower temperature for more focused generation
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = generated_text.split("Answer:")[-1].strip()
    return predicted_answer


def refine_prediction_with_embeddings(generated_answer, choices):
    """
    Use cosine similarity to refine the generated answer.
    """
    choice_embeddings = embedder.encode(choices, convert_to_tensor=True)
    generated_embedding = embedder.encode(generated_answer, convert_to_tensor=True)
    
    # Calculate cosine similarities
    cosine_similarities = util.cos_sim(generated_embedding, choice_embeddings)[0]
    best_index = torch.argmax(cosine_similarities).item()

    # If "None of the above" is the highest but has a very low similarity, avoid selecting it
    if choices[best_index].lower() == "none of above" and cosine_similarities[best_index] < 0.4:
        second_best_index = torch.topk(cosine_similarities, 2)[1][1].item()
        return choices[second_best_index]

    return choices[best_index]


def evaluate_on_test(test_data):
    correct_predictions = 0
    results = []
    for idx, item in enumerate(test_data):
        question = item['text']
        choices = item['choices']
        correct_answer = item['correct_answer']

        # Generate initial answer
        generated_answer = generate_answer(question)
        
        print(f"Generated Answer: {generated_answer}")
        print(f"Choices: {choices}")

        # Refine prediction using cosine similarity
        refined_answer = refine_prediction_with_embeddings(generated_answer, choices)

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
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question Text", "Generated Answer", "Refined Answer", "Correct Answer", "Correct?"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to {filename}")


# Evaluate using the best model and save the results
results = evaluate_on_test(processed_test_data)
save_predictions_to_csv(results)
