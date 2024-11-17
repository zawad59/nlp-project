def evaluate_on_test(test_data):
    predictions = []
    correct_predictions = 0

    # Iterate through each item in the test dataset
    for idx, item in enumerate(test_data):
        question = item['text']
        choices = item['choices']
        true_label = item['label']
        correct_answer = choices[true_label]

        # Generate the predicted answer using the fine-tuned model
        generated_answer = generate_answer(question, choices)
        refined_answer = refine_prediction_with_similarity(generated_answer, choices)

        # Check if the predicted answer matches the correct answer
        is_correct = "yes" if refined_answer == correct_answer else "no"
        predictions.append({
            "Question ID": idx + 1,
            "Question": question,
            "Choices": ', '.join(choices),
            "Predicted Answer": refined_answer,
            "Correct Answer": correct_answer,
            "Correct": is_correct
        })

        # Count correct predictions for accuracy calculation
        if is_correct == "yes":
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(test_data)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    return predictions, accuracy

def save_predictions_to_csv(predictions, filename="phi1_5_predictions.csv"):
    """
    Save the predictions to a CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Question ID", "Question", "Choices",
                                                  "Predicted Answer", "Correct Answer", "Correct"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Predictions saved to {filename}")

# Run evaluation and save results
predictions, accuracy = evaluate_on_test(processed_test_data)
save_predictions_to_csv(predictions, filename="phi1_5_predictions.csv")
print(f"Final Test Accuracy: {accuracy:.4f}")
