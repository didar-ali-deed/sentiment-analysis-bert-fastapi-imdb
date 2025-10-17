import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Define paths
models_dir = "../models"
results_dir = "../results"
model_path = os.path.join(models_dir, "bert-imdb-final")
output_path = os.path.join(results_dir, "predictions.csv")

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a prediction pipeline
print("Creating prediction pipeline...")
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)  # device=0 for GPU

# Test sentences
test_sentences = [
    "This movie was absolutely fantastic! I loved every moment of it.",
    "The plot was boring and the acting was terrible, a complete waste of time.",
    "It was an okay movie, not great but not awful either.",
    "The cinematography was stunning, but the story lacked depth.",
    "I couldn't stop laughing, this comedy was pure gold!"
]

# Make predictions
print("Making predictions on test sentences...")
predictions = sentiment_pipeline(test_sentences)

# Map label IDs to human-readable labels
label_map = {0: "Negative", 1: "Positive"}
results = [
    {"text": text, "predicted_label": label_map[int(pred["label"].split("_")[1])], "score": pred["score"]}
    for text, pred in zip(test_sentences, predictions)
]

# Save predictions to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

# Print predictions
print("Predictions:")
for result in results:
    print(f"Text: {result['text']}")
    print(f"Predicted Sentiment: {result['predicted_label']} (Score: {result['score']:.4f})\n")

print(f"Predictions saved to {output_path}")