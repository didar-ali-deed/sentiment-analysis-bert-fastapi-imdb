import os
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Define paths
data_dir = "../data"
models_dir = "../models"
results_dir = "../results"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load tokenized datasets
print("Loading tokenized datasets...")
tokenized_dataset = DatasetDict({
    "train": Dataset.from_parquet(os.path.join(data_dir, "tokenized_imdb_train.parquet")),
    "validation": Dataset.from_parquet(os.path.join(data_dir, "tokenized_imdb_validation.parquet")),
    "test": Dataset.from_parquet(os.path.join(data_dir, "tokenized_imdb_test.parquet"))
})

# Filter out any samples with None labels
tokenized_dataset = {k: v.filter(lambda x: x['label'] is not None) for k, v in tokenized_dataset.items()}
print("Dataset structure:", tokenized_dataset)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(models_dir, "bert-imdb-checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,  # Enable mixed precision for GPU acceleration
)

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
print("Training model...")
trainer.train()

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test set results:", test_results)

# Save test results
results_df = pd.DataFrame([test_results])
results_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)

# Save the model and tokenizer
print("Saving model and tokenizer...")
model.save_pretrained(os.path.join(models_dir, "bert-imdb-final"))
tokenizer.save_pretrained(os.path.join(models_dir, "bert-imdb-final"))
print(f"Model and tokenizer saved to {os.path.join(models_dir, 'bert-imdb-final')}")