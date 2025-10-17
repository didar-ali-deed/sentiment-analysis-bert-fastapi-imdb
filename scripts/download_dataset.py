from datasets import load_dataset
import pandas as pd
import os

# Define paths
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)

# Load the IMDb dataset
print("Downloading IMDb dataset...")
dataset = load_dataset("imdb")

# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Save to Parquet files in the data directory
train_path = os.path.join(data_dir, "imdb_train.parquet")
test_path = os.path.join(data_dir, "imdb_test.parquet")
train_df.to_parquet(train_path)
test_df.to_parquet(test_path)

print(f"Dataset downloaded and saved to {data_dir}")
print(f"Training data saved to: {train_path}")
print(f"Test data saved to: {test_path}")
print("Sample training data:", train_df.head().to_dict(orient='records'))