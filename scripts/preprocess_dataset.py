import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import os

# Define paths
data_dir = "../data"
train_path = os.path.join(data_dir, "imdb_train.parquet")
test_path = os.path.join(data_dir, "imdb_test.parquet")
output_train_path = os.path.join(data_dir, "tokenized_imdb_train.parquet")
output_test_path = os.path.join(data_dir, "tokenized_imdb_test.parquet")

# 1.2.1 Load the saved dataset
print("Loading IMDb dataset from Parquet files...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Convert to Hugging Face Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# 1.2.2 Explore the dataset
print("Dataset structure:", dataset)
print("Sample training data:", dataset['train'][0])

# Check for missing values
print("Missing values in train:", train_df.isnull().sum())
print("Missing values in test:", test_df.isnull().sum())

# Check class distribution
sns.countplot(x='label', data=train_df)
plt.title("Sentiment Class Distribution (Train)")
plt.xlabel("Sentiment (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.savefig(os.path.join(data_dir, "class_distribution.png"))
plt.close()

# Check text length distribution
train_df['text_length'] = train_df['text'].apply(len)
print("Average text length (train):", train_df['text_length'].mean())
print("Text length stats (train):\n", train_df['text_length'].describe())

sns.histplot(train_df['text_length'], bins=50)
plt.title("Text Length Distribution (Train)")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(data_dir, "text_length_distribution.png"))
plt.close()

# 1.2.3 Split the dataset (IMDb already has train/test, but optionally split train into train/validation)
# For simplicity, we'll use 10% of train as validation
train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
dataset['train'] = train_val_split['train']
dataset['validation'] = train_val_split['test']

# 1.2.4 Tokenization
print("Tokenizing dataset...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Format for PyTorch
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Save tokenized datasets
print("Saving tokenized datasets...")
tokenized_dataset['train'].to_parquet(output_train_path)
tokenized_dataset['validation'].to_parquet(os.path.join(data_dir, "tokenized_imdb_validation.parquet"))
tokenized_dataset['test'].to_parquet(output_test_path)

print(f"Tokenized datasets saved to {data_dir}")
print("Sample tokenized training data:", tokenized_dataset['train'][0])