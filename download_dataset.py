import pandas as pd
import os
import datasets

print(f"Loading dataset (Size: 247M train, 166k test)...")
dataset_name = "AI-MO/NuminaMath-CoT"
dataset = datasets.load_dataset(dataset_name)
print(f"Dataset loaded.")

dataset_directory = "datasets"
dataset_subdirectory = "original"
file_path = f"{dataset_directory}/{dataset_subdirectory}/cn_k12_math_problems.csv"

# Combine train and test sets (keep info), filter to cn_k12, add index
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
train_df["set"] = "train"
test_df["set"] = "test"
df = pd.concat([train_df, test_df])
print(f"Combined train and test sets ({len(df)} rows)...")

# Filter to only the cn_k12 subset
df = df[df["source"] == "cn_k12"]

# Shuffle the dataframe
print("Shuffling dataframe...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Set an explicit "id" column (since the dataset doesn't have one)
print("Setting id column...")
df["id"] = range(len(df))

# Save to csv
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists(os.path.join(dataset_directory, dataset_subdirectory)):
    os.makedirs(os.path.join(dataset_directory, dataset_subdirectory))

df.to_csv(file_path, index=False)
print(f"Saved to csv at {file_path}")
