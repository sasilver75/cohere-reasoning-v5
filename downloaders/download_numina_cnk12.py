import pandas as pd
import os
import datasets

print(f"Loading dataset...")
dataset_name = "AI-MO/NuminaMath-CoT"
dataset = datasets.load_dataset(dataset_name)
print(f"Dataset loaded.")

# Combine train and test sets (keep info), filter to cn_k12, add index
# TODO: Do we want to kep the train set info too? (Train is 859494 [276564 cnk12] and Test is 100 [cnk12])
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
train_df["set"] = "train"
test_df["set"] = "test"
df = pd.concat([train_df, test_df])
print(f"Combined train ({len(train_df)} rows) and test ({len(test_df)} rows) sets into ({len(df)} rows)...")

# Filter to only the cn_k12 subset
df = df[df["source"] == "cn_k12"]
print(f"Filtered to {len(df)} rows of cn_k12 problems")

# Set an explicit "row_id" column (since the dataset doesn't have one)
print("Setting row_id column...")
df["row_id"] = range(len(df))

# Just keep the columns we need
df = df[["row_id", "problem", "solution"]]

# Save to csv
dataset_directory = "datasets"
dataset_subdirectory = "original"
file_path = f"{dataset_directory}/{dataset_subdirectory}/numina_cnk12.csv"

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists(os.path.join(dataset_directory, dataset_subdirectory)):
    os.makedirs(os.path.join(dataset_directory, dataset_subdirectory))

print(f"Saving to csv at {file_path}...")
df.to_csv(file_path, index=False)
print(f"Saved to csv at {file_path}")
