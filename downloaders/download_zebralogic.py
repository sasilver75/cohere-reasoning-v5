import pandas as pd
import os
import datasets

# Download the dataset
print(f"Loading dataset...")
dataset_name = "allenai/ZebraLogicBench-private"
dataset = datasets.load_dataset(dataset_name, "mc_mode")

# Exrtact the test split into a df (there's only a test split of 3259 rows)
test_df = pd.DataFrame(dataset["test"])

# Create the columns we need
test_df["row_id"] = range(len(test_df))
test_df["problem"] = test_df.apply(
    lambda row: f"{row['puzzle']}\n\n Question: {row['question']}\n\nSelect from the following options: {row['choices']}", 
    axis=1
)
test_df["solution"] = test_df["answer"]

# Keep only the columns we need
test_df = test_df[["row_id", "problem", "solution"]]

# Save to csv
dataset_directory = "datasets"
dataset_subdirectory = "original"
file_path = f"{dataset_directory}/{dataset_subdirectory}/zebralogic_mc.csv"

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists(os.path.join(dataset_directory, dataset_subdirectory)):
    os.makedirs(os.path.join(dataset_directory, dataset_subdirectory))

print(f"Saving to CSV")
test_df.to_csv(file_path, index=False)
print(f"Saved to CSV at {file_path}")