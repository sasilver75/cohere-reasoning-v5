import pandas as pd
import os
import datasets


# Download hte dataset
print(f"Loading dataset...")
dataset_name = "hendrycks/competition_math"
dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)
print(f"Dataset loaded")

# Extract the test split into a df
test_df = pd.DataFrame(dataset["test"])
print(f"Test split size: {len(test_df)}")

# Add row ids
test_df["row_id"] = range(len(test_df))

# Keep only the columins we need (Interestingly it already has a problem and solution column!)
test_df = test_df[["row_id", "problem", "solution"]]

# Save to csv
dataset_directory = "datasets"
dataset_subdirectory = "original"
file_path = f"{dataset_directory}/{dataset_subdirectory}/competition_math.csv"

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists(os.path.join(dataset_directory, dataset_subdirectory)):
    os.makedirs(os.path.join(dataset_directory, dataset_subdirectory))

print(f"Saving to CSV")
test_df.to_csv(file_path, index=False)
print(f"Saved to CSV at {file_path}")