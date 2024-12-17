import pandas as pd
import os
import datasets

# print(f"Loading dataset (Size: 247M train, 166k test)...")
print(f"Loading dataset...")
dataset_name = "hendrycks/competition_math"
dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)
print(f"Dataset loaded")

# Extract the test split
test_df = pd.DataFrame(dataset["test"])
print(f"Test split size: {len(test_df)}")

# Add row ids
test_df["row_id"] = range(len(test_df))

# Keep only the columins we need (Interestingly it already has a problem and solution column!)
test_df = test_df[["row_id", "problem", "solution"]]

# Save to csv
print(f"Saving to CSV")
test_df.to_csv("datasets/original/competition_math.csv", index=False)
print(f"Saved to CSV at datasets/original/competition_math.csv")