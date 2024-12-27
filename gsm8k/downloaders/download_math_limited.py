import pandas as pd
from datasets import load_dataset


"""
Competition Math is only 500 examples
Let's just... arbitrarily use the 500 exmaples from the Let's Verify Step by Step paper's test set, lol
https://huggingface.co/datasets/HuggingFaceH4/MATH-500
"""

dataset = load_dataset("HuggingFaceH4/MATH-500")
df = pd.DataFrame(dataset["test"])
print(f"Loaded {len(df)} rows with columns {df.columns.tolist()}")

# Add, reename, filter columns, reorder
df["problem_id"] = range(len(df))
# NOTE this one has an additional subject and level, if we want it.
df = df[["problem_id", "problem", "answer", "subject", "level"]]

# Save to csv
output_filepath = "gsm8k/datasets/original/math_limited.csv"
df.to_csv(output_filepath, index=False)
print(f"Saved to CSV at {output_filepath}")
