from datasets import load_dataset
import pandas as pd


dataset_name = "openai/gsm8k"

dataset = load_dataset(dataset_name, "main")

test_df = pd.DataFrame(dataset["test"])

def get_final_answer(answer: str) -> str:
    return answer.split("####")[1].strip()

# Assign new columns
test_df["problem_id"] = range(len(test_df))
test_df["answer"] = test_df["answer"].apply(get_final_answer)

# Rename columns
test_df = test_df.rename(columns={"question": "problem"})

# Keep only the columns we need
test_df = test_df[["problem_id", "problem", "answer"]]

# Save to csv
output_filepath = "gsm8k/datasets/original/gsm8k.csv"
test_df.to_csv(output_filepath, index=False)
print(f"Saved to CSV at {output_filepath}")
