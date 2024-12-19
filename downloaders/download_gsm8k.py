from datasets import load_dataset
import pandas as pd


dataset_name = "openai/gsm8k"

dataset = load_dataset(dataset_name, "main")

test_df = pd.DataFrame(dataset["test"])

def get_reasoning(answer: str) -> str:
    return answer.split("####")[0].strip()

def get_final_answer(answer: str) -> str:
    return answer.split("####")[1].strip()

# Assign new columns
test_df["row_id"] = range(len(test_df))
test_df["reasoning"] = test_df["answer"].apply(get_reasoning)
test_df["solution"] = test_df["answer"].apply(get_final_answer)

# Rename columns
test_df = test_df.rename(columns={"question": "problem"})

# Keep only the columns we need (++ row_id)
test_df = test_df[["row_id", "problem", "reasoning", "solution"]]

# Save to csv
test_df.to_csv("datasets/original/gsm8k.csv", index=False)

print(f"Saved to CSV at datasets/original/gsm8k.csv")
