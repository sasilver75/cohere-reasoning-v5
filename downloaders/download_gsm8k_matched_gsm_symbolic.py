import pandas as pd
from datasets import load_dataset

# Load each dataset
gsm_symbolic_df = pd.DataFrame(load_dataset("apple/GSM-Symbolic", "main")["test"])
gsm_df = pd.DataFrame(load_dataset("openai/gsm8k", "main")["test"])

# Filter GSM8k to only include questions that are in GSM-Symbolic
gsm_symbolic_questions = set(gsm_symbolic_df["original_question"])
filtered_gsm_df = gsm_df[gsm_df["question"].isin(gsm_symbolic_questions)]


def get_reasoning(answer: str) -> str:
    return answer.split("####")[0].strip()

def get_final_answer(answer: str) -> str:
    return answer.split("####")[1].strip()

# Add new columns
filtered_gsm_df["problem_id"] = range(len(filtered_gsm_df))
filtered_gsm_df["reasoning"] = filtered_gsm_df["answer"].apply(get_reasoning)
filtered_gsm_df["solution"] = filtered_gsm_df["answer"].apply(get_final_answer)
# Rename columns
filtered_gsm_df = filtered_gsm_df.rename(columns={"question": "problem"})

# Keep only the columns we need
filtered_gsm_df = filtered_gsm_df[["problem_id", "problem", "reasoning", "solution"]]

# Save
print(f"Saving df with {len(filtered_gsm_df)} rows and columns {filtered_gsm_df.columns.tolist()}")
filtered_gsm_df.to_csv("datasets/original/gsm8k_matched_gsm_symbolic.csv", index=False)
print(f"Saved to CSV at datasets/original/gsm8k_matched_gsm_symbolic.csv")
