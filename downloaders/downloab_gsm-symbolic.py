import pandas as pd 
from datasets import load_dataset

print("Loading dataset")
dataset = load_dataset("apple/GSM-Symbolic", name="main")
df = pd.DataFrame(dataset["test"])
print(f"Loaded {len(df)} rows")

# TODO: Would we like to evaluate more than just the first instance of each problem?
print("Selecting index=0 rows")
sub_df = df[df["instance"] == 0]
print(f"Yielded {len(sub_df)} rows")


def get_reasoning(answer: str) -> str:
    return answer.split("####")[0].strip()

def get_final_answer(answer: str) -> str:
    return answer.split("####")[1].strip()

# Compute/rename to final columns
sub_df = sub_df.rename(columns={"id": "problem_id", "question": "problem"})
sub_df["reasoning"] = sub_df["answer"].apply(get_reasoning)
sub_df["solution"] = sub_df["answer"].apply(get_final_answer)
sub_df = sub_df[["problem_id", "problem", "reasoning", "solution"]]

print(f"Final columns: {", ".join(sub_df.columns.tolist())}")

# Save to CSV
filename = "datasets/original/gsm-symbolic.csv"
sub_df.to_csv(filename, index=False)
print(f"Saved to CSV at {filename}")
