import asyncio
import pandas as pd
from models import CohereExperimentHelper
from pathlib import Path
"""
This is basically a combination of gnerate_straight_shot and generate_solvable_incorrect
from the cohere-reasoning-v4 project.

(Change from v4): Appraise problems until we reach the target number of of solvable problems.
"""

# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()
SOURCE_PATH = Path("datasets/original/cn_k12_math_problems.csv")
SINK_PATH = Path("datasets/derived/interesting_problems.csv")  # Doesn't include CSV
TARGET_N_SOLVABLE_PROBLEMS = 100
SEED = 42
LOWER_BOUND_FOR_SOLVABLE = 0.2
UPPER_BOUND_FOR_SOLVABLE = 0.6


async def appraise_problem(row: pd.Series) -> list[pd.Series]:
    ...



async def main():
    n_solvable_problems = 0
    incorrect_solutions = []

    print(f"Loading problems from {SOURCE_PATH}...")
    df = pd.read_csv(SOURCE_PATH)
    print(f"Loaded {len(df)} problems.")

    # Shuffle, using random seed
    shuffled_df = df.sample(frac=1, random_state=SEED)

    # For each problem, appraise it. If it's solvable, add the incorrect solutions to the accumulator.
    for _, row in shuffled_df.iterrows():
        if results := await appraise_problem(row):
            n_solvable_problems += 1
            incorrect_solutions.extend(results)
    
    # Create a dataframe from the incorrect solutions
    incorrect_df = pd.DataFrame(incorrect_solutions)

    # Write incorrect solutions todisk as a CSV
    incorrect_df.to_csv(SINK_PATH, index=False)
    # Write the solvable problem IDs to a text file
    solvable_problem_ids = list(incorrect_df["row_id"])
    with open(SINK_PATH.with_suffix(".txt"), "w") as f:


    



        




if __name__ == "__main__":
    asyncio.run(main())

