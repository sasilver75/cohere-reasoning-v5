from pathlib import Path
import pandas as pd
import asyncio
from time import perf_counter
from models import CohereExperimentHelper
import logging

"""
This is like... from v4
TODO: Write me!
"""

# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()
SOURCE_PATH = Path("datasets/derived/interesting_problems.txt")
SINK_PATH = Path("datasets/derived/interesting_problems_off_policy_solutions.csv")
TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM = 3  # Number of incorrect solutions to generate per problem.
MAX_SOLUTION_GENERATION_ATTEMPTS = 10  # How many generate-verify loops to try before giving up and generating a single incorrect solution.
# END OF TUNABLE PARAMETERS
# PARAMETER CHECKS (Do not change)
if not (TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM > 0):
    raise ValueError("TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM must be greater than 0")
if not (MAX_SOLUTION_GENERATION_ATTEMPTS > 0):
    raise ValueError("MAX_SOLUTION_GENERATION_ATTEMPTS must be greater than 0")
# END OF PARAMETER CHECKS

logger = logging.getLogger(__name__)

async def _generate_incorrect_solutions(problem_row_ids: list[int]) -> pd.DataFrame:
    ...


async def main():
    # Read the input data
    print(f"Reading input data from {SOURCE_PATH}...")
    with open(SOURCE_PATH, "r") as f:
        problem_row_ids = [int(line.strip()) for line in f.readlines()]
    print(f"Read {len(problem_row_ids)} row_ids as input data.")

    # Generate incorrect solutions
    print(f"Generating {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM} incorrect solutions per problem, up to {MAX_SOLUTION_GENERATION_ATTEMPTS} attempts per solution...")
    df = await _generate_incorrect_solutions(problem_row_ids)
    print(f"Generated {len(df)} incorrect solutions ({df['row_id'].nunique()} problems, {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM} solutions per problem).")

    # Save the output
    print(f"Saving results to {SINK_PATH}...")
    df.to_csv(SINK_PATH, index=False)
    print(f"Saved results to {SINK_PATH}.")



if __name__ == "__main__":
    print("Starting...")
    start = perf_counter()
    asyncio.run(main())
    print(f"Done! Elapsed: {perf_counter() - start:.2f}s")