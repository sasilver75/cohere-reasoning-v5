from time import perf_counter
import asyncio
from pathlib import Path
from models import CohereExperimentHelper
import pandas as pd

# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()  # Encapsulates logic about the specific models we're using
SOURCE_PATH = Path("datasets/derived/interesting_problems_on_policy_solutions.csv")
SINK_PATH = Path("datasets/derived/interesting_problems_completed.csv")
N_COMPLETIONS_PER_PREFIX = 2  # For each problem, the number of solution attempts over which we'll evaluate problem difficulty. Note that without retries we'll have 2*{N_SOLUTION_ATTEMPTS_PER_PROBLEM} API calls per problem.
PREFIX_SIZE = 0.7  # The proportion of the incorrect solution to use as a reaosning stub from which to complete.
# END OF TUNABLE PARAMETERS
# PARAMETER CHECKS (Do not change)
if not (N_COMPLETIONS_PER_PREFIX > 0):
    raise ValueError("N_COMPLETIONS_PER_PREFIX must be greater than 0")
if not (0 <= PREFIX_SIZE <= 1):
    raise ValueError("PREFIX_SIZE must be in [0, 1]")
# END OF CHECKS



async def _generate_completions(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Write me!
    ...

async def main():
    # Read the input data
    print("Reading input data...")
    df = pd.read_csv(SOURCE_PATH)
    print(f"Loaded {df["row_id"].nunique()} problems, with {max(df["solution_id"])+1} solutions per problem.")

    # Generate completions
    print(f"Generating {N_COMPLETIONS_PER_PREFIX} completions (and verifications) per incorrect solution, using a prefix size of {PREFIX_SIZE}")
    completed_df = await _generate_completions(df)
    print(f"Generated {len(completed_df)} completions and verifications.")

    # Save the output
    print(f"Saving results to {SINK_PATH}...")
    completed_df.to_csv(SINK_PATH, index=False)
    print(f"Saved results to {SINK_PATH}.")

if __name__ == "__main__":
    print("Starting...")
    start = perf_counter()
    asyncio.run(main())
    print(f"Done! Elapsed: {perf_counter() - start:.2f}s")