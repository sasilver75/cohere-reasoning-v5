from time import perf_counter
import asyncio
from pathlib import Path
from models import CohereExperimentHelper, DummyExperimentHelper, OpenRouterExperimentHelper, OpenRouterProvider, OpenRouterModel
import pandas as pd
from tqdm.asyncio import tqdm_asyncio as atqdm

# TUNABLE PARAMETERS
HELPER = OpenRouterExperimentHelper(strong_completer=OpenRouterModel.QWEN_2_5_72B_INSTRUCT)  # Encapsulates logic about the specific models we're using
EXPERIMENT_NAME = "experiment-ZEBRAMC-qwen2.5_70b-20-12_17_2024"  # The name of the experiment; used for directory naming for results.
SOURCE_PATH = Path(f"datasets/experiments/{EXPERIMENT_NAME}/interesting_problems_on_policy_solutions.csv")
SINK_PATH = Path(f"datasets/experiments/{EXPERIMENT_NAME}/interesting_problems_completed.csv")
N_COMPLETIONS_PER_PREFIX = 2  # For each problem, the number of solution attempts over which we'll evaluate problem difficulty. Note that without retries we'll have 2*{N_SOLUTION_ATTEMPTS_PER_PROBLEM} API calls per problem.
# END OF TUNABLE PARAMETERS
# PARAMETER CHECKS (Do not change)
if not (N_COMPLETIONS_PER_PREFIX > 0):
    raise ValueError("N_COMPLETIONS_PER_PREFIX must be greater than 0")
# END OF CHECKS



async def _generate_completions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate completions for each incorrect solution in the DataFrame.
    """
    async def process_completion(row: pd.Series, completion_id: int) -> dict:
        """Helper function to process a single completion"""
        # Generate the prefix and completion
        prefix, completion = await HELPER.get_prefix_and_completion(row)
        # Generate the verification of the completion of the f"{prefix}{completion}"
        verification_result, verification_reasoning = await HELPER.get_verification(f"{prefix}{completion}", row)
        
        # Create new row with the completion data
        new_row = row.copy()
        new_row["completion_id"] = completion_id
        new_row["prefix"] = prefix
        new_row["completion"] = completion
        new_row["completion_verification_result"] = verification_result
        new_row["completion_verification_reasoning"] = verification_reasoning
        
        return new_row.to_dict()

    async def process_completions(row: pd.Series) -> list[dict]:
        """
        Helper function to orchestrate the generation of multiple completions for a single incorrect solution"""
        completion_tasks = [
            process_completion(row, completion_id)
            for completion_id in range(N_COMPLETIONS_PER_PREFIX)
        ]
        return await asyncio.gather(*completion_tasks)

    # Generate all incorrect solutions in parallel, across all row_ids.
    solution_tasks = [
        process_completions(row)
        for _, row in df.iterrows()
    ]
    
    # Gather all completions
    all_completions = await atqdm.gather(
        *solution_tasks, 
        desc=f"Generating completions for {len(df)} solutions", 
        total=len(df), 
        colour="green"
    )
    
    # Flatten the list of lists and convert to DataFrame
    completions_df = pd.DataFrame([
        completion 
        for solution_completions in all_completions 
        for completion in solution_completions
    ])
    
    return completions_df.sort_values(["row_id", "solution_id", "completion_id"]).reset_index(drop=True)

async def main():
    # Read the input data
    print("Reading input data...")
    df = pd.read_csv(SOURCE_PATH)
    print(f"Loaded {df["row_id"].nunique()} problems, with {max(df["solution_id"])+1} solutions per problem.")
    # Check for and remove NaN solutions
    nan_count = df["candidate_solution"].isna().sum()
    print(f"Found {nan_count} rows with NaN candidate solutions - dropping these rows, since we can't make a prefix out of them")
    df = df.dropna(subset=["candidate_solution"]).reset_index(drop=True)

    # Generate completions
    print(f"Generating {N_COMPLETIONS_PER_PREFIX} completions (and verifications) per incorrect solution")
    completed_df = await _generate_completions(df)
    print(f"Generated {len(completed_df)} completions and verifications.")

    # Save the output; We assume that the directories have been created by generate_incorrect_solutions_on_policy.py
    print(f"Saving results to {SINK_PATH}...")
    completed_df.to_csv(SINK_PATH, index=False)
    print(f"Saved results to {SINK_PATH}.")

if __name__ == "__main__":
    print("Starting...")
    start = perf_counter()
    asyncio.run(main())
    print(f"Done! Elapsed: {perf_counter() - start:.2f}s")