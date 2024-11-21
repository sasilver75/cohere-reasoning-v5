import asyncio
from models import CohereExperimentHelper
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
from time import perf_counter

"""
This is similar to the generate_prefixes_remaining_on_policy.py script from v4, but uses a Token Bucket strategy instead,
as well as the Helper classes from v5.
"""

# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()
SOURCE_PATH = Path("datasets/derived/interesting_problems.csv")
SINK_PATH = Path("datasets/derived/interesting_problems_padded.csv")
TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM = 3  # Number of desired incorrect solutions per problem. Will truncate existing ones if we have more existing ones than desired.
MAX_SOLUTION_GENERATION_ATTEMPTS = 10  # How many generate-verify loops to try before giving up and generating a single incorrect solution.
# END OF TUNABLE PARAMETERS
# PARAMETER CHECKS (Do not change)

# END OF PARAMETER CHECKS

logger = logging.getLogger(__name__)

async def _persistently_generate_incorrect_solution(base_row: pd.Series, solution_id: int) -> pd.Series:
    """
    Given a row, generate an incorrect solution using the strong completer for that row's problem.
    """

    attempts = 0
    while attempts < MAX_SOLUTION_GENERATION_ATTEMPTS:
        attempts += 1

        # Generate and verify a candidate solution
        candidate_solution = await HELPER.get_solution(base_row)
        verification_result, verification_reasoning = await HELPER.get_verification(candidate_solution, base_row)

        # If we found an incorrect solution, return it
        if not verification_result:
            new_row = base_row.copy()
            new_row["solution_id"] = solution_id
            new_row["candidate_solution"] = candidate_solution
            new_row["candidate_verification_result"] = verification_result
            new_row["candidate_verification_reasoning"] = verification_reasoning
            return new_row
        
        if attempts > (.8 * MAX_SOLUTION_GENERATION_ATTEMPTS):
            logger.warning(f"Reached {attempts}/{MAX_SOLUTION_GENERATION_ATTEMPTS} attempts without finding an incorrect solution for problem {base_row['row_id']}.")
    
    # If we're here nad haven't returned, we weren't able to find an incorrect solution. This should be rare enough to be considered noise, so we'll return a copy of the base row (with a new solution_idx)
    new_row = base_row.copy()
    new_row["solution_id"] = solution_id
    return new_row
        


# async def _pad_incorrect_solutions(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Given a dataframe with partially-completed number of incorrect solutions for each row_id, pad the remaining solutions
#     with incorrect solutions, using the strong completer.
#     """
#     all_padded_solutions: list[dict] = []

#     # TODO: THIS NEEDS TO BE PARALLELIZED
#     for row_id in tqdm(df["row_id"].unique(), desc="Processing problems"):
#         # Get the rows for this problem (row_id)
#         problem_rows = df[df["row_id"] == row_id]
#         n_existing_solutions = len(problem_rows)
        
#         if n_existing_solutions >= TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM:
#             # Truncate to target count if we have too many solutions
#             all_padded_solutions.extend(problem_rows.iloc[:TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM].to_dict('records'))
#         else:
#             # Add existing solutions
#             all_padded_solutions.extend(problem_rows.to_dict('records'))
            
#             # Generate additional solutions
#             base_row = problem_rows.iloc[0]
#             # TODO: THIS NEEDS TO BE PARALLELIZED
#             for solution_id in range(n_existing_solutions, TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM):
#                 new_solution = await _persistently_generate_incorrect_solution(base_row, solution_id)
#                 all_padded_solutions.append(new_solution.to_dict())  # Note: pd.Series doesn't need 'records', just pd.DataFrames do.
    
#     # Convert the list of rows into a dataframe and return it, correctly sorted (though it already should be)
#     padded_df = pd.DataFrame(all_padded_solutions)
#     return padded_df.sort_values(["row_id", "solution_id"]).reset_index(drop=True)


async def _pad_incorrect_solutions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with partially-completed number of incorrect solutions for each row_id, pad the remaining solutions
    with incorrect solutions, using the strong completer.
    """
    async def process_problem(problem_rows: pd.DataFrame) -> list[dict]:
        """Helper function to process a single problem and its solutions"""
        row_solutions = []
        n_existing_solutions = len(problem_rows)
        
        if n_existing_solutions >= TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM:
            # Truncate to target count if we have too many solutions
            return problem_rows.iloc[:TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM].to_dict('records')
        
        # Add existing solutions
        row_solutions.extend(problem_rows.to_dict('records'))
        
        # Generate additional solutions in "parallel"
        base_row = problem_rows.iloc[0]
        remaining_solutions = range(n_existing_solutions, TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM)
        tasks = [_persistently_generate_incorrect_solution(base_row, solution_id) for solution_id in remaining_solutions]
        new_solutions: list[pd.Series] = await asyncio.gather(*tasks)
        
        row_solutions.extend([solution.to_dict() for solution in new_solutions])
        return row_solutions

    # Process all problems in parallel
    problem_tasks = [
        process_problem(df[df["row_id"] == row_id])
        for row_id in df["row_id"].unique()
    ]
    all_padded_solutions: list[list[dict]] = await asyncio.gather(*problem_tasks)
    
    # Flatten the list of lists and convert to DataFrame
    padded_df = pd.DataFrame([
        solution 
        for problem_solutions in all_padded_solutions 
        for solution in problem_solutions
    ])
    
    return padded_df.sort_values(["row_id", "solution_id"]).reset_index(drop=True)

    
        
        
async def main():
    # Load source data
    print(f"Loading problems from {SOURCE_PATH}")
    df = pd.read_csv(SOURCE_PATH)
    print(f"Loaded {df["row_id"].nunique()} problems, with {len(df)} total solutions; padding to {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM} solutions per problem (for {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM * df["row_id"].nunique()} total solutions)")

    # Pad solutions
    print(f"Padding solutions...")
    padded_df = await _pad_incorrect_solutions(df)
    print(f"Padded to {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM} solutions per problem. Total solutions: {len(padded_df)} for {padded_df["row_id"].nunique()} problems.")

    # Save to sink
    print(f"Saving to {SINK_PATH}")
    padded_df.to_csv(SINK_PATH, index=False)
    print("Done!")


if __name__ == "__main__":
    print("Starting...")
    start = perf_counter()
    asyncio.run(main())
    print(f"Done! Elapsed: {perf_counter() - start:.2f}s")
