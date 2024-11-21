from pathlib import Path
import pandas as pd
import asyncio
from time import perf_counter
from models import CohereExperimentHelper
import logging
from tqdm.asyncio import tqdm_asyncio as atqdm

"""
This is like... from v4
TODO: Write me!
"""

# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()
ORIGINAL_DATASET_PATH = Path("datasets/original/cn_k12_math_problems.csv")
SOURCE_PATH = Path("datasets/derived/interesting_problems.txt")
SINK_PATH = Path("datasets/derived/interesting_problems_off_policy_solutions.csv")
TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM = 2  # Number of incorrect solutions to generate per problem.
MAX_SOLUTION_GENERATION_ATTEMPTS = 10  # How many generate-verify loops to try before giving up when trying to generate a single incorrect solution.
# END OF TUNABLE PARAMETERS
# PARAMETER CHECKS (Do not change)
if not (TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM > 0):
    raise ValueError("TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM must be greater than 0")
if not (MAX_SOLUTION_GENERATION_ATTEMPTS > 0):
    raise ValueError("MAX_SOLUTION_GENERATION_ATTEMPTS must be greater than 0")
# END OF PARAMETER CHECKS

logger = logging.getLogger(__name__)

async def _generate_incorrect_solutions(problem_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate incorrect solutions for each problem in the DataFrame.
    """
    async def generate_single_solution(row: pd.Series, solution_id: int) -> dict:
        """Helper function (to the helper function) to generate a single incorrect solution"""
        attempts = 0
        while attempts < MAX_SOLUTION_GENERATION_ATTEMPTS:
            attempts += 1
            
            # Generate and verify a candidate solution
            candidate_solution = await HELPER.get_solution(row)
            verification_result, verification_reasoning = await HELPER.get_verification(candidate_solution, row)

            # If we found an incorrect solution, return it
            if not verification_result:
                new_row = row.copy()
                new_row["solution_id"] = solution_id
                new_row["candidate_solution"] = candidate_solution
                new_row["candidate_verification_result"] = verification_result
                new_row["candidate_verification_reasoning"] = verification_reasoning
                return new_row.to_dict()
            
            if attempts > (.8 * MAX_SOLUTION_GENERATION_ATTEMPTS):
                logger.warning(f"Reached {attempts}/{MAX_SOLUTION_GENERATION_ATTEMPTS} attempts without finding an incorrect solution for problem {row['row_id']}, solution {solution_id}.")
        
        # If we didn't find an incorrect solution after MAX_SOLUTION_GENERATION_ATTEMPTS attempts, return placeholder (The weak completer is so weak that I imagine this will rarely happen)
        new_row = row.copy()
        new_row["solution_id"] = solution_id
        new_row["candidate_solution"] = "<Placeholder Solution>"
        new_row["candidate_verification_result"] = False
        new_row["candidate_verification_reasoning"] = "<Placeholder Reasoning>"
        return new_row.to_dict()

    async def process_problem(row: pd.Series) -> list[dict]:
        """Helper function to process all solutions for a single problem in parallel"""
        solution_tasks = [
            generate_single_solution(row, solution_id)
            for solution_id in range(TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM)
        ]
        return await asyncio.gather(*solution_tasks)

    # Process all problems in parallel
    problem_tasks = [
        process_problem(row)
        for _, row in problem_df.iterrows()
    ]
    all_solutions = await atqdm.gather(*problem_tasks, desc=f"Generating solutions for {len(problem_df)} problems", total=len(problem_df), colour="green")
    
    # Flatten the list of lists and convert to DataFrame
    solutions_df = pd.DataFrame([
        solution 
        for problem_solutions in all_solutions 
        for solution in problem_solutions
    ])
    
    return solutions_df.sort_values(["row_id", "solution_id"]).reset_index(drop=True)


async def main():
    # Read the input data
    print(f"Reading input data from {SOURCE_PATH}...")
    with open(SOURCE_PATH, "r") as f:
        problem_row_ids = [int(line.strip()) for line in f.readlines()]
    print(f"Read {len(problem_row_ids)} row_ids as input data.")

    print(f"Reading original dataset from {ORIGINAL_DATASET_PATH} for solvable problems...")
    original_df = pd.read_csv(ORIGINAL_DATASET_PATH)
    solvable_problem_df = original_df[original_df["row_id"].isin(problem_row_ids)]
    print(f"Read {len(original_df)} rows from the original dataset before filtering to {len(solvable_problem_df)} solvable problem rows.")

    # Generate incorrect solutions
    print(f"Generating {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM} incorrect solutions per problem, using up to {MAX_SOLUTION_GENERATION_ATTEMPTS} attempts per incorrect solution...")
    incorrect_solutions_df = await _generate_incorrect_solutions(solvable_problem_df)
    print(f"Generated {len(incorrect_solutions_df)} incorrect solutions ({incorrect_solutions_df['row_id'].nunique()} problems, {TARGET_N_INCORRECT_SOLUTIONS_PER_PROBLEM} solutions per problem).")

    # Save the output
    print(f"Saving results to {SINK_PATH}...")
    incorrect_solutions_df.to_csv(SINK_PATH, index=False)
    print(f"Saved results to {SINK_PATH}.")



if __name__ == "__main__":
    print("Starting...")
    start = perf_counter()
    asyncio.run(main())
    print(f"Done! Elapsed: {perf_counter() - start:.2f}s")