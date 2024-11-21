import asyncio
import pandas as pd
from models import CohereExperimentHelper
from pathlib import Path
from tqdm import tqdm
from time import perf_counter

"""
This is basically a combination of gnerate_straight_shot and generate_solvable_incorrect
from the cohere-reasoning-v4 project.

(Change from v4): Appraise problems until we reach the target number of of solvable problems.
"""

# TUNABLE PARAMETERS
HELPER = CohereExperimentHelper()  # Encapsulates logic about the specific models we're using
SOURCE_PATH = Path("datasets/original/cn_k12_math_problems.csv")
SINK_PATH = Path("datasets/derived/interesting_problems.csv")
TARGET_N_SOLVABLE_PROBLEMS = 10  # The number of solvable problems we want to identify
N_SOLUTION_ATTEMPTS_PER_PROBLEM = 4  # For each problem, the number of solution attempts over which we'll evaluate problem difficulty. Note that without retries we'll have 2*{N_SOLUTION_ATTEMPTS_PER_PROBLEM} API calls per problem.
LOWER_SUCCESS_RATE_BOUND = 0.2  # The lower bound on the success rate of the solutions we'll accept as solvable/interesting; Number if [0, 1)
UPPER_SUCCESS_RATE_BOUND = .6  # The upper bound on the success rate of the solutions we'll accept as solvable/interesting; Number in [0, 1)
EPSILON = 1e-10  # To help with floating point division giving .199999 when it really should be .2
SEED = 42  # Random seed for dataset shuffling; We'll iterate through rows of this shuffled dataset until we identify the target number of solvable problems.
# END OF TUNABLE PARAMETERS
# PARAMETER CHECKS (Do not change)
if not (0 <= LOWER_SUCCESS_RATE_BOUND < UPPER_SUCCESS_RATE_BOUND < 1):
    raise ValueError("Success rate bounds must be in [0, 1) and satisfy LOWER_SUCCESS_RATE_BOUND < UPPER_SUCCESS_RATE_BOUND")
# END OF CHECKS


async def _generate_and_verify_solution(row: pd.Series) -> tuple[bool, pd.Series]:
    """
    Given a row from the source dataframe, generate a solution to the problem and verify it.
    """
    # Get the solution and verification information
    candidate_solution = await HELPER.get_solution(row)
    verification_result, verification_reasoning = await HELPER.get_verification(candidate_solution, row)

    # Construct a new row with the solution and verification results
    augmented_row = row.copy()
    augmented_row["candidate_solution"] = candidate_solution
    augmented_row["candidate_verification_result"] = verification_result
    augmented_row["candidate_verification_reasoning"] = verification_reasoning

    return verification_result, augmented_row


async def _appraise_problem(row: pd.Series) -> tuple[bool, list[pd.Series]]:
    """
    Given a problem, appraise it by generating N_SOLUTIONS_PER_PROBLEM solutions and determining whether
    the success rate of the solutions is within the bounds [LOWER_BOUND_FOR_SOLVABLE, UPPER_BOUND_FOR_SOLVABLE].

    If the problem is solvable, return a list of the incorrect solutions so that they can be re-used down the pipeline as 
    on-policy incorrect solutions.
    """
    # Generate N_SOLUTION_ATTEMPTS_PER_PROBLEM solutions and collect the incorrect ones
    solution_attempts = [
        _generate_and_verify_solution(row)
        for _ in range(N_SOLUTION_ATTEMPTS_PER_PROBLEM)
    ]
    results = await asyncio.gather(*solution_attempts)
    incorrect_solutions: list[pd.Series] = [attempt_data for attempt_success, attempt_data in results if not attempt_success]
    
    # Augment the incorrect solutions with the solution_ids
    for idx, augmented_row in enumerate(incorrect_solutions):
        augmented_row["solution_id"] = idx

    # Did we find the appropriate number of incorrect solutions?
    failure_rate = len(incorrect_solutions) / N_SOLUTION_ATTEMPTS_PER_PROBLEM
    success_rate = 1 - failure_rate

    print(f"Success rate for row {row["row_id"]}: {success_rate}")

    # Return a tuple of (is_solvable, incorrect_solutions)
    is_solvable = (LOWER_SUCCESS_RATE_BOUND - EPSILON) <= success_rate <= (UPPER_SUCCESS_RATE_BOUND + EPSILON)
    return (is_solvable, incorrect_solutions)





async def main():
    n_solvable_problems = 0
    incorrect_solutions = []

    # ~~~ (1) Load Problems and Shuffle ~~~
    print(f"Loading problems from {SOURCE_PATH}...")
    df = pd.read_csv(SOURCE_PATH)
    print(f"Loaded {len(df)} problems.")

    # Shuffle, using random seed
    shuffled_df = df.sample(frac=1, random_state=SEED)

    # ~~~ (2) Appraise Problems ~~~
    # For each problem, appraise it. If it's solvable, add the incorrect solutions to the accumulator.
    # Evaluate problems until we've found the target number of solvable problems.
    print(f"Appraising problems until we've found {TARGET_N_SOLVABLE_PROBLEMS} solvable problems...")
    
    # Let's choose to do this sequentially for now, until we have enough solutions.
    # TODO: This is far too slow. We need to parallelize this, but also not make too many calls needlessly. Should we just rely on the bucket and its settings to save us?
    # For small N, we don't want to exhaust our bucket for no reason ($); we can use semaphores, or batching, or...
    pbar = tqdm(total=TARGET_N_SOLVABLE_PROBLEMS, colour="green", desc="Finding solvable problems")
    for _, row in shuffled_df.iterrows():
        is_solvable, incorrect_solution_rows = await _appraise_problem(row)
        if is_solvable:
            n_solvable_problems += 1
            incorrect_solutions.extend(incorrect_solution_rows)
            pbar.update(1)

            if n_solvable_problems >= TARGET_N_SOLVABLE_PROBLEMS:
                break
    print(f"Found {n_solvable_problems} solvable problems.")
    
    # Create a dataframe from the incorrect solutions and make sure it's correctly sorted
    incorrect_df = pd.DataFrame(incorrect_solutions)
    incorrect_df = incorrect_df.sort_values(["row_id", "solution_id"]).reset_index(drop=True)

    # Reorder the columns of the dataframe (and discard redundant columns from the original dataframe)
    incorrect_df = incorrect_df[["row_id", "problem", "solution", "solution_id", "candidate_solution", "candidate_verification_reasoning", "candidate_verification_result"]]

    # ~~~ (3) Save Results ~~~
    print(f"Saving results to {SINK_PATH} and {SINK_PATH.with_suffix('.txt')}...")
    # i. Make sure the output directory exists
    SINK_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ii. Write the incorrect solutions to a CSV
    incorrect_df.to_csv(SINK_PATH, index=False)
    # iii. Write the solvable problem IDs to a text file
    solvable_problem_row_ids = set(incorrect_df["row_id"])
    with open(SINK_PATH.with_suffix(".txt"), "w") as f:
        for row_id in solvable_problem_row_ids:
            f.write(f"{row_id}\n")
    print(f"Saved results to {SINK_PATH} and {SINK_PATH.with_suffix('.txt')}.")


    


if __name__ == "__main__":
    print("Starting...")
    start = perf_counter()
    asyncio.run(main())
    print(f"Done! Elapsed: {perf_counter() - start:.2f}s")