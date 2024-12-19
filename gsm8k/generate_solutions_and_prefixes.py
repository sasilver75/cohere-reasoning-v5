import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_providers import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider
from utils import TokenBucket
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")

logger = logging.getLogger(__name__)

# Configuration
TOKEN_BUCKET = TokenBucket(400)
MODEL = OpenRouterModel.LLAMA_3_3_70B_INSTRUCT
PROVIDER = OPENROUTER_MODEL_PROVIDERS[MODEL]
COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}

lightweight_solution_prompt =  """
Solve the following math or reasoning problem, clearly presenting your reasoning and final answer.

Your input is as follows:
<problem>
{problem}
</problem>
"""

lightweight_verify_prompt = """
You will be given the following:
- A math problem
- The ground-truth answer to the problem
- A candidate solution, which contains both a reasoning process and a final answer

Your goal is to determine if the candidate solution is correct.
You will output a single word, "correct" or "incorrect", to indicate if the candidate solution is a valid solution to the problem.
You should only care about the final answer presented in the candidate solution.

Your input is as follows:
<problem>
{problem}
</problem>
<ground_truth_answer>
{answer}
</ground_truth_answer>
<candidate_solution>
{solution}
</candidate_solution>

Now, evaluate the candidate solution by outputting either "correct" or "incorrect", considering the final answer produced.
Do not output any other text than "correct" or "incorrect". Do not output any form of reasoning or explanation. Only output "correct" or "incorrect", this is absolutely critical.
"""

lightweight_prefix_prompt = """..."""  # TODO

def _get_solution_prompt(problem: str) -> str:
    return lightweight_solution_prompt.format(problem=problem)
    

def _get_verify_prompt(problem: str, answer: str, solution: str) -> str:
    return lightweight_verify_prompt.format(problem=problem, answer=answer, solution=solution)

def _get_prefix_prompt() -> str:
    # TODO
    ...


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_solution(problem: str, session: aiohttp.ClientSession) -> str:
    await TOKEN_BUCKET.acquire()
    
    async with session.post(
        COMPLETION_URL,
        headers=HEADERS,
        json = {
            "model": MODEL.value,
            "messages": [
                {"role": "user", "content": _get_solution_prompt(problem)},
            ],
            "temperature": .2,
            "top_p": 0.8,
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        return response_json["choices"][0]["message"]["content"]
    

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution(problem: str, answer: str, solution: str, session: aiohttp.ClientSession) -> bool:
    await TOKEN_BUCKET.acquire()
    
    async with session.post(
        COMPLETION_URL,
        headers=HEADERS,
        json = {
            "model": MODEL.value,
            "messages": [
                {"role": "user", "content": _get_verify_prompt(problem, answer, solution)}
            ],
            "temperature": 0,
            "top_k": 0  # Use Greedy for verification
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        response_content = response_json["choices"][0]["message"]["content"]
    
    verified = response_content.lower() == 'correct'
    print(f"Verification response is {response_content}, which is {verified}")
    return verified


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_prefix(problem: str, solution: str, session: aiohttp.ClientSession) -> str:
    await TOKEN_BUCKET.acquire()
    ...


async def process_row(row: pd.Series, session: aiohttp.ClientSession) -> dict:
    row_id = row["row_id"]
    problem = row["problem"]
    reasoning = row["reasoning"]
    answer = row["solution"]  # TODO: This is a little messy, can we just change what we name it in download_gsm8k.py?

    # Attempt to generate a correct, verified solution
    N_ATTEMPTS = 10
    for _ in range(N_ATTEMPTS):
        solution = await generate_solution(problem, session)
        verified = await verify_solution(problem, answer, solution, session)
        if verified:
            break
    else:
        raise ValueError(f"Failed to generate a correct solution after {N_ATTEMPTS} attempts for row {row['row_id']}")
    
    # prefix = await generate_prefix(problem, solution, session)
    prefix = "Placeholder"  # I want to look at some solutions to think about few-shots, then implement the prompt with examples.

    return {
        "row_id": row_id,
        "problem": problem,
        "ground_truth_reasoning": reasoning,
        "ground_truth_solution": answer,
        "verified_solution": solution,
        "prefix": prefix,
    }


async def main():
    # Load dataset
    print(f"Loading GSM8k datasset...")
    df = pd.read_csv("datasets/original/gsm8k.csv")
    # TODO: REMOVE THIS LINE BELOW
    df = df.head(25)
    # TODO: REMOVE THIS LINE ABOVE
    print(f"Loaded GSM8k datasset with {len(df)} rows and columns {list(df.columns)}")

    # Process the rows
    async with aiohttp.ClientSession() as session:
        acc = []
        tasks = [process_row(row, session) for _, row in df.iterrows()]
        with tqdm(total=len(tasks), desc="Processing rows") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                acc.append(result)
                pbar.update(1)

    
    # Convert the list of dicts to a dataframe and save
    df = pd.DataFrame(acc)
    filepath = "gsm8k/datasets/gsm8k_solutions_and_prefixes.csv"    
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

    




if __name__ == "__main__":
    asyncio.run(main())