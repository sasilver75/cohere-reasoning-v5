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

# For rate limiting
TOKEN_BUCKET = TokenBucket(400)

lightweight_generate_prompt = """..."""
lightweight_verify_prompt = """..."""
lightweight_prefix_prompt = """..."""

def _get_generate_prompt() -> str:
    ...

def _get_verify_prompt() -> str:
    ...

def _get_prefix_prompt() -> str:
    ...

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_solution() -> str:
    await TOKEN_BUCKET.acquire()
    ...

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution() -> bool:
    await TOKEN_BUCKET.acquire()
    ...

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_prefix() -> str:
    await TOKEN_BUCKET.acquire()
    ...


async def process_row(row: pd.Series) -> dict:

    # Attempt to generate a correct, verified solution
    N_ATTEMPTS = 10
    for _ in range(N_ATTEMPTS):
        solution = await generate_solution(row)
        verified = await verify_solution(solution)
        if verified:
            break
    else:
        raise ValueError(f"Failed to generate a correct solution after {N_ATTEMPTS} attempts for row {row['row_id']}")
    

    prefix = await generate_prefix(row, solution)

    return {
        "row_id": row["row_id"],
        "problem": row["problem"],
        "ground_truth_reasoning": row["reasoning"],
        "ground_truth_solution": row["solution"],
        "verified_solution": solution,
        "prefix": prefix,
    }


async def main():
    print(f"Loading GSM8k datasset...")
    df = pd.read_csv("datasets/original/gsm8k.csv")
    print(f"Loaded GSM8k datasset with {len(df)} rows and columns {list(df.columns)}")

    acc = []
    tasks = [process_row(row) for _, row in df.iterrows()]
    with tqdm(total=len(tasks), desc="Processisng rows") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            acc.append(result)
            pbar.update(1)

    
    # Convert the list of dicts to a dataframe and save
    df = pd.DataFrame(acc)
    df.to_csv("gsm8k/datsets/gsm8k_solutions_and_prefixes.csv", index=False)

    




if __name__ == "__main__":
    asyncio.run(main())