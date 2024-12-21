import random
import sys
import os
from gsm_models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import asyncio
import re
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import gsm_prompts
from gsm_utils import TokenBucket

"""
This generates "off-policy" stubs and perturbations from GSM8k, using the off-policy model of DeepSeek 2.5.
DeepSeek 2.5 is not a model under evaluation (because no OpenRouter providers support assistant prefilling for it).
"""

# Load in .env file and confirm needed keys are present
load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")


# CONFIGURATION
# ~ Experiment parameters
STUB_N_TOKENS = 100
N_PROBLEMS = 100  # None means "All" problems
PREFIX_AND_PERTURB_MODEL = OpenRouterModel.DEEPSEEK_2_5_1210_INSTRUCT

# ~ Rate limiting
OPENROUTER_TOKEN_BUCKET = TokenBucket(400)

# ~ Things for making requests
OPENROUTER_COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}
# END OF CONFIGURATION

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_solution_stub(problem: str, session: aiohttp.ClientSession) -> str:
    """
    Generate the first STUB_TOKENS tokesn of a solution to a problem.
    """
    await OPENROUTER_TOKEN_BUCKET.acquire()
    
    async with session.post(
        OPENROUTER_COMPLETION_URL,
        headers=OPENROUTER_HEADERS,
        json = {
            "model": PREFIX_AND_PERTURB_MODEL.value,
            "messages": [
                {"role": "user", "content": gsm_prompts.get_solution_prompt(problem)},
            ],
            "temperature": .2,
            "top_p": 0.8,
            "max_tokens": STUB_N_TOKENS,
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
async def get_perturbed_stub(problem: str, stub: str, session: aiohttp.ClientSession) -> str:
    """
    Perturb a stub solution
    """
    await OPENROUTER_TOKEN_BUCKET.acquire()
    
    async with session.post(
        OPENROUTER_COMPLETION_URL,
        headers=OPENROUTER_HEADERS,
        json = {
            "model": PREFIX_AND_PERTURB_MODEL.value,
            "messages": [
                {"role": "user", "content": gsm_prompts.get_perturb_prompt(problem=problem, stub=stub)},
            ],
            "temperature": 0,
            "top_k": 0,  # Greedy
        },
        timeout=60
    ) as response:
        response_json = await response.json()

    response_content = response_json["choices"][0]["message"]["content"]

    # Extract the content between the <perturbed_stub> tags
    pattern = r'<perturbed_stub>\s*(.*?)\s*</perturbed_stub>'
    match = re.search(pattern, response_content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Couldn't find perturbed stub in response: {response_content}")
    

def get_perturbed_stub_deterministic(stub: str) -> str:
    """
    Perturb a stub solution deterministically
    """

    possible_perturbations = []
    
    # Check for numbers
    if re.search(r'\d+(?:\.\d+)?', stub):
        possible_perturbations.append('number_modification')
        
    # Check for operators
    if any(op in stub for op in ['+', '-', '*', '/', '>', '<']):
        possible_perturbations.append('operator_swap')
        
    # Check for units
    if any(unit in stub.lower() for unit in ['hours', 'dollars', 'meters']):
        possible_perturbations.append('unit_swap')
        
    # Check for percentages
    if re.search(r'\d+(?:\s)?(?:%|percent)', stub):
        possible_perturbations.append('percentage_modification')
        
    # Check for fractions
    if re.search(r'\d+/\d+', stub):
        possible_perturbations.append('fraction_inversion')
    
    if not possible_perturbations:
        return stub  # No perturbation possible
        
    # Choose and apply a perturbation strategy
    strategy = random.choice(possible_perturbations)
    
    # Apply the chosen strategy
    if strategy == 'number_modification':
        # Implementation here
        pass
    elif strategy == 'operator_swap':
        # Implementation here
        pass
    # ... etc for other strategies
    
    return stub



async def process_row(row: pd.Series, session: aiohttp.ClientSession) -> dict:
    problem_id = row["problem_id"]
    problem = row["problem"]
    answer = row["solution"]  # TODO: This is a little messy, can we just change what we name it in download_gsm8k.py?

    stub = await generate_solution_stub(problem, session)
    perturbed_stub = await get_perturbed_stub(problem, stub, session)
    # perturbed_stub_deterministic = get_perturbed_stub_deterministic(stub)
    
    return {
        "problem_id": problem_id,
        "problem": problem,
        "answer": answer,
        "stub_and_perturb_model": PREFIX_AND_PERTURB_MODEL.value,
        "stub_and_perturb_model_provider": OPENROUTER_MODEL_PROVIDERS[PREFIX_AND_PERTURB_MODEL].value,
        "stub": stub,
        "perturbed_stub_lm": perturbed_stub,
        # perturbed_stub_deterministic: ...
        # perturbed_stub_deterministic_perturbation_type: ...
    }



async def main():
    # Load dataset
    print(f"Loading GSM8k datasset...")
    df = pd.read_csv("datasets/original/gsm8k.csv")
    print(f"Loaded GSM8k datasset with {len(df)} rows and columns {list(df.columns)}")


    if N_PROBLEMS is not None:  
        df = df.head(N_PROBLEMS)    
        print(f"Using first {N_PROBLEMS} problems of {len(df)} problems")

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

    # Make sure that the rows are ordered by problem_id asc; just aesthetics :)
    df = df.sort_values(by="problem_id", ascending=True)

    filepath = "gsm8k/datasets/gsm8k_stubs_and_perturbations_off_policy.csv"    
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

    




if __name__ == "__main__":
    asyncio.run(main())