import random
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
import re
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")

logger = logging.getLogger(__name__)

"""
Hey morning Sam

This file is just a copy of generate_stubs_and_perturbations
I think the idea is that we basically want to add a GENERATOR_MODEL and PERTURBER_MODEL column to each of our produced rows.

I think that our generator model should be the on-policy one (otherwise, we should use the same (maybe L3.1 405B? Model to make the generations)
I think right now that our perturber model should always be fixed (maybe L3.1 405B?)
- It'd be nice to have this one be one that's not one of the models being evaluated; IIRC L3.1 405B exists/works but isn't under evaluation because completions don't work; we don't need completions for either of these.

So then we'll have: 
row_id	problem	ground_truth_reasoning	ground_truth_solution generator_model	stub perturber_model	perturbed_stub_lm

So we want to support on-policy as well as off-policy generation...
- For off-policy, there will just be a single row_id = x that is completed by all models
- For on-policy, there will be a row_id = x row for each model under evaluation

And then the NEXT script we rewrite (NEW_generate_completions.py) will be the one that...
generates completions.

It can look at the dataframe that's read in and determine if it's on-policy or off-policy based on whether there's == 1 row_id for a given row_id.
If there's == 1, it's off policy completions, and it can just generate completions for all models.
If there's > 1, it's on-policy completinos, and it needs to generate completions for all models BUT each model has to only process its own completions.

The other thing to do is to wire up R7B to do the same thing. I think we can be a little messy with that.
"""

# Configuration
STUB_TOKENS = 100
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

# NOTE: Right now I'm not giving the perturber the context of the original problem, just giving the stub. I think there are arguments for and against this.
lightweight_perturb_prompt = """
You will be given a reasoning stub from a math word problem solution. Your task is to perturb this stub by introducing a single mathematical or reasoning error.

Your goal is to modify this stub in a way that will change the results of future reasoning and lead to an incorrect answer if followed. Follow these guidelines:

1. Make minimal changes to the stub. Do not add additional reasoning or try to solve for the right answer.
2. Focus on changing numbers, operators, or key words/phrases that will affect the result of the reasoning if it were played out.
3. Do not change the overall structure or length of the stub.
4. Do not apply any additional formatting (e.g. bolding, italicizing, etc.) your perturbation.

Good perturbations may include:
- Changing a number (e.g., from 40 hours to 45 hours)
- Switching an operator (e.g., from multiplication to division, or addition to subtraction)
- Altering a key phrase (e.g., from "60% of $5" to "60% more than $5")
- Reversing the order of operations (e.g., from "(2 + 3) * 4" to "2 + (3 * 4)")
- Changing units of measurement (e.g., from "meters" to "feet" without converting)
- Swapping comparison operators (e.g., from "greater than (>)" to "less than (<)")
- Modifying time relationships (e.g., from "per hour" to "per minute")
- Inverting fractions (e.g., from "3/4" to "4/3")
- Changing decimal places (e.g., from "1.5" to "15.0")

Bad perturbations include:
- Adding new steps or explanations
- Removing large portions of the stub
- Changing the reasoning stub entirely.


Here are some examples of good perturbations, along with explanations of what perturbation was applied:

<examples>

Original: "To solve this problem, let's break it down step by step.\n\n1. **First, calculate the cost of the first glass and the second glass.**\n   - The first glass costs $5.\n   - The second glass costs 60% of $5, which is $5 * 0.60 = $3.\n\n2. **Determine the pattern for the cost of the glasses.**\n   - Every odd-numbered glass (1st, 3rd, 5th"
Perturbed: "To solve this problem, let's break it down step by step.\n\n1. First, calculate the cost of the first glass and the second glass.\n - The first glass costs $5.\n - The second glass costs 60% more than $5, which is $5 1.60 = $8.\n\n2. Determine the pattern for the cost of the glasses.\n - Every odd-numbered glass (1st, 3rd, 5th"
Explanation: The perturbation changed "60% of $5" to "60% more than $5" and consequently updated the calculation from "$5 0.60 = $3" to "$5 1.60 = $8". It will lead to incorrect subsequent calculations.

Original: "To find out how many cups of feed Wendi needs to give her chickens in the final meal of the day, we first need to calculate the total amount of feed required for all the chickens for the entire day. Since each chicken requires 3 cups of feed per day, and there are 20 chickens, the total amount of feed needed per day is:\n\n20 chickens * 3 cups/chicken = 60 cups\n\nNow, let's calculate how much feed Wendi has already given to her"
Perturbed: "To find out how many cups of feed Wendi needs to give her chickens in the final meal of the day, we first need to calculate the total amount of feed required for all the chickens for the entire day. Since each chicken requires 2 cups of feed per day, and there are 20 chickens, the total amount of feed needed per day is:\n\n20 chickens * 2 cups/chicken = 40 cups\n\nNow, let's calculate how much feed Wendi has already given to her"
Explanation: The perturbation changed "3 cups of chicken" to "2 cups per chicken" and updated the calculation result accordingly ("60 cups" to "40 cups"). This will lead to an incorrect answer since the daily feed requirement is now underestimated.

Original: "To find the profit Josh made, we first need to calculate the new value of the house after the repairs, and then subtract the total amount he spent (the cost of the house plus the cost of repairs) from this new value.\n\n1. **Calculate the new value of the house:**\n   - Original value of the house = $80,000\n   - Increase in value due to repairs = 150% of $80,000\n   - Increase amount = 150% of $"
Perturbed: "To find the profit Josh made, we first need to calculate the new value of the house after the repairs, and then subtract the total amount he spent (the cost of the house plus the cost of repairs) from this new value.\n\n1. Calculate the new value of the house:\n - Original value of the house = $8,000\n - Increase in value due to repairs = 150% of $8,000\n - Increase amount = 150% of $"
Explanation: The perturbation removed a zero from the house value ($80,000 to $8,000), which will lead to an incorrect answer since the profit calculation is now underestimated.

Original: "To find Eliza's earnings for this week, we need to calculate her regular pay for the first 40 hours and her overtime pay for the additional 5 hours.\n\n1. **Calculate regular pay for the first 40 hours:**\n   - Regular hourly rate = $10\n   - Number of regular hours worked = 40 hours\n   - Regular pay = Regular hourly rate * Number of regular hours worked\n   - Regular pay = $10 * 40 = $400\n\n2."
Perturbed: "To find Eliza's earnings for this week, we need to calculate her regular pay for the first 40 hours and her overtime pay for the additional 5 hours.\n\n1. Calculate regular pay for the first 40 hours:\n - Regular hourly rate = $10\n - Number of regular hours worked = 40 hours\n - Regular pay = Regular hourly rate Number of regular hours worked\n - Regular pay = $10 40 = $500\n\n2."
Explanation: The perturbation changed the calculation result of "10 * 40 = $400" to "10 * 40 = $500". This will lead to an incorrect answer since the regular pay is now overestimated.

</examples>

Here is the reasoning stub that I want you to perturb:

<reasoning_stub>
{stub}
</reasoning_stub>

Apply one a single perturbation to the reasoning stub.

Provide your perturbed version of the reasoning stub inside <perturbed_stub> tags. 
Ensure that your perturbation is impactful and that it maintains the original structure and length of the stub as closely as possible.
"""


def _get_stub_prompt(problem: str) -> str:
    return lightweight_solution_prompt.format(problem=problem)
    

def _get_perturb_prompt(problem: str, stub: str) -> str:
    return lightweight_perturb_prompt.format(problem=problem, stub=stub)


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
    await TOKEN_BUCKET.acquire()
    
    async with session.post(
        COMPLETION_URL,
        headers=HEADERS,
        json = {
            "model": MODEL.value,
            "messages": [
                {"role": "user", "content": _get_stub_prompt(problem)},
            ],
            "temperature": .2,
            "top_p": 0.8,
            "max_tokens": STUB_TOKENS,
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
    await TOKEN_BUCKET.acquire()
    
    async with session.post(
        COMPLETION_URL,
        headers=HEADERS,
        json = {
            "model": MODEL.value,
            "messages": [
                {"role": "user", "content": _get_perturb_prompt(problem=problem, stub=stub)},
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
        return match.group(1).strip()
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
    row_id = row["row_id"]
    problem = row["problem"]
    reasoning = row["reasoning"]
    answer = row["solution"]  # TODO: This is a little messy, can we just change what we name it in download_gsm8k.py?

    stub = await generate_solution_stub(problem, session)
    perturbed_stub = await get_perturbed_stub(problem, stub, session)
    # perturbed_stub_deterministic = get_perturbed_stub_deterministic(stub)
    
    return {
        "row_id": row_id,
        "problem": problem,
        "ground_truth_reasoning": reasoning,
        "ground_truth_solution": answer,
        "stub": stub,
        "perturbed_stub_lm": perturbed_stub,
        # "perturbed_stub_deterministic": perturbed_stub_deterministic
    }



async def main():
    # Load dataset
    print(f"Loading GSM8k datasset...")
    df = pd.read_csv("datasets/original/gsm8k.csv")
    # TODO: REMOVE THIS LINE BELOW
    df = df.head(50)
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
    filepath = "gsm8k/datasets/gsm8k_stubs_and_perturbations.csv"    
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

    




if __name__ == "__main__":
    asyncio.run(main())