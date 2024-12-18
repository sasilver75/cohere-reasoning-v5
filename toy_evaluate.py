import asyncio
import time
import aiohttp
from model_providers import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider
import pandas as pd
from tqdm import tqdm
from toy_data import TOY_PROBLEMS
import requests
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")

logger = logging.getLogger(__name__)


lightweight_verify_prompt = """
You are given a math problem, its ground-truth solution, and a candidate solution to that problem, and your goal is to verify that the candidate solution is correct.

You will be given the following information:
- The problem
- The ground-truth solution to the problem
- The candidate solution

You will output a single word, "correct" or "incorrect", to indicate if the candidate solution is a valid solution to the problem.
You should not mark a candidate solution as incorrect because its reasoning does not match the solution. You should only care about the final answer produced.

Your input is as follows:
<problem>
{problem}
</problem>
<ground_truth_solution>
{solution}
</ground_truth_solution>
<candidate_solution>
{candidate_solution}
</candidate_solution>

Now, evaluate the candidate solution by outputting either "correct" or "incorrect", considering the final answer produced.
Do not output any other text than "correct" or "incorrect". Do not output any form of reasoning or explanation. Only output "correct" or "incorrect", this is absolutely critical.
"""

lightweight_detect_prompt = """
You are a math tutor. You are given a math problem and a ground-truth solution to that problem.
You are given a "reasoning prefix," which represents an incomplete reasoning process containng an error, which the candidate solution was given.
A model under evaluation is given the prefix, and asked to complete the reasoning to solve the problem.
Your goal is to determine if the model under evaluation made a reasoning correction to the reasoning prefix to solve the problem.

These reasoning corrections MAY take the form of generations similar to "Wait, I made a mistake" or "However, this is a common misconception" or "However, I made an error in my reasoning" or "Oops!" or "However, there's a small error in the calculation."
There are other ways to express reasoning corrections. Use your best judgement to determine if the model under evaluation made a reasoning correction to the reasoning prefix to solve the problem.

You will be given the following information:
- The problem
- The solution to the problem
- A prefix that indicates the reasoning that the model under evaluation was given, and asked to complete to solve the problem
- The completion of the prefix by the model under evaluation, which should solve the problem

You will output either "correction" or "no correction", depending on whether the model under evaluation made a reasoning correction to the reasoning prefix during its completion to solve the problem.
Do not output any other text than "correction" or "no correction". Do not output any form of reasoning or explanation. Only output "correction" or "no correction", this is absolutely critical.

<problem>
{problem}
</problem>
<ground_truth_solution>
{solution}
</ground_truth_solution>
<prefix>
{prefix}
</prefix>
<completion>
{completion}
</completion>
"""

def _get_verification_prompt(problem: str, solution: str, candidate_solution: str) -> str:
    return lightweight_verify_prompt.format(problem=problem, solution=solution, candidate_solution=candidate_solution)

def _get_detection_prompt(problem: str, solution: str, prefix: str, completion: str) -> str:
    return lightweight_detect_prompt.format(problem=problem, solution=solution, prefix=prefix, completion=completion)


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_completion(session: aiohttp.ClientSession, model: OpenRouterModel, provider: OpenRouterProvider, problem: str, prefix: str) -> str:
    """
    Get a completion to a prefix, from a model using async request
    """
    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": prefix}
            ],
            "provider": {
                "order": [provider.value],
                "allow_fallbacks": False,
            },
            "temperature": 0.2,
            "top_p": 0.8
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        return response_json["choices"][0]["message"]["content"]

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution(session: aiohttp.ClientSession, problem: str, solution: str, candidate_solution: str) -> bool:
    """
    Async version of solution verification
    """
    model = OpenRouterModel.LLAMA_3_3_70B_INSTRUCT
    provider = OpenRouterProvider.NOVITA

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": _get_verification_prompt(problem, solution, candidate_solution)},
            ],
            "provider": {
                "order": [provider.value],
                "allow_fallbacks": False,
            },
            "temperature": 0,
            "top_k": 0
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        response_content = response_json["choices"][0]["message"]["content"]

    verified = response_content.lower() == 'correct'
    print(f"Verification response is {response_content}, which is {verified}")
    return verified

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def detect_correction(session: aiohttp.ClientSession, problem: str, solution: str, prefix: str, completion: str) -> bool:
    """
    Async version of correction detection
    """
    model = OpenRouterModel.LLAMA_3_3_70B_INSTRUCT
    provider = OpenRouterProvider.NOVITA

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": _get_detection_prompt(problem, solution, prefix, completion)},
            ],
            "provider": {
                "order": [provider.value],
                "allow_fallbacks": False,
            },
            "temperature": 0,
            "top_k": 0
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        response_content = response_json["choices"][0]["message"]["content"]

    correction_detected = response_content.lower() == 'correction'
    print(f"Detection response is {response_content}, which is {correction_detected}")
    return correction_detected


async def test_single_problem(session: aiohttp.ClientSession, model: OpenRouterModel, provider: OpenRouterProvider, 
                            problem_data: tuple, idx: int) -> dict:
    """Handle a single problem evaluation"""
    problem, prefix, solution = problem_data
    print(f"Testing model {model.value} on problem {idx}")
    
    try:
        completion = await get_completion(session, model, provider, problem, prefix)
        candidate_solution = f"{prefix} {completion}"
        
        # Run verification and detection concurrently
        verified, correction_detected = await asyncio.gather(
            verify_solution(session, problem, solution, candidate_solution),
            detect_correction(session, problem, solution, prefix, completion)
        )

        return {
            "model": str(model.value),  # Convert to string explicitly
            "provider": str(provider.value),  # Convert to string explicitly
            "problem_id": idx,
            "problem": problem,
            "solution": solution,
            "prefix": prefix,
            "completion": completion,
            "candidate_solution": candidate_solution,
            "verified": verified,
            "correction_detected": correction_detected
        }
    except Exception as e:
        print(f"Error processing problem {idx} with model {str(model.value)}: {str(e)}")
        # Return a partial result with error information
        return {
            "model": str(model.value),
            "provider": str(provider.value),
            "problem_id": idx,
            "problem": problem,
            "error": str(e),
            "status": "failed"
        }

async def test_model(session: aiohttp.ClientSession, model: OpenRouterModel, provider: OpenRouterProvider) -> list[dict]:
    """Test all problems for a given model concurrently"""
    semaphore = asyncio.Semaphore(30)
    
    async def rate_limited_test(problem_data, idx):
        async with semaphore:
            return await test_single_problem(session, model, provider, problem_data, idx)
    
    # Create all tasks
    tasks = [
        rate_limited_test(problem_data, idx)
        for idx, problem_data in enumerate(TOY_PROBLEMS)
    ]
    
    # Run tasks concurrently and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any None results and handle exceptions
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with error: {str(result)}")
        elif result is not None:
            valid_results.append(result)
    
    return valid_results

async def async_main():
    acc = []
    async with aiohttp.ClientSession() as session:
        for model, provider in OPENROUTER_MODEL_PROVIDERS.items():
            try:
                print(f"\nStarting tests for model: {str(model.value)}")
                results = await test_model(session, model, provider)
                acc.extend(results)
                print(f"Completed testing model: {str(model.value)}")
            except Exception as e:
                print(f"Error testing model {str(model.value)}: {str(e)}")
                continue

    print(f"Total results collected: {len(acc)}")
    print("Saving results...")
    df = pd.DataFrame(acc)
    df.to_csv("toy_evaluate.csv", index=False)
    print("Results saved to toy_evaluate.csv")

def main():
    print(f"Number of toy problems: {len(TOY_PROBLEMS)}")
    print(f"Number of models to test: {len(OPENROUTER_MODEL_PROVIDERS)}")
    print(f"API key present: {'OPENROUTER_API_KEY' in os.environ}")
    
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()