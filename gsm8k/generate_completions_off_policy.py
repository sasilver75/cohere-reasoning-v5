import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import aiohttp
import sys
import os
from models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider, CohereModel
import pandas as pd
from tqdm import tqdm
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
from utils import TokenBucket
import cohere

# Load in .env file and confirm needed keys are present
load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")
if not "COHERE_API_KEY" in os.environ:
    raise ValueError("COHERE_API_KEY must be set in the environment")


# CONFIGURATION
N_PROBLEMS = 2  # None = All; It's fine if N_PROBLEMS is greater than the number of problems in the source dataset
MODELS = {
    CohereModel.COHERE_R7B,
    # CohereModel.COHERE_CRP,
    # OpenRouterModel.MISTRAL_NEMO_12B_INSTRUCT,
    # OpenRouterModel.QWEN_QWQ_32B_PREVIEW,
    # OpenRouterModel.GEMMA_2_27B_INSTRUCT,
    # OpenRouterModel.LLAMA_3_3_70B_INSTRUCT,
    # OpenRouterModel.QWEN_2_5_72B_INSTRUCT,
    
}
VERIFIER_MODEL = OpenRouterModel.DEEPSEEK_2_5_1210_INSTRUCT
OPENROUTER_TOKEN_BUCKET = TokenBucket(350, "OpenRouter")
COHERE_TOKEN_BUCKET = TokenBucket(400, "Cohere")
COHERE_SYNC_CLIENT = cohere.Client(api_key=os.getenv("COHERE_API_KEY")) # For completions, we need to use the V1 Client
# END OF CONFIGURATION


logger = logging.getLogger(__name__)


lightweight_verify_prompt = """
You are given a math problem, its ground-truth solution, and a candidate solution to that problem, and your goal is to verify that the candidate solution is correct.

You will be given the following information:
- The problem
- The answer to the problem
- The candidate solution, which includes reasoning and a final answer

You will output a single word, "correct" or "incorrect", to indicate if the candidate solution is a valid solution to the problem.
You should not mark a candidate solution as incorrect because of any error in its reasoning. You should only care about whether the final answer produced matches the answer.

Your input is as follows:
<problem>
{problem}
</problem>
<answer>
{answer}
</answer>
<candidate_solution>
{candidate_solution}
</candidate_solution>

Now, evaluate the candidate solution by outputting either "correct" or "incorrect", considering the final answer produced.
Do not output any other text than "correct" or "incorrect". Do not output any form of reasoning or explanation. Only output "correct" or "incorrect", this is absolutely critical.
"""

lightweight_problem_prompt = """
Solve the following math or reasoning problem, clearly presenting your reasoning and final answer.

Your input is as follows:
<problem>
{problem}
</problem>
"""

lightweight_problem_cohere_prompt = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{problem}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{stub}"""

def _get_verification_prompt(problem: str, answer: str, candidate_solution: str) -> str:
    return lightweight_verify_prompt.format(problem=problem, answer=answer, candidate_solution=candidate_solution)

def _get_problem_prompt(problem: str) -> str:
    return lightweight_problem_prompt.format(problem=problem)

def _get_problem_cohere_prompt(problem: str, stub: str) -> str:
    return lightweight_problem_cohere_prompt.format(problem=problem, stub=stub)

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_completion_openrouter(session: aiohttp.ClientSession, model: OpenRouterModel, problem: str, perturbed_stub: str) -> str:
    """
    Robustly get a completion for an OpenRouter model
    """
    await OPENROUTER_TOKEN_BUCKET.acquire()

    async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": model.value,
                "messages": [
                    {"role": "user", "content": _get_problem_prompt(problem)},
                    {"role": "assistant", "content": perturbed_stub}
                ],
                "provider": {
                    "order": [OPENROUTER_MODEL_PROVIDERS[model].value],
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
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_completion_cohere(model: CohereModel, problem: str, perturbed_stub: str) -> str:
    """
    Get a completion for a Cohere model
    Although the Cohere v1 client (the only one that supports completions is synchronous), we'd still like to parallelize it.
    We can use ThreadPoolExecutor to prevent these synchronous calls from blocking the event loop.    
    """
    await COHERE_TOKEN_BUCKET.acquire()

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        completion_response = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: COHERE_SYNC_CLIENT.chat(
                    model=model.value,
                    message=_get_problem_cohere_prompt(problem, perturbed_stub),
                    temperature=0.2,
                    p=0.8,
                    raw_prompting=True,
                )
            ),
            timeout=90
        )
    return completion_response.text


async def get_completion(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, problem: str, perturbed_stub: str) -> str:
    """
    Get a completion to a prefix, from a model using async request
    ROUTES between OpenRouter and Cohere based on model type 
    """
    if isinstance(model, OpenRouterModel):
        return await get_completion_openrouter(session, model, problem, perturbed_stub)
    elif isinstance(model, CohereModel):
        return await get_completion_cohere(model, problem, perturbed_stub)
    else:
        logger.error(f"FATAL: Unknown model type: {type(model)}")
        raise ValueError(f"Unknown model type: {type(model)}")

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution(session: aiohttp.ClientSession, problem: str, answer: int, candidate_solution: str) -> bool:
    """
    Asynchronous verification of LM answer using an OpenRouter model not under evaluation (VERIFIER_MODEL, likely DeepSeek 2.5, which doesn't support completions)
    """

    await OPENROUTER_TOKEN_BUCKET.acquire()

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": VERIFIER_MODEL.value,
            "messages": [
                {"role": "user", "content": _get_verification_prompt(problem, answer, candidate_solution)},
            ],
            "provider": {
                "order": [OPENROUTER_MODEL_PROVIDERS[VERIFIER_MODEL].value],
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

async def test_single_problem(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, row: pd.Series) -> dict:
    """Handle a single problem evaluation"""
    problem_id: int = row["problem_id"]
    problem: str = row["problem"]
    answer: int = row["ground_truth_solution"] # eg "18"
    stub: str = row["stub"]
    perturbed_stub_lm: str = row["perturbed_stub_lm"]

    print(f"Testing model {model.value} on problem {problem_id}")

    # TODO: Consider wrapping all of the below in a try/except block, and return "placeholder" values if the tasks actually fail; This would only happen if we 

    # Get the completion using the LM-perturbed stub
    perturbed_stub_lm_completion = await get_completion(session, model, problem, perturbed_stub_lm)
    
    # Get the verification and correction detection results
    perturbed_stub_lm_verified = await verify_solution(session, problem, answer, f"{perturbed_stub_lm}{perturbed_stub_lm_completion}") # Does the full solution match the answer?
    
    return {
        "model": str(model.value),
        "provider": str(OPENROUTER_MODEL_PROVIDERS[model].value) if isinstance(model, OpenRouterModel) else "Cohere",
        "problem_id": problem_id,
        "problem": problem,
        "answer": answer,
        "stub": stub,
        "perturbed_stub_lm": perturbed_stub_lm,
        "perturbed_stub_lm_completion": perturbed_stub_lm_completion,
        "perturbed_stub_lm_solution_verified": perturbed_stub_lm_verified,
        #"perturbed_stub_deterministic": perturbed_stub_deterministic
        #...
    }

async def test_model(session: aiohttp.ClientSession, model: OpenRouterModel, df: pd.DataFrame) -> list[dict]:
    """Test all problems for a given model concurrently"""
    semaphore = asyncio.Semaphore(30)
    
    async def rate_limited_test(row: pd.Series):
        async with semaphore:
            return await test_single_problem(session, model, row)
    
    # Create all tasks
    tasks = [
        rate_limited_test(row)
        for _, row in df.iterrows()
    ]
    
    # TODO(SAM): Below could use some polish (use as_completed, etc.)

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

    print(f"Loading dataset...")
    df = pd.read_csv("gsm8k/datasets/gsm8k_stubs_and_perturbations_off_policy.csv")
    print(f"Loaded dataset with {len(df)} rows and columns {list(df.columns)}")

    print(f"Testing {N_PROBLEMS if N_PROBLEMS is not None else "all"} problems, out of {len(df)} available problems")
    if N_PROBLEMS is not None:
        df = df.head(N_PROBLEMS)

    acc = []
    async with aiohttp.ClientSession() as session:
        for model in MODELS:
            try:
                print(f"\nStarting tests for model: {str(model.value)}")
                results = await test_model(session, model, df)
                acc.extend(results)
                print(f"Completed testing model: {str(model.value)}")
            except Exception as e:
                print(f"Error testing model {str(model.value)}: {str(e)}")
                continue

    print(f"Total results collected: {len(acc)}")
    print("Saving results...")
    df = pd.DataFrame(acc)
    filepath = "gsm8k/datasets/gsm8k_completions_off_policy.csv"
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def main():
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