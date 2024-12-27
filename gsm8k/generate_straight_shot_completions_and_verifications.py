import asyncio
import aiohttp
import os
from gsm_models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider, CohereModel
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
from gsm_utils import TokenBucket
import cohere
import gsm_prompts

"""
The goal of this file is just to generate completions and verifications for all problems in the GSM8K dataset.
As a pass@1 baseline that we can compare the perturbed performances against.
This uses the same prompts as the other files, but just generates completions/solutions to the problems directly, with no perturbations.
"""

# Load environment variables
load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")
if not "COHERE_API_KEY" in os.environ:
    raise ValueError("COHERE_API_KEY must be set in the environment")

# CONFIGURATION
# ~ Experiment parameters
N_PROBLEMS = None  # None = All problems
INPUT_FILEPATH = "gsm8k/datasets/original/gsm8k.csv"
OUTPUT_FILEPATH = "gsm8k/datasets/gsm8k_straight_shot_solutions.csv"
MODELS = [
    OpenRouterModel.QWEN_2_5_72B_INSTRUCT,
    CohereModel.COHERE_R7B,
    OpenRouterModel.MISTRAL_NEMO_12B_INSTRUCT,
    OpenRouterModel.QWEN_QWQ_32B_PREVIEW,
    OpenRouterModel.GEMMA_2_27B_INSTRUCT,
    OpenRouterModel.LLAMA_3_3_70B_INSTRUCT,
]
VERIFIER_MODEL = OpenRouterModel.DEEPSEEK_3

# ~ Rate limiting
OPENROUTER_TOKEN_BUCKET = TokenBucket(350, "OpenRouter")
COHERE_TOKEN_BUCKET = TokenBucket(400, "Cohere")

# ~ Request configuration
COHERE_SYNC_CLIENT = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
OPENROUTER_COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_completion_openrouter(session: aiohttp.ClientSession, model: OpenRouterModel, problem: str) -> str:
    """Get a completion from an OpenRouter model"""
    await OPENROUTER_TOKEN_BUCKET.acquire()

    async with session.post(
            OPENROUTER_COMPLETION_URL,
            headers=OPENROUTER_HEADERS,
            json={
                "model": model.value,
                "messages": [
                    {"role": "user", "content": gsm_prompts.get_solution_prompt(problem)}
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
            if "error" in response_json:
                print("Provider error: ", response_json)
            return response_json["choices"][0]["message"]["content"]

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_completion_cohere(model: CohereModel, problem: str) -> str:
    """Get a completion from a Cohere model"""
    await COHERE_TOKEN_BUCKET.acquire()

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        completion_response = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: COHERE_SYNC_CLIENT.chat(
                    model=model.value,
                    message=gsm_prompts.get_solution_prompt_cohere(problem, ""), # No stub for a straight shot completion
                    temperature=0.2,
                    p=0.8,
                    raw_prompting=True,
                )
            ),
            timeout=90
        )
    return completion_response.text.replace("<|END_RESPONSE|>", "")

async def get_completion(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, problem: str) -> str:
    """Route completion requests to appropriate provider"""
    if isinstance(model, OpenRouterModel):
        return await get_completion_openrouter(session, model, problem)
    elif isinstance(model, CohereModel):
        return await get_completion_cohere(model, problem)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution(session: aiohttp.ClientSession, problem: str, answer: int, candidate_solution: str) -> bool:
    """Verify a solution using the verification model"""
    await OPENROUTER_TOKEN_BUCKET.acquire()

    async with session.post(
        OPENROUTER_COMPLETION_URL,
        headers=OPENROUTER_HEADERS,
        json={
            "model": VERIFIER_MODEL.value,
            "messages": [
                {"role": "user", "content": gsm_prompts.get_verification_prompt(problem, answer, candidate_solution)},
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
        if "error" in response_json:
            print("Provider error: ", response_json)
        response_content = response_json["choices"][0]["message"]["content"]

    verified = response_content.lower() == 'correct'
    print(f"Verification response is {response_content}, which is {verified}")
    return verified

async def test_single_problem(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, row: pd.Series) -> dict:
    """Process a single problem"""
    problem_id = row["problem_id"]
    problem = row["problem"]
    answer = row["answer"]

    print(f"Testing model {model.value} on problem {problem_id}")

    # Get completion and verification
    solution = await get_completion(session, model, problem)
    solution_verified = await verify_solution(session, problem, answer, solution)

    return {
        "problem_id": problem_id,
        "problem": problem,
        "answer": answer,
        "solution_model": model.value,
        "solution_model_provider": OPENROUTER_MODEL_PROVIDERS[model].value if isinstance(model, OpenRouterModel) else "Cohere",
        "solution": solution,
        "solution_verified": solution_verified
    }

async def test_model(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, df: pd.DataFrame) -> list[dict]:
    """Test all problems for a given model"""
    semaphore = asyncio.Semaphore(15)
    
    async def rate_limited_test(row: pd.Series):
        async with semaphore:
            return await test_single_problem(session, model, row)
    
    # Create all tasks
    tasks = [rate_limited_test(row) for _, row in df.iterrows()]
    
    # Process tasks as they complete with progress bar
    valid_results = []
    with tqdm(total=len(tasks), desc=f"Testing {model.value}") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result is not None:
                    valid_results.append(result)
            except Exception as e:
                print(f"Task failed with error: {str(e)}")
            pbar.update(1)
    
    return valid_results

async def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILEPATH)
    print(f"Loaded dataset with {len(df)} rows and columns {list(df.columns)} from {INPUT_FILEPATH}")

    if N_PROBLEMS is not None:
        df = df.head(N_PROBLEMS)
        print(f"Using first {N_PROBLEMS} problems")

    acc = []
    async with aiohttp.ClientSession() as session:
        for model in MODELS:
            print(f"\nStarting tests for model: {str(model.value)}")
            results = await test_model(session, model, df)
            acc.extend(results)
            print(f"Completed testing model: {str(model.value)}")

    print(f"Total results collected: {len(acc)}")
    print("Saving results...")
    df = pd.DataFrame(acc)
    df = df.sort_values(by=["solution_model", "problem_id"], ascending=[True, True])
    filepath = OUTPUT_FILEPATH
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    asyncio.run(main())