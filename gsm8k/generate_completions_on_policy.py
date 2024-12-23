import asyncio
import aiohttp
import os
from gsm_models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, CohereModel
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
from gsm_utils import TokenBucket
import cohere
import gsm_prompts
from concurrent.futures import ThreadPoolExecutor

"""
For every model under evaluation, generate completions and verifications for the stubs that IT generated
in generate_stubs_and_perturbations_on_policy.py.

Results can be viewed using view_completions_on_policy.py (TODO)
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
MODELS = [
    OpenRouterModel.QWEN_2_5_72B_INSTRUCT,
    CohereModel.COHERE_R7B,
    CohereModel.COHERE_CRP,
    OpenRouterModel.MISTRAL_NEMO_12B_INSTRUCT,
    OpenRouterModel.QWEN_QWQ_32B_PREVIEW,
    OpenRouterModel.GEMMA_2_27B_INSTRUCT,
    OpenRouterModel.LLAMA_3_3_70B_INSTRUCT,
]
VERIFIER_MODEL = OpenRouterModel.DEEPSEEK_2_5_1210_INSTRUCT

# ~ Rate limiting
OPENROUTER_TOKEN_BUCKET = TokenBucket(350, "OpenRouter")
COHERE_TOKEN_BUCKET = TokenBucket(400, "Cohere")

# ~ Things for making requests
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
async def get_completion_openrouter(session: aiohttp.ClientSession, model: OpenRouterModel, problem: str, perturbed_stub: str) -> str:
    """Get a completion from an OpenRouter model"""
    await OPENROUTER_TOKEN_BUCKET.acquire()

    async with session.post(
        OPENROUTER_COMPLETION_URL,
        headers=OPENROUTER_HEADERS,
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": gsm_prompts.get_solution_prompt(problem)},
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
    """Get a completion from a Cohere model"""
    await COHERE_TOKEN_BUCKET.acquire()

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        completion_response = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: COHERE_SYNC_CLIENT.chat(
                    model=model.value,
                    message=gsm_prompts.get_solution_prompt_cohere(problem, perturbed_stub),
                    temperature=0.2,
                    p=0.8,
                    raw_prompting=True,
                )
            ),
            timeout=90
        )
    # Remove the <|END_RESPONSE|> token if present
    return completion_response.text.replace("<|END_RESPONSE|>", "")

async def get_completion(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, problem: str, perturbed_stub: str) -> str:
    """Route to appropriate completion generator based on model type"""
    if isinstance(model, OpenRouterModel):
        return await get_completion_openrouter(session, model, problem, perturbed_stub)
    elif isinstance(model, CohereModel):
        return await get_completion_cohere(model, problem, perturbed_stub)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution(session: aiohttp.ClientSession, problem: str, answer: int, candidate_solution: str) -> bool:
    """Verify a solution using the verifier model"""
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
        response_content = response_json["choices"][0]["message"]["content"]
        
    verified = response_content.lower() == 'correct'
    print(f"Verification response is {response_content}, which is {verified}")
    return verified

async def test_single_problem(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, row: pd.Series) -> dict:
    """Process a single problem for a model using its own stub"""
    # Get all existing fields from input row
    result = row.to_dict()
    
    print(f"Testing model {model.value} on problem {result['problem_id']}")

    # Get completions for both perturbation types in parallel
    perturbed_stub_lm_completion, perturbed_stub_deterministic_completion = await asyncio.gather(
        get_completion(session, model, result["problem"], result["perturbed_stub_lm"]),
        get_completion(session, model, result["problem"], result["perturbed_stub_deterministic"])
    )
    
    # Get verifications for both perturbation types in parallel
    perturbed_stub_lm_verified, perturbed_stub_deterministic_verified = await asyncio.gather(
        verify_solution(
            session, 
            result["problem"], 
            result["answer"], 
            f"{result['perturbed_stub_lm']}{perturbed_stub_lm_completion}"
        ),
        verify_solution(
            session, 
            result["problem"], 
            result["answer"], 
            f"{result['perturbed_stub_deterministic']}{perturbed_stub_deterministic_completion}"
        )
    )

    # Add new completion-related fields
    result.update({
        "completer_model": model.value,
        "completer_model_provider": OPENROUTER_MODEL_PROVIDERS[model].value if isinstance(model, OpenRouterModel) else "Cohere",
        
        # LM perturbation results
        "perturbed_stub_lm_completion": perturbed_stub_lm_completion,
        "perturbed_stub_lm_solution_verified": perturbed_stub_lm_verified,
        
        # Deterministic perturbation results
        "perturbed_stub_deterministic_completion": perturbed_stub_deterministic_completion,
        "perturbed_stub_deterministic_solution_verified": perturbed_stub_deterministic_verified,
        
        # Verifier model info
        "verifier_model": VERIFIER_MODEL.value,
        "verifier_model_provider": OPENROUTER_MODEL_PROVIDERS[VERIFIER_MODEL].value,
    })

    return result

async def test_model(session: aiohttp.ClientSession, model: OpenRouterModel | CohereModel, df: pd.DataFrame) -> list[dict]:
    """Test all problems for a given model using its own stubs"""
    # Filter dataframe to only include rows where this model generated the stub
    model_df = df[df["stub_model"] == model.value].copy()
    print(f"Testing {len(model_df)} problems for model {model.value}")
    
    semaphore = asyncio.Semaphore(30)
    
    async def rate_limited_test(row: pd.Series):
        async with semaphore:
            return await test_single_problem(session, model, row)
    
    tasks = [rate_limited_test(row) for _, row in model_df.iterrows()]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with error: {str(result)}")
        elif result is not None:
            valid_results.append(result)
    
    return valid_results

async def main():
    print(f"Loading dataset...")
    df = pd.read_csv("gsm8k/datasets/gsm8k_stubs_and_perturbations_on_policy.csv")
    print(f"Loaded dataset with {len(df)} rows and columns {list(df.columns)}")

    if N_PROBLEMS is not None:
        print(f"Using first {N_PROBLEMS} problems of {len(df)} problems")
        df = df.head(N_PROBLEMS)

    acc = []
    async with aiohttp.ClientSession() as session:
        for model in MODELS:
            print(f"\nStarting tests for model: {str(model.value)}")
            results = await test_model(session, model, df)
            acc.extend(results)
            print(f"Completed testing model: {str(model.value)}")


    
    print("Saving results...")
    df = pd.DataFrame(acc)

    # sorting by stub should be the same thing as sorting by completer_model, since it's on_policy.
    df = df.sort_values(by=["completer_model", "problem_id"], ascending=[True, True])
    
    filepath = "gsm8k/datasets/gsm8k_completions_on_policy.csv"
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

if __name__ == "__main__":
    asyncio.run(main())
