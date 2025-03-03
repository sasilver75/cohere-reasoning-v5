from multiprocessing import process
import random
import sys
import os
from deterministic_perturbations import get_perturbed_stub_deterministic
from gsm_models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, CohereModel
from gsm_utils import TokenBucket
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import asyncio
import re
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import cohere
import gsm_prompts

"""
This generates "on-policy" stubs for each model under evaluation, with off-policy perturbations applied using
DeepSeek 2.5, which isn't a model under evaluation (because no OpenRouter providers support assistant prefilling for it).
"""

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")
if not "COHERE_API_KEY" in os.environ:
    raise ValueError("COHERE_API_KEY must be set in the environment")


# TODO: MOVE THE SOURCE DATASET TO THE CONFIGURATION UP HERE; IT MIGHT BE GSM-SYMBOLIC
# CONFIGURATION
# ~ Experiment parameters
STUB_N_TOKENS = 100
N_PROBLEMS = None  # None means "All" problems

PERTURB_MODEL = OpenRouterModel.LLAMA_3_1_405B_INSTRUCT
MODELS = [
    # OpenRouterModel.QWEN_2_5_72B_INSTRUCT,
    # CohereModel.COHERE_R7B,
    # OpenRouterModel.MISTRAL_NEMO_12B_INSTRUCT,
    # OpenRouterModel.QWEN_QWQ_32B_PREVIEW,
    # OpenRouterModel.GEMMA_2_27B_INSTRUCT,
    # OpenRouterModel.LLAMA_3_3_70B_INSTRUCT,
    OpenRouterModel.DEEPSEEK_R1
]

INPUT_FILENAME = "gsm8k/datasets/original/gsm8k_matched_gsm_symbolic.csv"
OUTPUT_FILENAME = "gsm8k/datasets/gsm8k_stubs_and_perturbations_on_policy.csv"

# ~ Rate limiting
OPENROUTER_TOKEN_BUCKET = TokenBucket(250)
COHERE_TOKEN_BUCKET = TokenBucket(400)

# ~ Things for making requests
COHERE_V2_CLIENT = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])  # We don't need to generate completions, so we can use the async v2 cilent.
OPENROUTER_COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}
# END OF CONFIGURATION

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_solution_stub_openrouter(problem: str, model: OpenRouterModel, session: aiohttp.ClientSession) -> str:
    """Generate a solution stub using an OpenRouter model"""
    await OPENROUTER_TOKEN_BUCKET.acquire()
    
    async with session.post(
        OPENROUTER_COMPLETION_URL,
        headers=OPENROUTER_HEADERS,
        json = {
            "model": model.value,
            "messages": [
                {"role": "user", "content": gsm_prompts.get_solution_prompt(problem)},
            ],
            "provider": {
                "order": [OPENROUTER_MODEL_PROVIDERS[model].value],
                "allow_fallbacks": False,
            },
            "temperature": .2,
            "top_p": 0.8,
            "max_tokens": STUB_N_TOKENS,
            "include_reasoning": True
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        reasoning = response_json["choices"][0]["message"]["reasoning"] if "reasoning" in response_json["choices"][0]["message"] else ""
        content = response_json["choices"][0]["message"]["content"]
        if reasoning and content:
            return f"{reasoning} \n {content}"
        elif reasoning and not content:
            # If the truncation means we didn't make it to content
            return reasoning
        else:
            # e.g. non-reasoning model
            return content

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_solution_stub_cohere(problem: str, model: CohereModel) -> str:
    """Generate a solution stub using a Cohere model"""
    await COHERE_TOKEN_BUCKET.acquire()
    
    response = await COHERE_V2_CLIENT.chat(
        model=model.value,
        messages=[
            {"role": "user", "content": gsm_prompts.get_solution_prompt(problem)}
        ],
        temperature=0.2,
        p=0.8,
        max_tokens=STUB_N_TOKENS
    )
    return response.message.content[0].text

async def generate_solution_stub(problem: str, model: OpenRouterModel | CohereModel, session: aiohttp.ClientSession) -> str:
    """Route to appropriate stub generator based on model type"""
    if isinstance(model, OpenRouterModel):
        return await generate_solution_stub_openrouter(problem, model, session)
    elif isinstance(model, CohereModel):
        return await generate_solution_stub_cohere(problem, model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_perturbed_stub(problem: str, stub: str, session: aiohttp.ClientSession) -> str:
    """Perturb a stub solution using the perturb model"""
    await OPENROUTER_TOKEN_BUCKET.acquire()
    
    async with session.post(
        OPENROUTER_COMPLETION_URL,
        headers=OPENROUTER_HEADERS,
        json = {
            "model": PERTURB_MODEL.value,
            "messages": [
                {"role": "user", "content": gsm_prompts.get_perturb_prompt(problem=problem, stub=stub)},
            ],
            "provider": {
                "order": [OPENROUTER_MODEL_PROVIDERS[PERTURB_MODEL].value],
                "allow_fallbacks": False,
            },
            "temperature": 0,
            "top_k": 0,  # Greedy
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        
    response_content = response_json["choices"][0]["message"]["content"]
    pattern = r'<perturbed_stub>\s*(.*?)\s*</perturbed_stub>'
    match = re.search(pattern, response_content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Couldn't find perturbed stub in response: {response_content}")

async def process_row(row: pd.Series, model: OpenRouterModel | CohereModel, session: aiohttp.ClientSession) -> dict:
    """Process a single row for a specific model"""
    problem_id = row["problem_id"]
    problem = row["problem"]
    answer = row["answer"]

    # Generate an on-policy stub
    stub = await generate_solution_stub(problem, model, session)
    
    # Get the LM-based and deterministic perturbations of the stub
    perturbed_stub_lm = await get_perturbed_stub(problem, stub, session)
    # perturbed_stub_deterministic, perturbation_type = get_perturbed_stub_deterministic(stub)
    
    return {
        "problem_id": problem_id,
        "problem": problem,
        "answer": answer,
        "stub_model": model.value,
        "stub_model_provider": OPENROUTER_MODEL_PROVIDERS[model].value if isinstance(model, OpenRouterModel) else "Cohere",
        "perturb_model": PERTURB_MODEL.value,
        "perturb_model_provider": OPENROUTER_MODEL_PROVIDERS[PERTURB_MODEL].value,
        "stub": stub,
        "perturbed_stub_lm": perturbed_stub_lm,
        # "perturbed_stub_deterministic": perturbed_stub_deterministic,
        # "perturbed_stub_deterministic_type": perturbation_type
    }

async def main():
    # Load dataset
    print(f"Loading GSM8k dataset...")
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Loaded GSM8k dataset with {len(df)} rows and columns {list(df.columns)} from {INPUT_FILENAME}")

    if N_PROBLEMS is not None:  
        print(f"Using first {N_PROBLEMS} problems of {len(df)} problems")
        df = df.head(N_PROBLEMS)    
    
    semaphore = asyncio.Semaphore(15)
    
    async def rate_limited_test(row: pd.Series):
        async with semaphore:
            return await process_row(row, model, session)
    
    # Process each model
    all_results = []
    async with aiohttp.ClientSession() as session:
        for model in MODELS:
            print(f"\nProcessing model: {model.value}")
            tasks = [rate_limited_test(row) for _, row in df.iterrows()]
            
            with tqdm(total=len(tasks), desc=f"Processing rows for {model.value}") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    all_results.append(result)
                    pbar.update(1)
    
    # Convert results to dataframe and save
    df_results = pd.DataFrame(all_results)

    # Make sure that we're ordered by stub_model problem_id asc.
    df_results = df_results.sort_values(by=["stub_model", "problem_id"], ascending=[True, True])
    
    df_results.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    asyncio.run(main())


