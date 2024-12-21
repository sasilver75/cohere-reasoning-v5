import random
import sys
import os
from gsm8k.gsm_models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider, MODELS_UNDER_EVALUATION
from utils import TokenBucket
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import asyncio
import re
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import cohere

"""
This generates "on-policy" stubs for each model under evaluation, with off-policy perturbations applied using
DeepSeek 2.5, which isn't a model under evaluation (because no OpenRouter providers support assistant prefilling for it).
"""

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")
if not "COHERE_API_KEY" in os.environ:
    raise ValueError("COHERE_API_KEY must be set in the environment")



# CONFIGURATION
# ~ Experiment parameters
STUB_N_TOKENS = 100
N_PROBLEMS = 100  # None means "All" problems

# ~ Rate limiting
OPENROUTER_TOKEN_BUCKET = TokenBucket(400)
COHERE_TOKEN_BUCKET = TokenBucket(400)

# ~ Things for making requests
COHERE_CLIENT = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])  # We don't need to generate completions, so we can use the async v2 cilent.
OPENROUTER_COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}
# END OF CONFIGURATION


logger = logging.getLogger(__name__)




"""
This generates "on-policy" stubs and perturbations for the models under evaluation.
"""




