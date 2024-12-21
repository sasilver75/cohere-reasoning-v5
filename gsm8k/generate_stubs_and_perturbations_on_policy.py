import random
import sys
import os
from models import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider, MODELS_UNDER_EVALUATION
from utils import TokenBucket
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import asyncio
import re
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

"""

"""

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")
if not "COHERE_API_KEY" in os.environ:
    raise ValueError("COHERE_API_KEY must be set in the environment")



# CONFIGURATION


# END OF CONFIGURATION


logger = logging.getLogger(__name__)




"""
This generates "on-policy" stubs and perturbations for the models under evaluation.
"""




