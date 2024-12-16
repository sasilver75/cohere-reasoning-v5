from abc import ABC, abstractmethod
from enum import Enum
import os
from utils import TokenBucket
import cohere
from dotenv import load_dotenv
import asyncio
import pandas as pd
import prompts
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from utils import extract_verification_from_response, get_naive_prefix
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import random
import requests
import aiohttp

# Logger
logger = logging.getLogger(__name__)


# Load dotenv; this is used for API keys via python-dotenv; Keys are used in these Helper classes
load_dotenv()

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class Helper(ABC):
    """
    Abstract base class for model helpers.
    Allows us to abstract away the details of getting solutions or completions from a model,
    while also handling rate limiting.

    You should never need to use more than a single ModelHelper. TokenBucket rate limiters are scoped
    to a single ModelHelper instance, so it's important that you don't use (eg) two ModelHelper instances that both
    target the same API-provider, since your token buckets can't communicate with each other.
    """
    def __init__(self, model_name: str):
        """
        args:
            model_name: str - The name of the model under evaluation. Used for file naming, etc.
        """
        # TODO: Do I really want to use this for anything?
        self.model_name = model_name  # Used for files, etc.

    @abstractmethod
    async def get_solution(self, row: pd.Series, use_weak_completer: bool = False) -> str:
        """
        Return a straight shot solution from the model.
        """
        ...
    
    @abstractmethod
    async def get_verification(self, candidate_solution: str, row: pd.Series) -> tuple[bool, str]:
        """
        Given a solution, return a verification from the model.
        """
        ...
    
    @abstractmethod
    async def get_prefix_and_completion(self, row: pd.Series) -> tuple[str, str]:
        """
        Given a partially-completed solution, return the completed solution from the model.
        """
        ...


class DummyExperimentHelper(Helper):
    """
    A dummy helper that simulates network latencies, rate limiting, and occasional failures
    without making actual API calls. Useful for testing the overall flow of experiments.
    """
    def __init__(self, 
                 min_latency: float = 0.5, 
                 max_latency: float = 2.0,
                 bucket_capacity: int = 400,
                 bucket_rate: Optional[float] = None,
                 failure_rate: float = 0.1):  # 10% chance of failure on API clals
        """
        args:
            min_latency: float - Minimum simulated latency in seconds
            max_latency: float - Maximum simulated latency in seconds
            bucket_capacity: int - Capacity of the token bucket
            bucket_rate: Optional[float] - Rate of token replenishment
            failure_rate: float - Probability of request failure (0-1)
        """
        super().__init__("dummy-model")
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.failure_rate = failure_rate
        self.token_bucket = TokenBucket(capacity=bucket_capacity, rate=bucket_rate)

    async def _simulate_latency(self):
        """Simulate a random network latency and possible failure"""
        delay = random.uniform(self.min_latency, self.max_latency)
        await asyncio.sleep(delay)
        
        if random.random() < self.failure_rate:
            raise Exception("Simulated API failure")

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_solution(self, row: pd.Series, use_weak_completer: bool = False) -> str:
        await self.token_bucket.acquire()
        await self._simulate_latency()
        return "<DummySolution>"

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_verification(self, candidate_solution: str, row: pd.Series) -> tuple[bool, str]:
        await self.token_bucket.acquire()
        await self._simulate_latency()
        return random.choice([True, False]), "<DummyVerificationReasoning>"

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_prefix_and_completion(self, row: pd.Series) -> tuple[str, str]:
        await self.token_bucket.acquire()
        await self._simulate_latency()
        return "<DummyPrefix>", "<DummyCompletion>"



class CohereExperimentHelper(Helper):
    """
    Helper for a scenario in which we use Cohere models as the Strong, Weak, and Verifier models.
    This should expose methods to enable all functionality we need from Cohere.
    """
    def __init__(self, bucket_capacity: int = 400, bucket_rate: Optional[float] = None, bucket_report_every: int = 50, bucket_verbose: bool = False, strong_completer: str = "command-r-plus-08-2024", prefix_size: float = 0.7):
        """
        args:
            bucket_capacity: int - The capacity of the token bucket for rate limiting (this should be a conservative interpretation of the per-minute rate limit for provider)
            report_every: int - How often to report the state of the token bucket
            strong_completer: str - The name of the Cohere model to use for strong completions
        """
        super().__init__(strong_completer) # Use the strong completer's name

        if "COHERE_API_KEY" not in os.environ:
            raise ValueError("COHERE_API_KEY must be set in the environment")

        # Cohere chat endpoints have a 500/min rate limit; let's be conservative!
        self.cohere_bucket = TokenBucket(capacity=bucket_capacity, name="CohereBucket", rate=bucket_rate, report_every=bucket_report_every, verbose=bucket_verbose)  # Used to handle rate limiting/concurrency
        self.sync_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY")) # For completions, we need to use the V1 Client
        self.async_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY")) # For full solutions, we can use the new V2 Asnyc Client
        self.strong_verifier = "command-r-plus-08-2024"
        self.strong_completer = "command-r-plus-08-2024"
        self.weak_completer =  "command-r-03-2024"
        self.prefix_size = prefix_size
    
    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_solution(self, row: pd.Series, use_weak_completer: bool = False) -> str:
        """
        Given a row from the source dataframe, generate a "straight-shot" solution from our model under evaluation.
        args:
            row: pd.Series - A row from the source dataframe (eg cn_k12_math_problems.csv, from NuminaMath-CoT)
        returns:
            solution: str - The generated solution
        """
        await self.cohere_bucket.acquire()
        response = await asyncio.wait_for(
            self.async_client.chat(
                model=self.strong_completer if not use_weak_completer else self.weak_completer,
                messages=[{
                    "role": "user",
                    "content": prompts.STRAIGHT_SHOT_SOLUTION_PROMPT.format(
                        problem=row["problem"]
                    ),
                }],
                temperature=0.3,
            ),
            timeout=90,
        )
        return response.message.content[0].text

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_verification(self, candidate_solution: str, row: pd.Series) -> tuple[bool, str]:
        """
        Given a candidate solution and a row from the source dataframe, verify the candidate solution.
        args:
            candidate_solution: str - The candidate solution to verify
            row: pd.Series - A row from the source dataframe (eg cn_k12_math_problems.csv, from NuminaMath-CoT)
        returns:
            verified: bool - Whether the candidate solution was verified as correct
            verification_reasoning: str - The reasoning for the verification result
        """
        problem = row["problem"]
        solution = row["solution"]
        await self.cohere_bucket.acquire()
        response = await asyncio.wait_for(
            self.async_client.chat(
                model=self.strong_verifier,
                messages=[
                    {
                        "role": "user",
                        "content": prompts.VERIFY_SOLUTION_PROMPT.format(
                            problem=problem,
                            solution=solution,
                            candidate_solution=candidate_solution,
                        ),
                    }
                ],
                temperature=0.0,
            ),
            timeout=90,
        )
        return extract_verification_from_response(response.message.content[0].text)

        
    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_prefix_and_completion(self, row: pd.Series) -> tuple[str, str]:
        """
        Get a prefix from the incorrect solution and generate a completion using the sync Cohere API.
        Use ThreadPoolExecutor to prevent the synchronous Cohere API call from blocking the event loop.

        TODO: It's possible that we might be making too many threads here?. Watch to see how much memory is used.
        If the memory usage is too high, we could create a single threadpool in the __init__ with a limit on the number of 
        threads that we make, and then use that same threadpool here to limit memory usage. Edit: It doesn't seem like an issue so far.
        """
        prefix = get_naive_prefix(row["candidate_solution"], self.prefix_size)
        
        user_turn = prompts.COMPLETION_PROMPT_USER.format(problem=row["problem"])
        assistant_turn = prompts.COMPLETION_PROMPT_ASSISTANT.format(prefix=prefix)
        
        await self.cohere_bucket.acquire()

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            completion_response = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: self.sync_client.chat(
                        model=self.weak_completer,
                        message=prompts.COMPLETION_TEMPLATE.format(
                            user_turn=user_turn,
                            assistant_turn=assistant_turn,
                        ),
                        temperature=0.3,
                        raw_prompting=True,
                    )
                ),
                timeout=90
            )
            
        return prefix, completion_response.text
    

class OpenRouterProvider(Enum):
    DEEPINFRA = "DeepInfra"
    HYPERBOLIC = "Hyperbolic"
    NOVITA = "Novita"


# TODO: I should probably be passing a "provider" argument here, so that we can make sure that the provider that we use for agiven model is in original precision.
class OpenRouterExperimentHelper(Helper):
    """
    Helper for a scenario in which we use the following models:
    - Qwen 2.5 72B Instruct as the strong completer (model under evaluation)
    - Cohere Command R Plus 08 2024 as the strong verifier
    - Cohere Command R 03 2024 as the weak completer

    Enscapsulates logic for interacting both with the Cohere API and the OpenRouter API, thorugh whic we access Qwen 2.5.
    Note that the "provider" is going to be fixed to DeepInfra, which serves Qwen 2.5 72B Instruct in its original bf16 precision.
    """
    def __init__(self, strong_completer: str, provider: OpenRouterProvider, cohere_bucket_capacity: int = 400, cohere_report_every: int = 10, cohere_bucket_verbose: bool = False, openrouter_bucket_capacity: int = 300, openrouter_report_every: int = 10, openrouter_bucket_verbose: bool = False, prefix_size: float = 0.7):
        super().__init__(strong_completer)

        if "COHERE_API_KEY" not in os.environ:
            raise ValueError("COHERE_API_KEY must be set in the environment")

        if "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError("OPENROUTER_API_KEY must be set in the environment")

        self.strong_completer = strong_completer
        self.strong_verifier = "command-r-plus-08-2024"
        self.weak_completer = "command-r-03-2024"

        # Note that Cohere's rate limit for chat endpoints is 500/min; We set conservatively at 400/minute.
        self.cohere_bucket = TokenBucket(capacity=cohere_bucket_capacity, name="CohereBucket", report_every=cohere_report_every, verbose=cohere_bucket_verbose)
        # Note that OpenRouter's rate limit is dynamic based on the number of credits in your account (lol), at 1 request/second per $ in your account. Let's conservatively set it at 300/minute.
        self.openrouter_bucket = TokenBucket(capacity=openrouter_bucket_capacity, name="OpenRouterBucket", report_every=openrouter_report_every, verbose=openrouter_bucket_verbose)

        self.cohere_async_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))  # Needed for verification

        self.prefix_size = prefix_size
        self.provider = provider

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _request_openrouter_solution(self, prompt: str) -> str:
        # Get permission to send a request
        await self.openrouter_bucket.acquire()

        # Send request with timeout
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=OPENROUTER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application_json"
                },
                json={
                    "model": self.strong_completer,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "provider": {
                        "order": [self.provider.value],
                        "allow_fallbacks": False,
                    }
                },
                timeout=90
            ) as response:
                response = await response.json()
                return response["choices"][0]["message"]["content"]

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _request_openrouter_completion(self, user_turn: str, assistant_turn: str) -> str:
        # Get permission to send a request
        await self.openrouter_bucket.acquire()

        async with aiohttp.ClientSession() as session:
            # Send the request with timeout
            async with session.post(
                url=OPENROUTER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.strong_completer,
                    "messages": [
                        {"role": "user", "content": user_turn},
                        {"role": "assistant", "content": assistant_turn}
                    ],
                    "provider": {
                        "order": ["DeepInfra"],
                        "allow_fallbacks": False,
                    }
                },
                timeout=90
            ) as response:
                response_json = await response.json()
                return response_json["choices"][0]["message"]["content"]

        

    async def get_solution(self, row: pd.Series, use_weak_completer: bool = False) -> str:
        """
        Given a row from the source dataframe, generate a "straight-shot" solution from our model under evaluation.
        args:
            row: pd.Series - A row from the source dataframe (eg cn_k12_math_problems.csv, from NuminaMath-CoT)
        returns:
            solution: str - The generated solution
        """
        # Prepare information
        problem = row["problem"]

        # TODO: Split here whether we're using use_weak_completer or not. The weak completer ones should use CR via CohereAPI; Make a new helper function for that.

        # Robustly get and return solution
        return await self._request_openrouter_solution(prompts.STRAIGHT_SHOT_SOLUTION_PROMPT.format(problem=problem))


    async def get_prefix_and_completion(self, row: pd.Series) -> tuple[str, str]:
        """
        Get a prefix from the incorrect solution and generate a completion using our strong completer.
        """
        # Prepare information
        prefix = get_naive_prefix(row["candidate_solution"], self.prefix_size)
        user_turn = prompts.COMPLETION_PROMPT_USER.format(problem=row["problem"])
        assistant_turn = prompts.COMPLETION_PROMPT_ASSISTANT.format(prefix=prefix)

        # Robustly get the completion
        completion = await self._request_openrouter_completion(user_turn, assistant_turn)

        # Return information
        return prefix, completion
    
    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_verification(self, candidate_solution: str, row: pd.Series) -> tuple[bool, str]:
        """
        Given a candidate solution and a row from the source dataframe, verify the candidate solution.
        args:
            candidate_solution: str - The candidate solution to verify
            row: pd.Series - A row from the source dataframe (eg cn_k12_math_problems.csv, from NuminaMath-CoT)
        returns:
            verified: bool - Whether the candidate solution was verified as correct
            verification_reasoning: str - The reasoning for the verification result

        Note: This is a copy-paste of the CohereExperimentHelper get_verification method.
        """
        problem = row["problem"]
        solution = row["solution"]
        await self.cohere_bucket.acquire()
        response = await asyncio.wait_for(
            self.cohere_async_client.chat(
                model=self.strong_verifier,
                messages=[
                    {
                        "role": "user",
                        "content": prompts.VERIFY_SOLUTION_PROMPT.format(
                            problem=problem,
                            solution=solution,
                            candidate_solution=candidate_solution,
                        ),
                    }
                ],
                temperature=0.0,
            ),
            timeout=90,
        )
        return extract_verification_from_response(response.message.content[0].text)