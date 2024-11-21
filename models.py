from abc import ABC, abstractmethod
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


# Logger
logger = logging.getLogger(__name__)


# Load dotenv; this is used for API keys via python-dotenv; Keys are used in these Helper classes
load_dotenv()


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
    async def get_solution(self, prompt: str) -> str:
        """
        Return a straight shot solution from the model.
        """
        ...
    
    @abstractmethod
    async def get_verification(self, prompt: str) -> tuple[bool, str]:
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
    # TODO: It might be nice to have a helper that just returns dummy data (with realistic "latencies" on requests). This could be nice to test with.
    ...



class CohereExperimentHelper(Helper):
    """
    Helper for a scenario in which we use Cohere models as the Strong, Weak, and Verifier models.
    This should expose methods to enable all functionality we need from Cohere.
    """
    def __init__(self, bucket_capacity: int = 400, report_every: int = 10, strong_completer: str = "command-r-plus-08-2024", prefix_size: float = 0.7):
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
        self.cohere_bucket = TokenBucket(capacity=bucket_capacity, report_every=report_every)  # Used to handle rate limiting/concurrency
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
    async def get_solution(self, row: pd.Series) -> str:
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
                model=self.strong_completer,
                messages=[{
                    "role": "user",
                    "content": prompts.STRAIGHT_SHOT_SOLUTION_PROMPT.format(
                        problem=row["problem"]
                    ),
                }],
                temperature=0.3,
            ),
            timeout=60,
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
            timeout=60,
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

        TODO: It's possible that we might be making too many threads here. Watch to see how much memory is used.
        If the memory usage is too high, we could create a single threadpool in the __init__ with a limit on the number of 
        threads that we make, and then use that same threadpool here to limit memory usage.
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
                timeout=60
            )
            
        return prefix, completion_response.text

class OpenRouterExperimentalHelper(Helper):
    """
    Helper for a variety of experimental scenarios in which we use a model frmo OpenRouter as the strong verifier, 
    and models from Cohere as the weak completer and strong verifier.
    
    Encapsulates the logic for interacting with the OpenRouter API, as well as the logic for rate limiting.
    """
    def __init__(self, strong_completer: str, cohere_bucket_capacity: int = 400, cohere_report_every: int = 10, openrouter_bucket_capacity: int = 400, openrouter_report_every: int = 10):
        super().__init__(strong_completer)
        self.strong_completer = strong_completer
        self.strong_verifier = "command-r-plus-08-2024"
        self.weak_completer = "command-r-03-2024"
        self.cohere_bucket = TokenBucket(capacity=cohere_bucket_capacity, report_every=cohere_report_every)
        self.openrouter_bucket = TokenBucket(capacity=openrouter_bucket_capacity, report_every=openrouter_report_every)
        # TODO: Clients for each?
    
    async def get_solution(self, row: pd.Series) -> str:
        ...
    
    async def get_verification(self, candidate_solution: str, row: pd.Series) -> tuple[bool, str]:
        ...

    async def get_completion(self, prompt: str) -> str:
        ...