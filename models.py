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
from utils import extract_verification_from_response


# Logger
logger = logging.getLogger(__name__)


# Load dotenv; this is used for API keys, which are used in these Helper classes
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
    async def get_completion(self, prompt: str) -> str:
        """
        Given a partially-completed solution, return the completed solution from the model.
        """
        ...


    



class CohereExperimentHelper(Helper):
    """
    Helper for a scenario in which we use Cohere models.
    This should expose methods to enable all functionality we need from Cohere.
    """
    def __init__(self, weak_completer: str = "command-r-03-2024", strong_completer: str = "command-r-plus-08-2024", strong_verifier: str = "command-r-plus-08-2024"):
        """
        args:
            weak_completer: str - The name of the Cohere model to use for weak completions
            strong_completer: str - The name of the Cohere model to use for strong completions
            strong_verifier: str - The name of the Cohere model to use for strong verifications
        """
        super().__init__(strong_completer)  # Use the strong completer's name

        if "COHERE_API_KEY" not in os.environ:
            raise ValueError("COHERE_API_KEY must be set in the environment")

        # Cohere chat endpoints have a 500/min rate limit; let's be conservative!
        self.cohere_bucket = TokenBucket(capacity=400, report_every=10)
        self.sync_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY")) # For completions, we need to use the V1 Client
        self.async_client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY")) # For full solutions, we can use the new V2 Asnyc Client
        self.strong_verifier = strong_verifier
        self.strong_completer = strong_completer
        self.weak_completer = weak_completer
    
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

        

    async def get_completion(self, prompt: str) -> str:
        ...
