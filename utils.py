import asyncio
import time
from typing import Optional
import re
import pandas as pd

class TokenBucket:
    def __init__(self, capacity: int, rate: Optional[float] = None,report_every: int | None = None, verbose: bool = False):
        """
        A simple token bucket rate limiter.
        args:
            rate: float - The rate at which tokens are added to the bucket, in tokens/second
            capacity: int - The maximum number of tokens in the bucket
            report_every: int | None - If provided, will print the number of requests that have beenmade every `report_every` requests
        """
        # Core bucket attributes
        self.capacity = capacity  # Maximum number of tokens in bucket
        self.tokens = capacity  # Current number of tokens in bucket
        self.rate = rate if rate is not None else capacity / 60  # Tokens/second added to bucket (capped at capacity)
        self.last_update = time.time()  # Last time tokens were added to the bucket

        # Lock for mutual exclusion if this ever to be used in a multithreaded context
        self.lock = asyncio.Lock()

        # For reporting
        self.request_count = 0  # I can't imagine that this would realistically overflow
        self.report_every = report_every
        self.verbose = verbose

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            # Only one coroutine can be in here at a time; the block is mutually exclusive
            # Add new tokens based on time passed since last acquire call
            now = time.time()
            new_tokens = (now - self.last_update) * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_update = now

            # If we need tokens, wait for them to be added, then update token counts
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                if self.verbose:
                    print(f"Sleeping for {wait_time:.2f} seconds to wait for tokens to be added")
                await asyncio.sleep(wait_time)
                # Recalculate tokens after sleep
                now = time.time()
                new_tokens = (now - self.last_update) * self.rate
                self.tokens = min(self.capacity, self.tokens + new_tokens)
                self.last_update = now

            # "Spend" a token
            if self.verbose:
                print(f"Spending a token: {self.tokens} before spending")
            self.tokens -= 1
            self.request_count += 1

            if self.report_every and self.request_count % self.report_every == 0:
                print(f"TokenBucket request count: {self.request_count}")


def get_naive_prefix(solution: str, prefix_size: float) -> str:
    """
    Given a solution, return a prefix of the solution that is `prefix_size` proportion of the solution.
    args:
        solution: str - The solution to get a prefix of
        prefix_size: float - The proportion of the solution to use as a prefix (eg 0.0 to 1.0)
    returns:
        prefix: str - The prefix of the solution
    """
    words = solution.split()
    n_words = len(words)
    n_words_to_take = max(1, int(prefix_size * n_words))
    return " ".join(words[:n_words_to_take])


def extract_verification_from_response(
    verification_response: str,
) -> tuple[str, bool]:
    """
    Given a verification response, return whether the verifiation response indicates that the candidate solution was correct.
    There shouldn't be any extraction errors. If there's a problem, we should raise an exception (which, outside, will trigger a retry).

    args:
        verification_response: str - The response from the completer model
    returns:
        verified: bool - Whether the candidate solution was verified as correct
        verification_reasoning: str - The reasoning for the verification result
    """
    # Extract REASONING
    verification_reasoning_pattern = (
        r"<verification_reasoning>(.*?)</verification_reasoning>"
    )
    match = re.search(verification_reasoning_pattern, verification_response, re.DOTALL)
    if not match:
        print(f"Could not parse verification reasoning for {verification_response}")
        raise Exception(
            f"Could not parse verification reasoning for {verification_response}"
        )
    verification_reasoning = match.group(1).strip()

    # Extract RESULT
    verification_pattern = r"<verification_result>(.*?)</verification_result>"
    match = re.search(verification_pattern, verification_response, re.DOTALL)
    if not match:
        print(f"Could not parse verification result for {verification_response}")
        raise Exception(
            f"Could not parse verification result for {verification_response}"
        )
    verified = match.group(1).strip().lower() == "correct"

    return verified, verification_reasoning
