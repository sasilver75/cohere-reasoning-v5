import asyncio
import time


class TokenBucket:
    def __init__(self, capacity: int, report_every: int | None = None):
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
        self.rate = capacity / 60  # Tokens/second added to bucket
        self.last_update = time.time()  # Last time tokens were added to the bucket

        # Lock for mutual exclusion if this ever to be used in a multithreaded context
        self.lock = asyncio.Lock()

        # For reporting
        self.request_count = 0  # I can't imagine that this would realistically overflow
        self.report_every = report_every

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
                await asyncio.sleep(wait_time)
                # Recalculate tokens after sleep
                now = time.time()
                new_tokens = (now - self.last_update) * self.rate
                self.tokens = min(self.capacity, new_tokens)
                self.last_update = now

            # "Spend" a token
            self.tokens -= 1
            self.request_count += 1

            if self.report_every and self.request_count % self.report_every == 0:
                print(f"TokenBucket request count: {self.request_count}")


def get_naive_prefix(solution: str) -> str:
    ... # 0.7

