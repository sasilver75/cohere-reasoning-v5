import asyncio
import time
from typing import Optional
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class TokenBucket:
    def __init__(self, capacity: int, name:str = "TokenBucket", rate: Optional[float] = None, report_every: int | None = None, verbose: bool = False):
        """
        A simple token bucket rate limiter.
        args:
            rate: float - The rate at which tokens are added to the bucket, in tokens/second
            capacity: int - The maximum number of tokens in the bucket
            report_every: int | None - If provided, will print the number of requests that have beenmade every `report_every` requests
        """
        # Core bucket attributes
        self.name = name
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
                    print(f"{self.name}: Sleeping for {wait_time:.2f} seconds to wait for tokens to be added")
                await asyncio.sleep(wait_time)
                # Recalculate tokens after sleep
                now = time.time()
                new_tokens = (now - self.last_update) * self.rate
                self.tokens = min(self.capacity, self.tokens + new_tokens)
                self.last_update = now

            # Only increment the request count and update tokens if we actually give out access
            if self.tokens >= 1:
                if self.verbose:
                    print(f"{self.name}: Spending a token: {self.tokens} before spending")
                self.tokens -= 1
                self.request_count += 1

                if self.report_every and self.request_count % self.report_every == 0:
                    print(f"{self.name}: TokenBucket request count: {self.request_count}")


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
    response: str,
) -> tuple[str, bool]:
    """
    Given a verification response, return whether the verifiation response indicates that the candidate solution was correct.
    There shouldn't be any extraction errors. If there's a problem, we should raise an exception (which, outside, will trigger a retry).

    args:
        response: str - The response from the completer model
    returns:
        verified: bool - Whether the candidate solution was verified as correct
        verification_reasoning: str - The reasoning for the verification result
    """
    # Extract REASONING
    verification_reasoning_pattern = (
        r"<verification_reasoning>(.*?)</verification_reasoning>"
    )
    match = re.search(verification_reasoning_pattern, response, re.DOTALL)
    if not match:
        print(f"Could not parse verification reasoning for {response}")
        raise Exception(
            f"Could not parse verification reasoning for {response}"
        )
    verification_reasoning = match.group(1).strip()

    # Extract RESULT
    verification_pattern = r"<verification_result>(.*?)</verification_result>"
    match = re.search(verification_pattern, response, re.DOTALL)
    if not match:
        print(f"Could not parse verification result for {response}")
        raise Exception(
            f"Could not parse verification result for {response}"
        )
    verified = match.group(1).strip().lower() == "correct"

    return verified, verification_reasoning


def plot_recovery_figures(df: pd.DataFrame):
    """
    Plot the recovery figures for a given experiment, given the "interesting_problems_completed.csv" dataframe
    """
    # Calculate overall recovery statistics
    total_completions = len(df)
    correct_completions = df['completion_verification_result'].sum()
    incorrect_completions = total_completions - correct_completions

    # Calculate per-problem recovery rates
    per_problem_stats = df.groupby('row_id')['completion_verification_result'].agg(
        recovery_rate=lambda x: 100 * x.mean()  # Convert to percentage
    ).reset_index()

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    plt.rcParams.update({'font.size': 14})

    # Left subplot - Overall recovery counts
    bars = ax1.bar(['Incorrect', 'Correct'],
                  [incorrect_completions, correct_completions],
                  color=['lightcoral', 'lightgreen'])

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=14)

    ax1.set_title('Recovery Results Distribution', fontsize=16, pad=20)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Right subplot - Recovery rates histogram
    bin_edges = [-5, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105]  # Shifted bins to center bars
    
    counts, bins, patches = ax2.hist(per_problem_stats['recovery_rate'],
                                   bins=bin_edges,
                                   color='skyblue',
                                   edgecolor='black', 
                                   weights=np.ones(len(per_problem_stats)) / len(per_problem_stats) * 100,
                                   rwidth=0.6)  # Reduced bar width

    # Set evenly spaced ticks for all values 0 through 100
    tick_positions = list(range(0, 101, 10))
    tick_labels = [str(x) for x in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=12)

    # Add count labels on histogram bars
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    for count, x in zip(counts, bin_centers):
        actual_count = int(count * len(per_problem_stats) / 100)
        if actual_count > 0:
            ax2.text(x, count + 2, str(actual_count),
                    ha='center', va='bottom', fontsize=12)

    ax2.set_title('Distribution of Recovery Rates', fontsize=16, pad=20)
    ax2.set_xlabel('Recovery Rate (%)', fontsize=14)
    ax2.set_ylabel('Percentage of Problems (%)', fontsize=14)

    # Format axes to show percentages
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

    ax2.set_ylim(0, 100)
    ax2.set_xlim(-5, 105)  # Give some padding on both sides
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(pad=3.0)
    plt.show()