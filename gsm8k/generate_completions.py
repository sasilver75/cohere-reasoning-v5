import asyncio
import time
import aiohttp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_providers import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider
import pandas as pd
from tqdm import tqdm
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging



# Configuration
N_PROBLEMS = 5  # None = All
MODELS = [
    OpenRouterModel.LLAMA_3_3_70B_INSTRUCT,
    OpenRouterModel.GEMMA_2_27B_INSTRUCT,
    OpenRouterModel.MISTRAL_NEMO_12B_INSTRUCT,
    OpenRouterModel.QWEN_2_5_72B_INSTRUCT,
    OpenRouterModel.QWEN_QWQ_32B_PREVIEW,
    OpenRouterModel.LLAMA_3_3_70B_INSTRUCT
]


load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")

logger = logging.getLogger(__name__)


lightweight_verify_prompt = """
You are given a math problem, its ground-truth solution, and a candidate solution to that problem, and your goal is to verify that the candidate solution is correct.

You will be given the following information:
- The problem
- The answer to the problem
- The candidate solution, which includes reasoning and a final answer

You will output a single word, "correct" or "incorrect", to indicate if the candidate solution is a valid solution to the problem.
You should not mark a candidate solution as incorrect because of any error in its reasoning. You should only care about whether the final answer produced matches the answer.

Your input is as follows:
<problem>
{problem}
</problem>
<answer>
{answer}
</answer>
<candidate_solution>
{candidate_solution}
</candidate_solution>

Now, evaluate the candidate solution by outputting either "correct" or "incorrect", considering the final answer produced.
Do not output any other text than "correct" or "incorrect". Do not output any form of reasoning or explanation. Only output "correct" or "incorrect", this is absolutely critical.
"""

lightweight_detect_prompt = """
You are given a math problem and a ground-truth solution to that problem.
You are given a "reasoning prefix," which represents an incomplete reasoning process containng an error, which the candidate solution was given.
A model under evaluation is given the prefix, and asked to complete the reasoning to solve the problem.
Your goal is to determine if the model under evaluation made a reasoning correction to the reasoning prefix to solve the problem.

These reasoning corrections MAY take the form of generations similar to "Wait, I made a mistake" or "However, this is a common misconception" or "However, I made an error in my reasoning" or "Oops!" or "However, there's a small error in the calculation."
There are other ways to express reasoning corrections. Use your best judgement to determine if the model under evaluation made a reasoning correction to the reasoning prefix to solve the problem.

You will be given the following information:
- The problem
- A prefix that indicates the reasoning that the model under evaluation was given, and asked to complete to solve the problem
- The completion of the prefix by the model under evaluation, which should solve the problem

You will output either "correction" or "no correction", depending on whether the model under evaluation made a reasoning correction to the reasoning prefix during its completion to solve the problem.

Below are some examples of inputs that show a reasoning correction. 
Observe that the correction changes the direction of the reasoning trajectory to solve the problem, and that you can still have a reasoning correction present even if the correct answer is not reached.
<correction_examples>

<correction_example_one>
<problem>
What's 5/4 + 2?
</problem>
<prefix>
5/4 is 1, so
</prefix>
<completion>
 the equation becomes 1 + 2, which equals 3. However, to be more precise, 5/4 is actually 1.25. So, 1.25 + 2 equals 3.25.
</completion>
</correction_example_one>

<correction_example_two>
<problem>
If a train travels 120 miles in 2 hours, what's its speed in miles per hour?
</problem>
<prefix>
Let me divide 2 by 120, so
</prefix>
<completion>
 the answer is 60 miles per hour. 

Correction: To find the speed of the train, you need to divide the distance traveled by the time it took. In this case, you divide 120 miles by 2 hours, which gives you 60 miles per hour. So, the train's speed is 60 miles per hour.
</completion>
</correction_example_two>

<correction_example_three>
<problem>
What's the average of 15, 20, and 25?
</problem>
<prefix>
To find the average, I'll add these numbers: 15 + 20 + 25 = 50. Now I'll divide by 2 since
</prefix>
<completion>
 there are 2 numbers. Oh, wait, I made a mistake. I should divide by 3 since there are 3 numbers. So, 60 / 3 = 20. The average is 20.
</completion>
</correction_example_three>

<correction_example_four>
<problem>
If 8 cookies are shared equally among 4 children, how many cookies does each child get?
</problem>
<prefix>
I'll multiply 8 * 4 to find out how many cookies each child gets, so
</prefix>
<completion>
 each child gets 32 cookies. 

However, that doesn't make sense because there are only 8 cookies to begin with. I made a mistake by multiplying instead of dividing. 

To find out how many cookies each child gets, I should divide 8 by 4. 

8 / 4 = 2

So each child gets 2 cookies.
</completion>
</correction_example_four>

<correction_example_five>
<problem>
In a class of 30 students, 60% play sports and 40% play music. If 5 students play both, how many students don't participate in either activity?
</problem>
<prefix>
60% of 30 is 18 students playing sports, 40% is 12 playing music. So 18 + 12 = 30 students total, meaning
</prefix>
<completion>
 all students are involved in at least one activity. However, since 5 students play both, we've counted them twice. So the actual number of students involved in at least one activity is 30 - 5 = 25. Therefore, 30 - 25 = 5 students don't participate in either activity.
</completion>
</correction_example_five>

<correction_example_six>
<problem>
Three friends split a pizza bill of $45. If Tom pays $5 more than Jack, and Jack pays $3 more than Mike, how much did each person pay?
</problem>
<prefix>
Let me start by dividing $45 by 3 to get each person's share, which is $15. Now,
</prefix>
<completion>
 let's use algebra to solve the problem. Let's say Mike pays x dollars. Then Jack pays x + 3 dollars, and Tom pays (x + 3) + 5 = x + 8 dollars. Since the total is $45, we can write the equation:
x + (x + 3) + (x + 8) = 45
Simplifying the equation, we get:
3x + 11 = 45
Subtracting 11 from both sides, we get:
3x = 34
Dividing both sides by 3, we get:
x = 34/3
However, this is not a whole number, which doesn't make sense in this context. I made a mistake in my initial assumption. Let me try again.
We know that the total is $45, and Tom pays $5 more than Jack, and Jack pays $3 more than Mike. Let's say Mike pays x dollars. Then Jack pays x + 3 dollars, and Tom pays (x + 3) + 5 = x + 8 dollars. We can write the equation:
x + (x + 3) + (x + 8) = 45
Simplifying the equation, we get:
3x + 11 = 45
Subtracting 11 from both sides, we get:
3x = 34
Dividing both sides by 3, we get:
x = 11
So, Mike pays $11, Jack pays $14 (11 + 3), and Tom pays $19 (11 + 8). 
Therefore, Mike pays $11, Jack pays $14, and Tom pays $19.
</completion>
</correction_example_six>


</correction_examples>

Here are some examples of inputs that do not show a reasoning correction:
<no_correction_examples>

<no_correction_example_one>
<problem>
A rectangle has a width of 4 inches and a length twice its width. What's its area?
</problem>
<prefix>
If the width is 4 inches, then the length is 4 + 2 = 6 inches. Now to find the area,
</prefix>
<completion>
 we multiply the length by the width, which is 4 * 6 = 24 square inches. 

Therefore, the area of the rectangle is 24 square inches.
</completion>
</no_correction_example_one>

<no_correction_example_two>
<problem>
A train leaves Station A at 2:00 PM traveling 60 mph. Another train leaves Station B at 2:30 PM traveling 75 mph in the opposite direction. If the stations are 300 miles apart, at what time do the trains meet?
</problem>
<prefix>
The trains are traveling at a combined speed of 60 + 75 = 135 mph. Since they need to cover 300 miles, I'll divide 300 by 135 to get
</prefix>
<completion>
 the time it takes for them to meet: 300 / 135 = 2.2222 hours, or 2 hours and 13.3333 minutes. Since the second train leaves at 2:30 PM, the trains will meet 2 hours and 13.3333 minutes after 2:30 PM, which is 4:43:20 PM.
</completion>
</no_correction_example_two>

<no_correction_example_three>
<problem>
What's 5/4 + 2?
</problem>
<prefix>
5/4 is 1, so
</prefix>
<completion>
 5/4 + 2 = 1 + 2 = **3**
</completion>
</no_correction_example_three>

<no_correction_example_four>
<problem>
A rectangle has a width of 4 inches and a length twice its width. What's its area?
</problem>
<prefix>
If the width is 4 inches, then the length is 4 + 2 = 6 inches. Now to find the area,
</prefix>
<completion>
 we multiply the length and width: 6 inches * 4 inches = 24 square inches.

**The area of the rectangle is 24 square inches.**
</completion>
</no_correction_example_four>

</no_correction_examples>


Here is the input for the problem you are evaluating:

<problem>
{problem}
</problem>
<prefix>
{prefix}
</prefix>
<completion>
{completion}
</completion>

Now, output either "correction" or "no correction", depending on whether the model under evaluation made a reasoning correction to the reasoning prefix during its completion to solve the problem.
Do not output any other text than "correction" or "no correction". Do not output any form of reasoning or explanation. Only output "correction" or "no correction", this is absolutely critical.

"""

lightweight_problem_prompt = """
Solve the following math or reasoning problem, clearly presenting your reasoning and final answer.

Your input is as follows:
<problem>
{problem}
</problem>
"""

def _get_verification_prompt(problem: str, answer: str, candidate_solution: str) -> str:
    return lightweight_verify_prompt.format(problem=problem, answer=answer, candidate_solution=candidate_solution)

def _get_detection_prompt(problem: str, prefix: str, completion: str) -> str:
    # TODO: The prompt uses prefix lingo, but the code uses perturbed_stub. We should change this to be consistent.
    return lightweight_detect_prompt.format(problem=problem, prefix=prefix, completion=completion)

def _get_problem_prompt(problem: str) -> str:
    return lightweight_problem_prompt.format(problem=problem)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def get_completion(session: aiohttp.ClientSession, model: OpenRouterModel, problem: str, perturbed_stub: str) -> str:
    """
    Get a completion to a prefix, from a model using async request
    """
    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": _get_problem_prompt(problem)},
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
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def verify_solution(session: aiohttp.ClientSession, problem: str, answer: int, candidate_solution: str) -> bool:
    """
    Async version of solution verification
    # TODO: Right now I'm assuming that this "answer" is enough to judge LM correctness. This might not be the case. Do I need to use the full "reasoning+answer" from the ground truth dataset to judge this?
    """
    model = OpenRouterModel.LLAMA_3_3_70B_INSTRUCT

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": _get_verification_prompt(problem, answer, candidate_solution)},
            ],
            "provider": {
                "order": [OPENROUTER_MODEL_PROVIDERS[model].value],
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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def detect_correction(session: aiohttp.ClientSession, problem: str, perturbed_stub: str, completion: str) -> bool:
    """
    Async version of correction detection
    """
    model = OpenRouterModel.LLAMA_3_3_70B_INSTRUCT

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": _get_detection_prompt(problem, perturbed_stub, completion)},
            ],
            "provider": {
                "order": [OPENROUTER_MODEL_PROVIDERS[model].value],
                "allow_fallbacks": False,
            },
            "temperature": 0,
            "top_k": 0
        },
        timeout=60
    ) as response:
        response_json = await response.json()
        response_content = response_json["choices"][0]["message"]["content"]

    correction_detected = response_content.lower() == 'correction'
    print(f"Detection response is {response_content}, which is {correction_detected}")
    return correction_detected


async def test_single_problem(session: aiohttp.ClientSession, model: OpenRouterModel, row: pd.Series) -> dict:
    """Handle a single problem evaluation"""
    row_id: int = row["row_id"]
    problem: str = row["problem"]
    answer: int = row["ground_truth_solution"] # eg "18"
    stub: str = row["stub"]
    perturbed_stub_lm: str = row["perturbed_stub_lm"]

    print(f"Testing model {model.value} on problem {row_id}")

    # These are robust, tenacity-decorated methods that call APIs that will retry if they raise Exceptions

    # Get the completion using the LM-perturbed stub
    perturbed_stub_lm_completion = await get_completion(session, model, problem, perturbed_stub_lm)
    perturbed_stub_lm_solution = f"{perturbed_stub_lm}{perturbed_stub_lm_completion}"
    
    # Get the verification and correction detection results
    perturbed_stub_lm_verified, perturbed_stub_correction_detected = await asyncio.gather(
            verify_solution(session, problem, answer, perturbed_stub_lm_solution), # Does the full solution match the answer?
            detect_correction(session, problem, perturbed_stub_lm, perturbed_stub_lm_completion) # Does the completion represent a correction from the stub, given the question?
        )
    
    return {
        "row_id": row_id,
        "problem": problem,
        "answer": answer,
        "stub": stub,
        "perturbed_stub_lm": perturbed_stub_lm,
        "perturbed_stub_lm_completion": perturbed_stub_lm_completion,
        "perturbed_stub_lm_solution": perturbed_stub_lm_solution,
        "perturbed_stub_lm_solution_verified": perturbed_stub_lm_verified,
        "perturbed_stub_lm_solution_correction_detected": perturbed_stub_correction_detected,
        #"perturbed_stub_deterministic": perturbed_stub_deterministic
        #...
    }

async def test_model(session: aiohttp.ClientSession, model: OpenRouterModel, df: pd.DataFrame) -> list[dict]:
    """Test all problems for a given model concurrently"""
    semaphore = asyncio.Semaphore(30)
    
    async def rate_limited_test(row: pd.Series):
        async with semaphore:
            return await test_single_problem(session, model, row)
    
    # Create all tasks
    tasks = [
        rate_limited_test(row)
        for _, row in df.iterrows()
    ]
    
    # TODO(SAM): Below could use some polish (use as_completed, etc.)

    # Run tasks concurrently and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any None results and handle exceptions
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with error: {str(result)}")
        elif result is not None:
            valid_results.append(result)
    
    return valid_results

async def async_main():

    print(f"Loading dataset...")
    df = pd.read_csv("gsm8k/datasets/gsm8k_stubs_and_perturbations.csv")
    print(f"Loaded dataset with {len(df)} rows and columns {list(df.columns)}")

    print(f"Testing {N_PROBLEMS if N_PROBLEMS is not None else "all"} problems")
    if N_PROBLEMS is not None:
        df = df.head(N_PROBLEMS)

    acc = []
    async with aiohttp.ClientSession() as session:
        for model in MODELS:
            try:
                print(f"\nStarting tests for model: {str(model.value)}")
                results = await test_model(session, model, df)
                acc.extend(results)
                print(f"Completed testing model: {str(model.value)}")
            except Exception as e:
                print(f"Error testing model {str(model.value)}: {str(e)}")
                continue

    print(f"Total results collected: {len(acc)}")
    print("Saving results...")
    df = pd.DataFrame(acc)
    df.to_csv("gsm8k/datasets/gsm8k_completions.csv", index=False)
    print("Results saved to gsm8k_completions.csv")

def main():
    print(f"Number of models to test: {len(OPENROUTER_MODEL_PROVIDERS)}")
    print(f"API key present: {'OPENROUTER_API_KEY' in os.environ}")

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()