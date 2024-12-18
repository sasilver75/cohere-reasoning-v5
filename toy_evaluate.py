from model_providers import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider
import pandas as pd
from tqdm import tqdm
from toy_data import TOY_PROBLEMS
import requests
from dotenv import load_dotenv
import os

load_dotenv()
if not "OPENROUTER_API_KEY" in os.environ:
    raise ValueError("OPENROUTER_API_KEY must be set in the environment")


lightweight_verify_prompt = """
You are given a math problem, its ground-truth solution, and a candidate solution to that problem, and your goal is to verify that the candidate solution is correct.

You will be given the following information:
- The problem
- The ground-truth solution to the problem
- The candidate solution

You will output a single word, "correct" or "incorrect", to indicate if the candidate solution is a valid solution to the problem.
You should not mark a candidate solution as incorrect because its reasoning does not match the solution. You should only care about the final answer produced.

Your input is as follows:
<problem>
{problem}
</problem>
<ground_truth_solution>
{solution}
</ground_truth_solution>
<candidate_solution>
{candidate_solution}
</candidate_solution>

Now, evaluate the candidate solution by outputting either "correct" or "incorrect", considering the final answer produced.
Do not output any other text than "correct" or "incorrect". Do not output any form of reasoning or explanation. Only output "correct" or "incorrect", this is absolutely critical.
"""

lightweight_detect_prompt = """
You are a math tutor. You are given a math problem and a ground-truth solution to that problem.
You are given a "reasoning prefix," which represents an incomplete reasoning process containng an error, which the candidate solution was given.
A model under evaluation is given the prefix, and asked to complete the reasoning to solve the problem.
Your goal is to determine if the model under evaluation made a reasoning correction to the reasoning prefix to solve the problem.

These reasoning corrections MAY take the form of generations similar to "Wait, I made a mistake" or "However, this is a common misconception" or "However, I made an error in my reasoning" or "Oops!" or "However, there's a small error in the calculation."
There are other ways to express reasoning corrections. Use your best judgement to determine if the model under evaluation made a reasoning correction to the reasoning prefix to solve the problem.

You will be given the following information:
- The problem
- The solution to the problem
- A prefix that indicates the reasoning that the model under evaluation was given, and asked to complete to solve the problem
- The completion of the prefix by the model under evaluation, which should solve the problem

You will output either "correction" or "no correction", depending on whether the model under evaluation made a reasoning correction to the reasoning prefix during its completion to solve the problem.
Do not output any other text than "correction" or "no correction". Do not output any form of reasoning or explanation. Only output "correction" or "no correction", this is absolutely critical.

<problem>
{problem}
</problem>
<ground_truth_solution>
{solution}
</ground_truth_solution>
<prefix>
{prefix}
</prefix>
<completion>
{completion}
</completion>
"""

def _get_verification_prompt(problem: str, solution: str, candidate_solution: str) -> str:
    return lightweight_verify_prompt.format(problem=problem, solution=solution, candidate_solution=candidate_solution)

def _get_detection_prompt(problem: str, solution: str, prefix: str, completion: str) -> str:
    return lightweight_detect_prompt.format(problem=problem, solution=solution, prefix=prefix, completion=completion)



def get_completion(model: OpenRouterModel, provider: OpenRouterProvider, problem: str, prefix: str) -> str:
    """
    Get a completion to a prefix, from a model
    """
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model.value,
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": prefix}  # Give the prefix as the assitant message; will 
            ],
            "provider": {
                "order": [provider.value],
                "allow_fallbacks": False,
            },
            "temperature": 0.2,
            "top_p": 0.8
        },
    )
    return response.json()["choices"][0]["message"]["content"]

def verify_solution(problem: str, solution: str, candidate_solution: str) -> bool:
    """
    Verifier and detecter use a hardcoded verifier (L3.3 70B)
    Candidate solution is the entire candidate solution, including both prefix and completion
    """
    # Hardcoded verifier
    model_name = "meta-llama/llama-3.3-70b-instruct"
    provider_name = "Novita"

    # Get the verification
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "user", "content": _get_verification_prompt(problem, solution, candidate_solution)},
            ],
            "provider": {
                "order": [provider_name],
                "allow_fallbacks": False,
            },
            "temperature": 0,
            "top_k": 0
        },
    )
    response_content = response.json()["choices"][0]["message"]["content"]

    # Extract the result
    verified = response_content.lower() == 'correct'
    print(f"Verification response is {response_content}, which is {verified}")
    return verified

def detect_correction(problem: str, solution: str, prefix: str, completion: str) -> bool:
    """
    Verifier and detecter use a hardcoded verifier (L3.3 70B)
    """
    # Hardcoded verifier
    model_name = "meta-llama/llama-3.3-70b-instruct"
    provider_name = "Novita"

    # Get the detection
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "user", "content": _get_detection_prompt(problem, solution, prefix, completion)},
            ],
            "provider": {
                "order": [provider_name],
                "allow_fallbacks": False,
            },
            "temperature": 0,
            "top_k": 0
        },
    )
    response_content = response.json()["choices"][0]["message"]["content"]

    # Extract the result
    correction_detected = response_content.lower() == 'correction'
    print(f"Detection response is {response_content}, which is {correction_detected}")
    return correction_detected



def test_model(model: OpenRouterModel, provider: OpenRouterProvider) -> list[dict]:
    acc = []
    for idx, (problem, prefix, solution) in tqdm(enumerate(TOY_PROBLEMS), desc=f"Testing problems for model {model.value}", total=len(TOY_PROBLEMS)):
        print(f"Testing model {model.value} on problem {idx}")
        completion = get_completion(model, provider, problem, prefix)

        candidate_solution = f"{prefix} {completion}"
        verified = verify_solution(problem, solution, candidate_solution)
        correction_detected = detect_correction(problem, solution, prefix, completion)

        acc.append({
            "model": model.value,
            "provider": provider.value,
            "problem_id": idx,
            "problem": problem,
            "solution": solution,
            "prefix": prefix,
            "completion": completion,
            "candidate_solution": candidate_solution,
            "verified": verified,
            "correction_detected": correction_detected
        })

    return acc

def main(): 
    acc = []
    for model, provider in tqdm(OPENROUTER_MODEL_PROVIDERS.items(), desc="Testing models"):
        acc.extend(test_model(model, provider))

    df = pd.DataFrame(acc)  # Change this to include column names
    df.to_csv("toy_evaluate.csv", index=False)


if __name__ == "__main__":
    main()

