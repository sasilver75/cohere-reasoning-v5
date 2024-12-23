def get_perturb_prompt(problem: str, stub: str) -> str:
    return lightweight_perturb_prompt.format(problem=problem, stub=stub)

def get_solution_prompt(problem: str) -> str:
    return lightweight_solution_prompt.format(problem=problem)

def get_verification_prompt(problem: str, answer: str, candidate_solution: str) -> str:
    return lightweight_verify_prompt.format(problem=problem, answer=answer, candidate_solution=candidate_solution)

def get_solution_prompt_cohere(problem: str, stub: str) -> str:
    return lightweight_cohere_solution_prompt.format(problem=problem, stub=stub)



lightweight_solution_prompt =  """
Solve the following math or reasoning problem, clearly presenting your reasoning and final answer.

Your input is as follows:
<problem>
{problem}
</problem>
"""

# TODO: Am I using this correctly?
# I think we can pass in an empty string for the stub to do a straight-shot completion.
# It's different from the lightweight solution prompt because it needs to be a single message, not two.
lightweight_cohere_solution_prompt = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Solve the following math or reasoning problem, clearly presenting your reasoning and final answer.\n\nYour input is as follows:\n<problem>\n{problem}\n</problem><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{stub}"""

lightweight_perturb_prompt = """
You will be given a reasoning stub from a math word problem solution. Your task is to perturb this stub by introducing a single mathematical or reasoning error.

Your goal is to modify this stub in a way that will change the results of future reasoning and lead to an incorrect answer if followed. Follow these guidelines:

1. Make minimal changes to the stub. Do not add additional reasoning or try to solve for the right answer.
2. Focus on changing numbers, operators, or key words/phrases that will affect the result of the reasoning if it were played out.
3. Do not change the overall structure or length of the stub.
4. Do not apply any additional formatting (e.g. bolding, italicizing, etc.) your perturbation.

Good perturbations may include:
- Changing a number (e.g., from 40 hours to 45 hours)
- Switching an operator (e.g., from multiplication to division, or addition to subtraction)
- Altering a key phrase (e.g., from "60% of $5" to "60% more than $5")
- Reversing the order of operations (e.g., from "(2 + 3) * 4" to "2 + (3 * 4)")
- Changing units of measurement (e.g., from "meters" to "feet" without converting)
- Swapping comparison operators (e.g., from "greater than (>)" to "less than (<)")
- Modifying time relationships (e.g., from "per hour" to "per minute")
- Inverting fractions (e.g., from "3/4" to "4/3")
- Changing decimal places (e.g., from "1.5" to "15.0")

Bad perturbations include:
- Adding new steps or explanations
- Removing large portions of the stub
- Changing the reasoning stub entirely.


Here are some examples of good perturbations, along with explanations of what perturbation was applied:

<examples>

Original: "To solve this problem, let's break it down step by step.\n\n1. **First, calculate the cost of the first glass and the second glass.**\n   - The first glass costs $5.\n   - The second glass costs 60% of $5, which is $5 * 0.60 = $3.\n\n2. **Determine the pattern for the cost of the glasses.**\n   - Every odd-numbered glass (1st, 3rd, 5th"
Perturbed: "To solve this problem, let's break it down step by step.\n\n1. First, calculate the cost of the first glass and the second glass.\n - The first glass costs $5.\n - The second glass costs 60% more than $5, which is $5 1.60 = $8.\n\n2. Determine the pattern for the cost of the glasses.\n - Every odd-numbered glass (1st, 3rd, 5th"
Explanation: The perturbation changed "60% of $5" to "60% more than $5" and consequently updated the calculation from "$5 0.60 = $3" to "$5 1.60 = $8". It will lead to incorrect subsequent calculations.

Original: "To find out how many cups of feed Wendi needs to give her chickens in the final meal of the day, we first need to calculate the total amount of feed required for all the chickens for the entire day. Since each chicken requires 3 cups of feed per day, and there are 20 chickens, the total amount of feed needed per day is:\n\n20 chickens * 3 cups/chicken = 60 cups\n\nNow, let's calculate how much feed Wendi has already given to her"
Perturbed: "To find out how many cups of feed Wendi needs to give her chickens in the final meal of the day, we first need to calculate the total amount of feed required for all the chickens for the entire day. Since each chicken requires 2 cups of feed per day, and there are 20 chickens, the total amount of feed needed per day is:\n\n20 chickens * 2 cups/chicken = 40 cups\n\nNow, let's calculate how much feed Wendi has already given to her"
Explanation: The perturbation changed "3 cups of chicken" to "2 cups per chicken" and updated the calculation result accordingly ("60 cups" to "40 cups"). This will lead to an incorrect answer since the daily feed requirement is now underestimated.

Original: "To find the profit Josh made, we first need to calculate the new value of the house after the repairs, and then subtract the total amount he spent (the cost of the house plus the cost of repairs) from this new value.\n\n1. **Calculate the new value of the house:**\n   - Original value of the house = $80,000\n   - Increase in value due to repairs = 150% of $80,000\n   - Increase amount = 150% of $"
Perturbed: "To find the profit Josh made, we first need to calculate the new value of the house after the repairs, and then subtract the total amount he spent (the cost of the house plus the cost of repairs) from this new value.\n\n1. Calculate the new value of the house:\n - Original value of the house = $8,000\n - Increase in value due to repairs = 150% of $8,000\n - Increase amount = 150% of $"
Explanation: The perturbation removed a zero from the house value ($80,000 to $8,000), which will lead to an incorrect answer since the profit calculation is now underestimated.

Original: "To find Eliza's earnings for this week, we need to calculate her regular pay for the first 40 hours and her overtime pay for the additional 5 hours.\n\n1. **Calculate regular pay for the first 40 hours:**\n   - Regular hourly rate = $10\n   - Number of regular hours worked = 40 hours\n   - Regular pay = Regular hourly rate * Number of regular hours worked\n   - Regular pay = $10 * 40 = $400\n\n2."
Perturbed: "To find Eliza's earnings for this week, we need to calculate her regular pay for the first 40 hours and her overtime pay for the additional 5 hours.\n\n1. Calculate regular pay for the first 40 hours:\n - Regular hourly rate = $10\n - Number of regular hours worked = 40 hours\n - Regular pay = Regular hourly rate Number of regular hours worked\n - Regular pay = $10 40 = $500\n\n2."
Explanation: The perturbation changed the calculation result of "10 * 40 = $400" to "10 * 40 = $500". This will lead to an incorrect answer since the regular pay is now overestimated.

</examples>

Here is the reasoning stub that I want you to perturb:

<reasoning_stub>
{stub}
</reasoning_stub>

Apply a single perturbation to the reasoning stub.

Provide your perturbed version of the reasoning stub inside <perturbed_stub> tags. 
Ensure that your perturbation is impactful and that it maintains the original structure and length of the stub as closely as possible.
"""

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

