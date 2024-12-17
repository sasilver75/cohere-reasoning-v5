STRAIGHT_SHOT_SOLUTION_PROMPT = """
{problem}

Please solve this problem step-by-step, boxing the final answer.
"""

VERIFY_SOLUTION_PROMPT = """
Here is a math problem and its ground-truth solution:
<problem>
{problem}
</problem>
<solution>
{solution}
</solution>

Here is a candidate solution that may or may not be correct:
<candidate_solution>
{candidate_solution}
</candidate_solution>

The candidate solution may have boxed (e.g. using the \\boxed{{...}} command) the answers to both explicitly stated subproblems (if they exist) and the final answer.

Given the above information, reason about whether the candidate solution is correct, where correctness is defined as producing a correct final answer.

First, reason about whether the solution is correct in <verification_reasoning></verification_reasoning> XML tags.
    - To do this, first state the final answer of the ground truth solution detailed in <solution> tags above.
    - Then, state the final answer of the candidate solution detailed in the <candidate_solution> tags above.
    - Finally, reason about whether the candidate solution is correct, specifically indicating the step and manner in which the reasoning may have gone wrong, if it did.
    - If the correct answer was produced in the candidate solution but not appropriately boxed (for example, maybe the answer was boxed instead of the related multiple choice option, or vice-versa) -- that should still be considered as a Correct solution.
Make sure to remember to close your <verification_reasoning> tag with a </verification_reasoning> tag.

Then, determine whether the candidate solution is either "Correct" or "Incorrect" in <verification_result></verification_result> XML tags, given your reasoning.
In terms of structure, a good verification result might look like:
<verification_result>
Incorrect
</verification_result>
or
<verification_result>
Correct
</verification_result>
Make sure to remember to close your <verification_result> tag with a </verification_result> tag.
"""

COMPLETION_PROMPT_USER = """
{problem}

Please solve this problem step-by-step, boxing the final answer.
"""

# Do we even want this preamble?
COMPLETION_PROMPT_ASSISTANT = """
Certainly. Here is the step-by-step reasoning and final answer to the problem:

{prefix}
"""

COMPLETION_TEMPLATE = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_turn}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{assistant_turn}"""
