from dotenv import load_dotenv
import os

from models import OpenRouterProvider

load_dotenv()
import requests

# NOTE: https://www.reddit.com/r/SillyTavernAI/comments/1fpyb1x/claude_35_sonnet_via_openrouter_on_text/


"""
OpenRouter is a unified API gateway that provides access to various AI models from different providers
(like Anthropic, Meta, Mistral, etc.) through a single, standardized API interface. 
It's similar to OpenAI's API but allows you to access many different models, often at better prices.
It also handles things like automatic fallbacks and routing to the best available provider.\

This is a sanity check to see if a model actually seems to be completing in the way we'd expect.
Frfom
"""

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(OPENROUTER_API_KEY)


# API endpoint
url = "https://openrouter.ai/api/v1/chat/completions"
# Headers
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

MODEL = "meta-llama/llama-3.3-70b-instruct"
PROVIDER = OpenRouterProvider.NOVITA


# Straight shot example
# print(f"STAIGHT SHOT EXAMPLE \n ~~~~~~~~~~~~~~~~~~ \n")
# prompt = '\nThe function $y=x^{2}+2ax+1$ is an increasing function in the interval $[2,+\\infty)$. Determine the range of values for the real number $a$.\n\nPlease solve this problem step-by-step, boxing the final answer.\n'
# data = {
#     "model": MODEL,  # Specify the model you want to use
#     "messages": [
#         {"role": "user", "content": prompt},  # Using your existing prefix
#         # {"role": "assistant", "content": prefix}  # Prefill example
#     ],
#     "provider": {
#         "order": [
#             PROVIDER.value
#         ],
#         "allow_fallbacks": False,
#     }
# }
# response = requests.post(url, headers=headers, json=data)
# print(response.json(), "\n")
# print(f"END OF STRAIGHT SHOT EXAMPLE \n ~~~~~~~~~~~~~~~~~~ \n")


# Completion examples
print(f"COMPLETION EXAMPLES \n ~~~~~~~~~~~~~~~~~~ \n")
prompts = [
    (
        "Hey, what's your favorite food?",
        "My favorite food is"  # Easy example. Finish the sentence.
    ),
    (
        "Return a JSON object with the keys name and age for me, Sam, age 29",
        "{\"name\": \"Sam\", \"age\": 2" # Will it add "9}"?
    ),
    (
        "Generate odd numbers from 1 to 20 (inclusive), comma-separated.",
        "1, 3, 5, 7, 9, 11, 13, 15, 17" # Will it add ", 19"?
    )
]
for user_turn, assistant_turn in prompts:
    data = {
    "model": MODEL,  # Specify the model you want to use
    "messages": [
        {"role": "user", "content": user_turn},  # Using your existing prefix
        {"role": "assistant", "content": assistant_turn}  # Prefill example
    ],
    "provider": {
        "order": [
            PROVIDER.value
        ],
        "allow_fallbacks": False,
        }
    }
    response = requests.post(url, headers=headers, json=data)
    print(response.json()["choices"][0]["message"]["content"])
print(f"END OF COMPLETION EXAMPLES \n ~~~~~~~~~~~~~~~~~~ \n")   

