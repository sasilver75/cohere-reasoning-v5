from dotenv import load_dotenv
import os

from model_providers import OPENROUTER_MODEL_PROVIDERS, OpenRouterModel, OpenRouterProvider

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

MODEL = OpenRouterModel.DEEPSEEK_2_5_1210_INSTRUCT
# PROVIDER = OPENROUTER_MODEL_PROVIDERS[MODEL]
PROVIDER = OpenRouterProvider.DEEPSEEK

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


# Completion test examples
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
        "model": MODEL.value,  # Specify the model you want to use
        "messages": [
            {"role": "user", "content": user_turn},  # Using your existing prefix
            {"role": "assistant", "content": assistant_turn}  # Prefill example
        ],
        "provider": {
        "order": [
                PROVIDER.value
            ],
            "allow_fallbacks": False,
        },
        "temperature": 0.2,
        "top_p": 0.8
    }
    print("sending request")
    response = requests.post(url, headers=headers, json=data)
    print(f"{response.json()["choices"][0]["message"]["content"]} \n --- \n")
print(f"END OF COMPLETION EXAMPLES \n ~~~~~~~~~~~~~~~~~~ \n")   


# '[\n  {\n    "received": "SFCompute",\n    "code": "invalid_enum_value",\n    "options": [\n      "OpenAI",\n      "Anthropic",\n      "Google",\n      "Google AI Studio",\n      "Amazon Bedrock",\n      "Groq",\n      "SambaNova",\n      "Cohere",\n      "Mistral",\n      "Together",\n      "Together 2",\n      "Fireworks",\n      "DeepInfra",\n      "Lepton",\n      "Novita",\n      "Avian",\n      "Lambda",\n      "Azure",\n      "Modal",\n      "AnyScale",\n      "Replicate",\n      "Perplexity",\n      "Recursal",\n      "OctoAI",\n      "DeepSeek",\n      "Infermatic",\n      "AI21",\n      "Featherless",\n      "Inflection",\n      "xAI",\n      "Cloudflare",\n      "SF Compute",\n      "01.AI",\n      "HuggingFace",\n      "Mancer",\n      "Mancer 2",\n      "Hyperbolic",\n      "Hyperbolic 2",\n      "Lynn 2",\n      "Lynn",\n      "Reflection"\n    ],\n    "path": [\n      "provider",\n      "order",\n      0\n    ],\n    "message": "Invalid enum value. Expected \'OpenAI\' | \'Anthropic\' | \'Google...



# Let's try some math out

# data = [
#     (
#         "What's 5/4 + 2?",
#         "5/4 is 1, so"  # Wrong fraction conversion, unfinished
#     ),
#     (
#         "If a train travels 120 miles in 2 hours, what's its speed in miles per hour?",
#         "Let me divide 2 by 120, so"  # Wrong division order, unfinished
#     ),
#     (
#         "Sally has 3 times as many marbles as Tom. If Tom has 12 marbles, how many do they have together?",
#         "If Tom has 12 marbles, Sally has 3 marbles, so"  # Misinterpreted 'times as many', unfinished
#     ),
#     (
#         "What's 2 + 3 × 4?",
#         "First I'll add 2 and 3 to get 5, then"  # Order of operations mistake, unfinished
#     ),
#     (
#         "A rectangle has a width of 4 inches and a length twice its width. What's its area?",
#         "If the width is 4 inches, then the length is 4 + 2 = 6 inches. Now to find the area,"  # Wrong interpretation of 'twice'
#     ),
#     (
#         "If you have $20 and spend 25% of it, how much do you have left?",
#         "25% of $20 is $5, so I'll add $5 to get"  # Wrong operation (adding instead of subtracting)
#     ),
#     (
#         "What's the average of 15, 20, and 25?",
#         "To find the average, I'll add these numbers: 15 + 20 + 25 = 50. Now I'll divide by 2 since"  # Wrong divisor
#     ),
#     (
#         "If 8 cookies are shared equally among 4 children, how many cookies does each child get?",
#         "I'll multiply 8 × 4 to find out how many cookies each child gets, so"  # Wrong operation
#     ),
#     (
#         "What's 1/2 of 30?",
#         "To find half of 30, I'll add 30 + 2, which gives me"  # Wrong operation
#     ),
#     (
#         "A square has a perimeter of 20 inches. What's its area?",
#         "If the perimeter is 20 inches, each side must be 20/2 = 10 inches. Now for the area,"  # Wrong perimeter calculation
#     ),
#     (
#         "How many quarters make $2.75?",
#         "Each quarter is 25 cents, which is $0.25. So I'll multiply 2.75 × 0.25 to get"  # Wrong approach
#     ),
#     (
#         "If it takes 3 minutes to boil one egg, how long will it take to boil 6 eggs at the same time?",
#         "With 6 eggs, it will take 6 × 3 = 18 minutes because"  # Wrong reasoning about parallel vs sequential
#     )
# ]
# for user_turn, assistant_turn in data:
#     data = {
#     "model": MODEL.value,  # Specify the model you want to use
#     "messages": [
#         {"role": "user", "content": user_turn},  # Using your existing prefix
#         {"role": "assistant", "content": assistant_turn}  # Prefill example
#     ],
#     "provider": {
#         "order": [
#             PROVIDER.value
#         ],
#         "allow_fallbacks": False,
#         }
#     }
#     response = requests.post(url, headers=headers, json=data)
#     completion = response.json()["choices"][0]["message"]["content"]
#     print(f"""
#     Problem: {user_turn}
#     Prefix: {assistant_turn}
#     Completion: {completion}
#     \n----------------------------------\n
    # """)
