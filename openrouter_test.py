from dotenv import load_dotenv
import os

load_dotenv()
import requests

# NOTE: https://www.reddit.com/r/SillyTavernAI/comments/1fpyb1x/claude_35_sonnet_via_openrouter_on_text/


"""
OpenRouter is a unified API gateway that provides access to various AI models from different providers
(like Anthropic, Meta, Mistral, etc.) through a single, standardized API interface. 
It's similar to OpenAI's API but allows you to access many different models, often at better prices.
It also handles things like automatic fallbacks and routing to the best available provider.
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



# Request body with assistant prefill
# See provider routing https://openrouter.ai/docs/provider-routing, ordering and disabling fallbacks
# prefix = "Hey, I'm Qwen 2.5 72B Instruct! I'm a model from the Qwen team at Alibaba Cloud Research. My favorite food is"
# data = {
#     "model": "qwen/qwen-2.5-72b-instruct",  # Specify the model you want to use
#     "messages": [
#         {"role": "user", "content": "Hey, what's your name?"},  # Using your existing prefix
#         {"role": "assistant", "content": prefix}  # Prefill example
#     ],
#     "provider": {
#         "order": [
#             "DeepInfra"  # Not sure where the list of these are
#         ],
#         "allow_fallbacks": False,
#     }
# }


# Make the request
# response = requests.post(url, headers=headers, json=data)
# print(response.json(), "\n")
# print(f"Prefix: {prefix}")
# print("Completion: ", response.json()["choices"][0]["message"]["content"])


# response = requests.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
# print(response.json())
# print(response.json()["data"]["label"])




# '{"error":{"message":"[\\n  {\\n    \\"received\\": \\"DeenInfra\\",\\n    \\"code\\": \\"invalid_enum_value\\",\\n    \\"options\\": [\\n      \\"OpenAI\\",\\n      \\"Anthropic\\",\\n      \\"Google\\",\\n      \\"Google AI Studio\\",\\n      \\"Amazon Bedrock\\",\\n      \\"Groq\\",\\n      \\"SambaNova\\",\\n      \\"Cohere\\",\\n      \\"Mistral\\",\\n      \\"Together\\",\\n      \\"Together 2\\",\\n      \\"Fireworks\\",\\n      \\"DeepInfra\\",\\n      \\"Lepton\\",\\n      \\"Novita\\",\\n      \\"Avian\\",\\n      \\"Lambda\\",\\n      \\"Azure\\",\\n      \\"Modal\\",\\n      \\"AnyScale\\",\\n      \\"Replicate\\",\\n      \\"Perplexity\\",\\n      \\"Recursal\\",\\n      \\"OctoAI\\",\\n      \\"DeepSeek\\",\\n      \\"Infermatic\\",\\n      \\"AI21\\",\\n      \\"Featherless\\",\\n      \\"Inflection\\",\\n      \\"xAI\\",\\n      \\"Cloudflare\\",\\n      \\"01.AI\\",\\n      \\"HuggingFace\\",\\n      \\"Mancer\\",\\n      \\"Mancer 2\\",\\n      \\"Hyperbolic\\",\\n      \\"Hyperbolic 2\\",\\...

prompt = '\nThe function $y=x^{2}+2ax+1$ is an increasing function in the interval $[2,+\\infty)$. Determine the range of values for the real number $a$.\n\nPlease solve this problem step-by-step, boxing the final answer.\n'
data = {
    "model": "qwen/qwen-2.5-72b-instruct",  # Specify the model you want to use
    "messages": [
        {"role": "user", "content": prompt},  # Using your existing prefix
        # {"role": "assistant", "content": prefix}  # Prefill example
    ],
    "provider": {
        "order": [
            "DeepInfra"  # Not sure where the list of these are
        ],
        "allow_fallbacks": False,
    }
}
response = requests.post(url, headers=headers, json=data)
print(response.json(), "\n")


