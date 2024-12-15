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


# API endpoint
url = "https://openrouter.ai/api/v1/chat/completions"

# Headers
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}



# Request body with assistant prefill
prefix = "Hey, I'm LLaMA 3.2 1B Instruct! I'm just a tiny model. My favorite food is"
data = {
    "model": "meta-llama/llama-3.2-1b-instruct",  # Specify the model you want to use
    "messages": [
        {"role": "user", "content": "Hey, what's your name?"},  # Using your existing prefix
        {"role": "assistant", "content": prefix}  # Prefill example
    ]
}


# Make the request
response = requests.post(url, headers=headers, json=data)
print(response.json(), "\n")

print(f"Prefix: {prefix}")
print("Completion: ", response.json()["choices"][0]["message"]["content"])


