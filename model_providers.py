from models import OpenRouterProvider

"""
It's important that we select providers that provide the model in the same precision as which it was released (e.g. if a model was released in bf16, don't use an fp8 precision version of it).
OpenRouter makes this information available: https://openrouter.ai/meta-llama/llama-3.3-70b-instruct/providers
Additionally, we'd like to select a provider that offers the "raw completion" ability we're looking for (this doesn't matter for verifier models, but it can't hurt).
"""

OPENROUTER_MODEL_PROVIDERS = {
    "qwen/qwen-2.5-72b-instruct": OpenRouterProvider.DEEPINFRA,
    "meta-llama/llama-3.3-70b-instruct": OpenRouterProvider.NOVITA
}