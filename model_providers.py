from enum import Enum
"""
It's important that we select providers that provide the model in the same precision as which it was released (e.g. if a model was released in bf16, don't use an fp8 precision version of it).
OpenRouter makes this information available: https://openrouter.ai/meta-llama/llama-3.3-70b-instruct/providers
Additionally, we'd like to select a provider that offers the "raw completion" ability we're looking for (this doesn't matter for verifier models, but it can't hurt).
"""

class OpenRouterModel(Enum):
    QWEN_2_5_72B_INSTRUCT = "qwen/qwen-2.5-72b-instruct"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
    GEMMA_2_27B_INSTRUCT = "google/gemma-2-27b-it"

class OpenRouterProvider(Enum):
    DEEPINFRA = "DeepInfra"
    HYPERBOLIC = "Hyperbolic"
    NOVITA = "Novita"


OPENROUTER_MODEL_PROVIDERS = {
    OpenRouterModel.QWEN_2_5_72B_INSTRUCT: OpenRouterProvider.DEEPINFRA,
    OpenRouterModel.LLAMA_3_3_70B_INSTRUCT: OpenRouterProvider.NOVITA,
    OpenRouterModel.GEMMA_2_27B_INSTRUCT: OpenRouterProvider.DEEPINFRA
}