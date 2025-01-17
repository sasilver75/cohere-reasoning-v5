from enum import Enum

"""
It's important that we select providers that provide the model in the same precision as which it was released (e.g. if a model was released in bf16, don't use an fp8 precision version of it).
OpenRouter makes this information available: https://openrouter.ai/meta-llama/llama-3.3-70b-instruct/providers
Additionally, we'd like to select a provider that offers the "raw completion" ability we're looking for (this doesn't matter for verifier models, but it can't hurt).
"""

class OpenRouterModel(Enum):
    QWEN_2_5_72B_INSTRUCT = "qwen/qwen-2.5-72b-instruct"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
    LLAMA_3_1_405B_INSTRUCT = "meta-llama/llama-3.1-405b-instruct"
    GEMMA_2_27B_INSTRUCT = "google/gemma-2-27b-it"
    DEEPSEEK_2_5_1210_INSTRUCT = "deepseek/deepseek-chat-v2.5"
    MISTRAL_NEMO_12B_INSTRUCT = "mistralai/mistral-nemo"
    NEMOTRON_4_340B_INSTRUCT = "nvidia/nemotron-4-340b-instruct"
    MISTRAL_8x22B_INSTRUCT = "mistralai/mixtral-8x22b-instruct"
    PHI_3_128K_MEDIUM_INSTRUCT = "microsoft/phi-3-medium-128k-instruct"
    QWEN_QWQ_32B_PREVIEW = "qwen/qwq-32b-preview"
    PHI_3_5_MINI_128K_INSTRUCT = "microsoft/phi-3.5-mini-128k-instruct"
    DEEPSEEK_3 = "deepseek/deepseek-chat"
    


class OpenRouterProvider(Enum):
    DEEPINFRA = "DeepInfra"
    HYPERBOLIC = "Hyperbolic"
    NOVITA = "Novita"
    AVIAN = "Avian"
    SF_COMPUTE = "SF Compute"  # BAD! Fuck these guys.
    DEEPSEEK = "DeepSeek"
    MISTRAL = "Mistral"
    AZURE = "Azure"
    FIREWORKS = "Fireworks"
    TOGETHER = "Together"


OPENROUTER_MODEL_PROVIDERS = {
    # Using Deepseek for prefix/perturbation generation; Unfortuantely Hyperbolic (bf16) has very low rate limits, so I'm going to try to use Deepseek (fp8); I think it will still be wsmart enough.
    OpenRouterModel.DEEPSEEK_2_5_1210_INSTRUCT: OpenRouterProvider.HYPERBOLIC, # Don't have a good provider for this one. DeepSeek ddoesn't do completions. Hyperbolic hangs way too often. But I can use it for prefix/perturbation generation, which doesn't need that. OKAY had to swap to hyperbolic because deepseek stealth-changed...
    OpenRouterModel.DEEPSEEK_3: OpenRouterProvider.DEEPSEEK,
    OpenRouterModel.QWEN_2_5_72B_INSTRUCT: OpenRouterProvider.DEEPINFRA,  
    OpenRouterModel.LLAMA_3_3_70B_INSTRUCT: OpenRouterProvider.NOVITA,
    OpenRouterModel.LLAMA_3_1_405B_INSTRUCT: OpenRouterProvider.DEEPINFRA, # Used only for perturb/verify
    OpenRouterModel.GEMMA_2_27B_INSTRUCT: OpenRouterProvider.DEEPINFRA,
    OpenRouterModel.MISTRAL_NEMO_12B_INSTRUCT: OpenRouterProvider.DEEPINFRA,  # Check: Shuld this be SF compute? Or should it be DeepInfra?
    # OpenRouterModel.MISTRAL_8x22B_INSTRUCT: OpenRouterProvider.MISTRAL,  # No providers list precision; Mistral doesn't do completions.
    # OpenRouterModel.PHI_3_128K_MEDIUM_INSTRUCT: OpenRouterProvider.AZURE,  # Only provider is Azure and it just returns EOS tokens when doing completions
    OpenRouterModel.QWEN_QWQ_32B_PREVIEW: OpenRouterProvider.DEEPINFRA,  # Check
    # OpenRouterModel.PHI_3_5_MINI_128K_INSTRUCT: OpenRouterProvider.AZURE,  # Phi doesn't do completions via Azure.
}



class CohereModel(Enum):
    COHERE_R7B = "command-r7b-12-2024"
    COHERE_CRP = "command-r-plus-08-2024"
