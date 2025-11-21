"""
Model configuration registry for managing different LLM providers and their settings.

This module provides a clean, extensible way to configure different models without
hardcoding configurations throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ClientType(Enum):
    """Types of API clients supported."""
    VLLM = "vllm"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    XAI = "xai"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    full_name: str
    client_type: ClientType
    max_tokens: int = 8192
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None

    def get_api_params(self) -> Dict[str, Any]:
        """Get API parameters for this model."""
        params = {}

        # Handle reasoning models (o3, o4)
        if self.reasoning_effort:
            params['max_completion_tokens'] = self.max_tokens
            params['reasoning_effort'] = self.reasoning_effort
        else:
            params['max_tokens'] = self.max_tokens

        if self.temperature is not None:
            params['temperature'] = self.temperature
        if self.top_p is not None:
            params['top_p'] = self.top_p

        # Add any extra parameters
        params.update(self.extra_params)

        return params


class ModelRegistry:
    """Central registry for all model configurations."""

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._register_default_models()

    def _register_default_models(self):
        """Register all default model configurations."""

        # Qwen models
        self.register(ModelConfig(
            name="Qwen2.5-Coder-7B-Instruct",
            full_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            client_type=ClientType.VLLM
        ))

        self.register(ModelConfig(
            name="Qwen2.5-Coder-14B-Instruct",
            full_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            client_type=ClientType.VLLM
        ))

        self.register(ModelConfig(
            name="Qwen2.5-Coder-32B-Instruct",
            full_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            client_type=ClientType.VLLM
        ))

        self.register(ModelConfig(
            name="Qwen2.5-72B",
            full_name="Qwen/Qwen2.5-72B-Instruct",
            client_type=ClientType.VLLM
        ))

        # QwQ reasoning model
        self.register(ModelConfig(
            name="QwQ-32B",
            full_name="Qwen/QwQ-32B",
            client_type=ClientType.VLLM,
            max_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            extra_params={"top_k": 40, "presence_penalty": 2},
            system_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
        ))

        # Qwen3 models
        for size in ["4B", "8B", "14B", "32B"]:
            self.register(ModelConfig(
                name=f"Qwen3-{size}",
                full_name=f"Qwen/Qwen3-{size}",
                client_type=ClientType.VLLM,
                max_tokens=38912
            ))

            # Non-thinking variants
            self.register(ModelConfig(
                name=f"Qwen3-{size}-Non-Thinking",
                full_name=f"Qwen/Qwen3-{size}",
                client_type=ClientType.VLLM,
                max_tokens=38912
            ))

        self.register(ModelConfig(
            name="Qwen3-30B",
            full_name="Qwen/Qwen3-30B-A3B",
            client_type=ClientType.VLLM,
            max_tokens=38912
        ))

        self.register(ModelConfig(
            name="Qwen3-30B-Non-Thinking",
            full_name="Qwen/Qwen3-30B-A3B",
            client_type=ClientType.VLLM,
            max_tokens=38912
        ))

        # Llama models
        self.register(ModelConfig(
            name="Llama-3.3-70B-Instruct",
            full_name="meta-llama/Llama-3.3-70B-Instruct",
            client_type=ClientType.VLLM
        ))

        self.register(ModelConfig(
            name="Llama-3.1-8B-Instruct",
            full_name="meta-llama/Llama-3.1-8B-Instruct",
            client_type=ClientType.VLLM
        ))

        # DeepSeek models
        self.register(ModelConfig(
            name="DeepSeek-Coder-V2-Lite-Instruct",
            full_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            client_type=ClientType.VLLM
        ))

        # DeepSeek R1 models
        for variant in ["Llama-8B", "Llama-70B", "Qwen-32B", "Qwen-14B", "Qwen-7B", "Qwen-1.5B"]:
            self.register(ModelConfig(
                name=f"DeepSeek-R1-Distill-{variant}",
                full_name=f"deepseek-ai/DeepSeek-R1-Distill-{variant}",
                client_type=ClientType.VLLM,
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                system_prompt="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            ))

        # DeepSeek API models
        self.register(ModelConfig(
            name="deepseek-reasoner",
            full_name="deepseek-reasoner",
            client_type=ClientType.DEEPSEEK
        ))

        self.register(ModelConfig(
            name="deepseek-chat",
            full_name="deepseek-chat",
            client_type=ClientType.DEEPSEEK
        ))

        # Gemini models
        self.register(ModelConfig(
            name="gemini-2.5-flash",
            full_name="gemini-2.5-flash-preview-04-17",
            client_type=ClientType.GEMINI,
            max_tokens=65536,
            temperature=1
        ))

        self.register(ModelConfig(
            name="gemini-2.5-pro",
            full_name="gemini-2.5-pro-exp-03-25",
            client_type=ClientType.GEMINI,
            max_tokens=65536
        ))
        
        # Grok Model
        self.register(ModelConfig(
            name="grok-4-fast-reasoning",
            full_name="grok-4-fast-reasoning",
            client_type=ClientType.XAI,
            max_tokens=100000
        ))

        self.register(ModelConfig(
            name="gemini-2.0-flash",
            full_name="gemini-2.0-flash",
            client_type=ClientType.GEMINI
        ))

        self.register(ModelConfig(
            name="gemini-2.0-flash-lite",
            full_name="gemini-2.0-flash-lite",
            client_type=ClientType.GEMINI
        ))

        # Mistral models
        self.register(ModelConfig(
            name="Codestral-22B-v0.1",
            full_name="mistralai/Codestral-22B-v0.1",
            client_type=ClientType.VLLM
        ))

        self.register(ModelConfig(
            name="Mistral-Small-3.1-24B-2503",
            full_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            client_type=ClientType.VLLM
        ))

        self.register(ModelConfig(
            name="Mistral-Large-Instruct-2411",
            full_name="mistralai/Mistral-Large-Instruct-2411",
            client_type=ClientType.VLLM
        ))

        # OpenAI GPT models
        self.register(ModelConfig(
            name="gpt-4.1",
            full_name="gpt-4.1",
            client_type=ClientType.AZURE_OPENAI,  # Default to Azure, can be overridden
            max_tokens=32768
        ))

        # O-series reasoning models
        for effort in ["low", "medium", "high"]:
            self.register(ModelConfig(
                name=f"gpt-o3-mini-{effort}",
                full_name="o3-mini-2025-01-31",
                client_type=ClientType.AZURE_OPENAI,
                max_tokens=100000,
                reasoning_effort=effort
            ))

            self.register(ModelConfig(
                name=f"gpt-o4-mini-{effort}",
                full_name="o4-mini",
                client_type=ClientType.AZURE_OPENAI,
                max_tokens=100000,
                reasoning_effort=effort
            ))

        # Default o3/o4 without effort specified
        self.register(ModelConfig(
            name="gpt-o3-mini",
            full_name="o3-mini-2025-01-31",
            client_type=ClientType.AZURE_OPENAI,
            max_tokens=100000,
            reasoning_effort="medium"
        ))

        self.register(ModelConfig(
            name="gpt-o4-mini",
            full_name="o4-mini",
            client_type=ClientType.AZURE_OPENAI,
            max_tokens=100000,
            reasoning_effort="medium"
        ))

        # GPT OSS model
        for effort in ["low", "medium", "high"]:
            self.register(ModelConfig(
                name=f"gpt-oss-20b-{effort}",
                full_name="openai/gpt-oss-20b",
                client_type=ClientType.VLLM,
                max_tokens=131072,
                reasoning_effort=effort
            ))

        self.register(ModelConfig(
            name="gpt-oss-20b",
            full_name="openai/gpt-oss-20b",
            client_type=ClientType.VLLM,
            max_tokens=131072,
            reasoning_effort="medium"
        ))

    def register(self, config: ModelConfig):
        """Register a new model configuration."""
        self._models[config.name] = config

    def get(self, model_name: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_name in self._models:
            return self._models[model_name]

        # Fallback: create a default config for unknown models
        # Try to infer client type from model name
        if "gemini" in model_name.lower():
            client_type = ClientType.GEMINI
        elif "deepseek" in model_name.lower():
            client_type = ClientType.DEEPSEEK
        elif "gpt" in model_name.lower():
            client_type = ClientType.AZURE_OPENAI
        else:
            client_type = ClientType.VLLM

        return ModelConfig(
            name=model_name,
            full_name=model_name,
            client_type=client_type
        )

    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._models.keys())


# Global registry instance
_registry = ModelRegistry()


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration from the global registry."""
    return _registry.get(model_name)


def register_model(config: ModelConfig):
    """Register a new model in the global registry."""
    _registry.register(config)


def list_available_models() -> list:
    """List all available model names."""
    return _registry.list_models()
