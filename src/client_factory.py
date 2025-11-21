"""
Client factory for creating API clients for different LLM providers.

This module provides a clean way to instantiate the appropriate client
based on the model configuration.
"""

import os
from openai import AsyncAzureOpenAI, AsyncOpenAI
from typing import Union
from model_config import ClientType, ModelConfig


class ClientFactory:
    """Factory for creating API clients based on model configuration."""

    @staticmethod
    def create_client(
        config: ModelConfig,
        vllm: bool = False,
        port: int = 8080,
        openai_client_type: str = 'azure'
    ) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """
        Create an appropriate API client based on the model configuration.

        Args:
            config: Model configuration
            vllm: Whether to use vLLM (overrides config)
            port: Port for vLLM server
            openai_client_type: Type of OpenAI client ('azure' or 'openai')

        Returns:
            Configured async API client
        """
        # vLLM override
        if vllm:
            return AsyncOpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1"
            )

        # Create client based on model type
        openai_api_key = None
        openai_base_url = None

        if config.client_type == ClientType.VLLM:
            return AsyncOpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1"
            )

        elif config.client_type == ClientType.DEEPSEEK:
            openai_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not openai_api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            openai_base_url = "https://api.deepseek.com"

        elif config.client_type == ClientType.GEMINI:
            openai_api_key = os.getenv("GEMINI_KEY")
            if not openai_api_key:
                raise ValueError("GEMINI_KEY environment variable not set")
            openai_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        elif config.client_type == ClientType.OPENAI:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            openai_base_url = "https://api.openai.com/v1"

        elif config.client_type == ClientType.XAI:
            openai_api_key = os.getenv("XAI_API_KEY")
            if not openai_api_key:
                raise ValueError("XAI_API_KEY environment variable not set")
            openai_base_url = "https://api.x.ai/v1"

        elif config.client_type == ClientType.AZURE_OPENAI:
            # Allow override via parameter
            if openai_client_type == 'openai':
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                print("Using OpenAI client")
                openai_base_url = "https://api.openai.com/v1"
            else:
                endpoint = os.getenv("ENDPOINT_URL")
                subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
                if not endpoint or not subscription_key:
                    raise ValueError("ENDPOINT_URL or AZURE_OPENAI_API_KEY environment variable not set")
                print("Using Azure OpenAI client")
                return AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=subscription_key,
                    api_version="2025-03-01-preview"
                )

        else:
            raise ValueError(f"Unsupported client type: {config.client_type}")

        if openai_api_key and openai_base_url:
            return AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url
            )

        raise ValueError("Failed to configure OpenAI client parameters")
