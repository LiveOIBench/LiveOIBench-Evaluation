"""
LLM API interface for generating solutions.

This module provides a clean interface for calling various LLM APIs
with proper rate limiting, error handling, and response management.
"""

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI
import asyncio
from typing import List, Dict, Union, Tuple, Optional, Any
import tqdm.asyncio
import backoff
import json
import os

from model_config import get_model_config, ModelConfig
from client_factory import ClientFactory
from code_extractor import CodeExtractor


async def generate_from_chat_completion(
    messages_list: List[List[Dict[str, str]]],
    model: str,
    max_tokens: int = 8192,
    verbose: bool = False,
    vllm: bool = False,
    port: int = 8080,
    requests_per_minute: int = 60,
    save_info: Optional[List[Tuple[str, int, str]]] = None,
    sequential: bool = False,
    openai_client: str = 'azure',
    **kwargs,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Generate completions from chat messages using specified model.

    Args:
        messages_list: List of message lists for each request
        model: Model name to use
        max_tokens: Maximum tokens to generate (can be overridden by model config)
        verbose: Whether to show progress bars
        vllm: Whether to use vLLM endpoint
        port: Port for vLLM server
        requests_per_minute: Rate limit for API calls
        save_info: Optional list of (task, seed, prediction_path) tuples for saving
        sequential: Whether to process requests sequentially
        openai_client: Type of OpenAI client ('azure' or 'openai')

    Returns:
        Tuple of (responses, token_usage_dict)
    """
    # Get model configuration
    config = get_model_config(model)

    # Create appropriate client
    client = ClientFactory.create_client(
        config=config,
        vllm=vllm,
        port=port,
        openai_client_type=openai_client
    )

    # Get the full model name for API calls
    full_model = config.full_name

    # Calculate delay for rate limiting
    delay_seconds = 60.0 / requests_per_minute if requests_per_minute > 0 else 0

    # Initialize token usage tracking
    token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'successful_requests': 0,
        'failed_requests': 0
    }

    # Process messages
    responses = []
    n = len(messages_list)

    if sequential:
        # Sequential processing
        responses = await _process_sequential(
            messages_list, client, full_model, config, save_info,
            model, token_usage, verbose
        )
    else:
        # Parallel processing with rate limiting
        responses = await _process_parallel(
            messages_list, client, full_model, config, save_info,
            model, token_usage, delay_seconds, verbose
        )

    # Print token usage summary
    _print_token_usage(token_usage)

    return responses, token_usage


async def _process_sequential(
    messages_list: List[List[Dict[str, str]]],
    client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    full_model: str,
    config: ModelConfig,
    save_info: Optional[List[Tuple[str, int, str]]],
    model: str,
    token_usage: Dict[str, int],
    verbose: bool
) -> List[Any]:
    """Process requests sequentially."""
    responses = []
    tqdm_iter = tqdm.tqdm(enumerate(messages_list), total=len(messages_list), disable=not verbose)

    for i, message in tqdm_iter:
        response = await _process_single_message(
            message, i, client, full_model, config, save_info, model, token_usage
        )
        responses.append(response)

    return responses


async def _process_parallel(
    messages_list: List[List[Dict[str, str]]],
    client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    full_model: str,
    config: ModelConfig,
    save_info: Optional[List[Tuple[str, int, str]]],
    model: str,
    token_usage: Dict[str, int],
    delay_seconds: float,
    verbose: bool
) -> List[Any]:
    """Process requests in parallel with rate limiting."""
    tasks = []
    n = len(messages_list)

    # Create tasks with rate limiting delays
    for i, message in enumerate(tqdm.tqdm(messages_list, total=n, disable=not verbose)):
        # Add delay for rate limiting (except first request)
        if i > 0 and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        # Create task
        task = asyncio.create_task(
            _process_single_message(
                message, i, client, full_model, config, save_info, model, token_usage
            )
        )
        tasks.append(task)

    # Collect results in original order
    responses = [None] * n
    pbar = tqdm.tqdm(total=n, disable=not verbose)

    for done in asyncio.as_completed(tasks):
        idx, resp = await done
        responses[idx] = resp
        pbar.update(1)

    pbar.close()
    return responses


async def _process_single_message(
    message: List[Dict[str, str]],
    idx: int,
    client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    full_model: str,
    config: ModelConfig,
    save_info: Optional[List[Tuple[str, int, str]]],
    model: str,
    token_usage: Dict[str, int]
) -> Tuple[int, Any]:
    """Process a single message and save results."""
    try:
        # Generate response
        response = await generate_answer(message, client, full_model, config)

        # Update token counts
        _update_token_counts(response, token_usage)

        # Save raw response and extract code if save_info provided
        if save_info and idx < len(save_info):
            task, seed, prediction_path = save_info[idx]
            CodeExtractor.save_raw_response(response, prediction_path, task, model, seed)

            # Extract and save code
            content = CodeExtractor.extract_from_response(response)
            if content:
                code, status = CodeExtractor.extract_code(content, task)
                CodeExtractor.save_code(code, prediction_path, task, model, seed)

        return idx, response

    except Exception as e:
        print(f"Error processing message {idx}: {str(e)}")
        token_usage['failed_requests'] += 1
        return idx, None


def _update_token_counts(response: Any, token_usage: Dict[str, int]):
    """Update token usage statistics."""
    if response and hasattr(response, "usage"):
        token_usage['prompt_tokens'] += response.usage.prompt_tokens
        token_usage['completion_tokens'] += response.usage.completion_tokens
        token_usage['total_tokens'] += response.usage.total_tokens
        token_usage['successful_requests'] += 1
    else:
        token_usage['failed_requests'] += 1


def _print_token_usage(token_usage: Dict[str, int]):
    """Print token usage statistics."""
    print("\n=== Token Usage Statistics ===")
    print(f"Total prompt tokens: {token_usage['prompt_tokens']}")
    print(f"Total completion tokens: {token_usage['completion_tokens']}")
    print(f"Total tokens: {token_usage['total_tokens']}")
    print(f"Successful requests: {token_usage['successful_requests']}")
    print(f"Failed requests: {token_usage['failed_requests']}")
    print("============================\n")


@backoff.on_exception(
    backoff.expo,
    openai.RateLimitError,
    max_tries=10,
    factor=2,
    max_value=60
)
async def generate_answer(
    prompt: List[Dict[str, str]],
    client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    model: str,
    config: ModelConfig
) -> Any:
    """
    Generate a single answer from the API.

    Args:
        prompt: Message list for this request
        client: API client to use
        model: Full model name for API call
        config: Model configuration

    Returns:
        API response object
    """
    # Get API parameters from config
    api_params = config.get_api_params()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            timeout=72000,
            **api_params
        )
        return response

    except Exception as e:
        # Return an empty response structure on error
        # This will be tracked in token_usage
        raise e


# Backward compatibility exports
__all__ = [
    'generate_from_chat_completion',
    'generate_answer',
]
