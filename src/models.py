import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI
import asyncio
from typing import List, Dict, Union, Tuple
import tqdm.asyncio  # Import the module
import backoff
import json
import os
import time

endpoint = os.getenv("ENDPOINT_URL") 
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
project_id = os.getenv("PROJECT_ID")
model_mapping = {
    "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-Coder-14B-Instruct": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "DeepSeek-Coder-V2-Lite-Instruct": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "QwQ-32B": "Qwen/QwQ-32B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek-Coder-V2-Lite-Instruct": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "QwQ-32B": "Qwen/QwQ-32B",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro": "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    "Codestral-22B-v0.1": "mistralai/Codestral-22B-v0.1",
    "Mistral-Small-3.1-24B-2503": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "Mistral-Large-Instruct-2411": "mistralai/Mistral-Large-Instruct-2411",
    "Qwen3-4B": "Qwen/Qwen3-4B", 
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-30B": "Qwen/Qwen3-30B-A3B",
    "Qwen3-4B-Non-Thinking": "Qwen/Qwen3-4B", 
    "Qwen3-8B-Non-Thinking": "Qwen/Qwen3-8B",
    "Qwen3-14B-Non-Thinking": "Qwen/Qwen3-14B",
    "Qwen3-32B-Non-Thinking": "Qwen/Qwen3-32B",
    "Qwen3-30B-Non-Thinking": "Qwen/Qwen3-30B-A3B",
    "gpt-oss-20b": "openai/gpt-oss-20b"
}
# {
#     "deepseek_client": AsyncOpenAI(
#         api_key=os.getenv("DEEPSEEK_API_KEY"),
#         base_url="https://api.deepseek.com"
#     ),
#     "azure_openai_client": AsyncAzureOpenAI(
#         azure_endpoint=endpoint,
#         api_key=subscription_key,
#         api_version="2024-12-01-preview",
#         timeout=600.0
#     ),
#     "vllm_client": AsyncOpenAI(
#         api_key="EMPTY",
#         base_url="http://localhost:8080/v1"
#     )
# }
async def generate_from_chat_completion(
        messages_list: List[Dict[str, str]],
        model: str,
        max_tokens=8192,
        verbose=False,
        vllm=False,
        port=8080,
        requests_per_minute=60,
        save_info=None,
        sequential=False,
        openai_client='azure',  # Add the new parameter
        **kwargs,
) -> Tuple[List[str], Dict[str, int]]:
    # Set up client based on model type
    if vllm:
        client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1"
        )
        if not "gpt-oss-20b" in model:
            full_model = model_mapping[model]    
        else:
            full_model = model
    elif "deepseek-reasoner" in model or "deepseek-chat" in model:
        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        full_model = model
    elif "gemini" in model:
        client = AsyncOpenAI(
            api_key = os.getenv("GEMINI_KEY"),
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        full_model = model_mapping[model]
    elif "gpt" in model:
        # Choose client based on openai_client parameter
        if openai_client == 'azure':
            client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=subscription_key,
                api_version="2025-03-01-preview"
            )
            print("Using Azure OpenAI client")
        elif openai_client == 'openai':
            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1"
            )
            print("Using OpenAI client")
        else:
            raise ValueError(f"Unsupported OpenAI client: {openai_client}")
        
        full_model = model
    else:
        # Default client handling for other models
        if vllm:
            client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1"
            )
            full_model = model_mapping.get(model, model)
        else:
            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1"
            )
            full_model = model
            
    # Calculate delay between requests for rate limiting
    delay_seconds = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
    
    # Initialize token counters
    token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'successful_requests': 0,
        'failed_requests': 0
    }
    
    # Helper function to update token counts
    def update_token_counts(response):
        if response and hasattr(response, "usage"):
            token_usage['prompt_tokens'] += response.usage.prompt_tokens
            token_usage['completion_tokens'] += response.usage.completion_tokens
            token_usage['total_tokens'] += response.usage.total_tokens
            token_usage['successful_requests'] += 1
        else:
            token_usage['failed_requests'] += 1
    
    # Helper function to save raw response
    async def save_raw_response(response, task, seed, prediction_path):
        if response is None or (hasattr(response, "choices") and response.usage.completion_tokens == 0):
            return  # Skip saving empty responses
            
        # Create raw directory if it doesn't exist
        os.makedirs(f"{prediction_path}/raw", exist_ok=True)
        
        # Save raw response
        raw_path = f"{prediction_path}/raw/{task}_{model}_{seed}.json"
        
        # Handle different response formats
        if hasattr(response, "model_dump_json"):
            with open(raw_path, "w") as f:
                f.write(response.model_dump_json())
        elif hasattr(response, "to_dict"):
            with open(raw_path, "w") as f:
                json.dump(response.to_dict(), f)
        else:
            with open(raw_path, "w") as f:
                json.dump(response, f)
        
        # Also extract and save code blocks
        try:
            # Create codes directory if it doesn't exist
            os.makedirs(f"{prediction_path}/codes", exist_ok=True)
            
            # Extract content from response
            if hasattr(response, "choices") and len(response.choices) > 0:
                if hasattr(response.choices[0], "message") and hasattr(response.choices[0].message, "content"):
                    content = response.choices[0].message.content
                    
                    # Extract code using regex
                    import re
                    pattern = r"```(?:cpp|c\+\+)?([\s\S]*?)```"
                    match = re.search(pattern, content, re.DOTALL)
                    
                    if match:
                        code = match.group(1).strip()
                        if code:
                            # Check and fix first line if needed
                            lines = code.split('\n')
                            first_line = lines[0].strip()
                            
                            # Fix problematic first lines
                            if first_line.lower() == "cpp" or first_line.endswith(".cpp"):
                                lines[0] = "// " + first_line
                                code = '\n'.join(lines)
                            
                            # Save extracted code
                            with open(f"{prediction_path}/codes/{task}_{model}_{seed}.cpp", "w") as f:
                                f.write(code)
                        else:
                            # Empty code block
                            with open(f"{prediction_path}/codes/{task}_{model}_{seed}.cpp", "w") as f:
                                f.write("")
                    else:
                        # No code block found
                        with open(f"{prediction_path}/codes/{task}_{model}_{seed}.cpp", "w") as f:
                            f.write("")
        except Exception as e:
            print(f"Error extracting code for {task} seed {seed}: {str(e)}")
            # Ensure an empty file is created even on error
            with open(f"{prediction_path}/codes/{task}_{model}_{seed}.cpp", "w") as f:
                f.write("")
    
    # Helper function to save error information
    async def save_error_info(error, task, seed, prediction_path):
        # Create raw directory if it doesn't exist
        os.makedirs(f"{prediction_path}/raw", exist_ok=True)
        
        print(f"Error processing task {task} with seed {seed}: {str(error)}")
    
    # Helper function to process a single message
    async def process_message(message, idx):
        try:
            response = await generate_answer(message, client, full_model, max_tokens)
            
            # Update token counts
            update_token_counts(response)
            
            # Save raw response immediately if save_info is provided
            if save_info and idx < len(save_info):
                task, seed, prediction_path = save_info[idx]
                await save_raw_response(response, task, seed, prediction_path)
                
            return idx, response
        except Exception as e:
            print(f"Error processing message {idx}: {str(e)}")
            
            # Track failed request
            token_usage['failed_requests'] += 1
            
            # # Save error information if save_info is provided
            # if save_info and idx < len(save_info):
            #     task, seed, prediction_path = save_info[idx]
            #     await save_error_info(e, task, seed, prediction_path)
                
            return idx, None
    
    responses = []
    n = len(messages_list)
    if sequential:
        # Process requests sequentially (one at a time)
        tqdm_iter = tqdm.tqdm(enumerate(messages_list), total=len(messages_list), disable=not verbose)
        for i, message in tqdm_iter:
            response = await process_message(message, i)
            responses.append(response)
    else:
        # Process requests in parallel with rate limiting
        tasks = []
        
        # Create tasks for all messages with appropriate delays
        for i, message in enumerate(tqdm.tqdm(messages_list, total=len(messages_list), disable=not verbose)):
            # Add delay for rate limiting (except for first request)
            if i > 0 and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            
            # Create a task for this message (non-blocking)
            task = asyncio.create_task(process_message(message, i))
            tasks.append(task)
        
        # collect in completion order, but stash by original index
        responses = [None] * n
        pbar = tqdm.tqdm(total=n, disable=not verbose)
        for done in asyncio.as_completed(tasks):
            idx, resp = await done
            responses[idx] = resp
            pbar.update(1)
        pbar.close()
    
    # Print token usage information
    print("\n=== Token Usage Statistics ===")
    print(f"Total prompt tokens: {token_usage['prompt_tokens']}")
    print(f"Total completion tokens: {token_usage['completion_tokens']}")
    print(f"Total tokens: {token_usage['total_tokens']}")
    print(f"Successful requests: {token_usage['successful_requests']}")
    print(f"Failed requests: {token_usage['failed_requests']}")
    print("============================\n")
    
    return responses, token_usage

# Fix the backoff decorator to properly handle exceptions
@backoff.on_exception(backoff.expo, 
                     openai.RateLimitError,  # Catch all exceptions, not just RateLimitError
                     max_tries=10, 
                     factor=2,
                     max_value=60  # Maximum wait of 60 seconds
                     )
async def generate_answer(prompt, client, model, max_tokens):
    """
    Send a prompt to OpenAI API and get the answer.
    :param prompt: the prompt to send.
    :return: the answer.
    """
    if "QwQ-32B" in model:
        args = {
            "max_tokens": 32768,
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body":{
                "top_k": 40,
                "presence_penalty": 2,
            }
        }
    elif "Qwen3" in model:
        args = {
            "max_tokens": 38912,
        }
    elif "R1-Distill" in model:
        args = {
            "max_tokens": 32768,
            "temperature": 0.6,
            "top_p": 0.95,
        }
    elif "gemini-2.5-flash" in model:
        args = {"max_tokens": 65536, "temperature": 1}
    elif "gemini-2.5-pro" in model:
        args = {'max_tokens': 65536}
    elif model == "gemini-2.0-flash":
        args = {'max_tokens': 8192}
    elif "gpt" in model:
        if "gpt-4.1" in model:
            args = {
                "max_tokens": 32768,
            }
        elif "gpt-o4-mini" in model:
            args = {
                "max_completion_tokens": 100000,
            }
            model = "o4-mini"
            if "high" in model:
                args['reasoning_effort'] = "high"
            elif "low" in model:
                args['reasoning_effort'] = "low"
            else:
                args['reasoning_effort'] = "medium"
        elif "gpt-o3-mini" in model:
            args = {
                "max_completion_tokens": 100000,
            }
            model = "o3-mini-2025-01-31"
            if "high" in model:
                args['reasoning_effort'] = "high"
            elif "low" in model:
                args['reasoning_effort'] = "low"
            else:
                args['reasoning_effort'] = "medium"
        elif "gpt-oss-20b" in model:
            args = {'max_tokens': 131072}
            if "high" in model:
                args['reasoning_effort'] = "high"
            elif "low" in model:
                args['reasoning_effort'] = "low"
            else:
                args['reasoning_effort'] = "medium"
            model = "openai/gpt-oss-20b"
    else:
        args = {'max_tokens': max_tokens}
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            timeout = 72000,
            **args   
        )
    except Exception as e:
        #print(f"Exception: {e}")
        response = {
            'id': '',
            'object': 'chat.completion',
            'created': 0,
            'model': model,
            'choices': [],
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        }
        raise e
    return response


if __name__ == "__main__":
    data_path = "data/IOI/2023"
    #tasks_dict = {"day1":["nile", "message", "tree"], "day2":["hieroglyphs", "mosaic", "sphinx"]}
    tasks_dict = {"day1":["closing", "longesttrip","soccer"], "day2":["beechtree", "overtaking", "robot"]}
    prompts = {}
    for day, tasks in tasks_dict.items():
        for task in tasks:
            with open(f"{data_path}/{day}/{task}/{task}_prompt.txt", "r") as f:
                prompts[task] = f.read()
    #task_names = ["nile", "message", "tree", "hieroglyphs", "mosaic", "sphinx"]
    task_names = ["closing", "soccer","beechtree", "overtaking", "robot"]
    query_prompts = []
    for task in task_names:
        for i in range(5):
            if os.path.exists(f"predictions/{task}_deepseek_{i}.txt"):
                continue
            query_prompts.append((task, i, prompts[task]))
    print(len(query_prompts))
    messages_list = [[{"role": "user", "content": prompt[2]}] for prompt in query_prompts]
    respoonses = asyncio.run(generate_from_chat_completion(messages_list, "deepseek-reasoner"))
    for i in range(len(query_prompts)):
        task = query_prompts[i][0]
        seed = query_prompts[i][1]
        if "choices" not in respoonses[i]:
            with open(f"predictions/{task}_deepseek_{seed}.txt", "w") as f:
                json.dump(respoonses[i].to_dict(), f)