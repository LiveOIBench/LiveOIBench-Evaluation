import os
import json
import time
import subprocess
import argparse
from typing import List, Dict, Any, Tuple
import requests
import tempfile
from models import *
import re
from evaluation.judges.problem import Problem
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBLEMS_DIR = os.environ.get("IOI_BENCH_DATA_DIR", str(REPO_ROOT / "data"))
DEFAULT_PARSE_DIR = os.environ.get("IOI_PARSED_DIR", str(REPO_ROOT / "parsed"))
DEFAULT_EVAL_DIR = os.environ.get("IOI_EVAL_RESOURCE_DIR", str(REPO_ROOT / "evaluation"))
DEFAULT_PREDICTION_DIR = os.environ.get("IOI_PREDICTIONS_DIR", str(REPO_ROOT / "ioi-predictions" / "predictions"))

class IoIEvaluation:
    def __init__(self, model_name: str, competitions, years, rounds, tasks, task_types, problems_dir: str, 
                 parse_dir, evaluation_dir: str, prediction_dir: str, vllm: bool, seeds: int, 
                 rerun: bool = False, port: int = 8080, system_prompt: bool = False,
                 requests_per_minute: int = 200, sequential: bool = False, reparse: bool = False,
                 openai_client: str = 'openai'):  # Add new parameter
        """
        Initialize the evaluation pipeline.
        
        Args:
            model_name: Name or API endpoint of the model to use
            problems_dir: Directory containing problem definitions and test cases
            output_dir: Directory to store generated solutions and evaluation results
            requests_per_minute: Maximum API requests per minute for rate limiting
            sequential: Whether to process requests sequentially (one at a time)
            reparse: Whether to reparse code from existing responses
            openai_client: Which OpenAI client to use ('azure' or 'openai')
        """
        self.model_name = model_name
        self.competitions = competitions
        self.years = years
        self.rounds = rounds
        self.tasks = tasks
        self.task_types = task_types
        self.problems_dir = problems_dir
        self.parse_dir = parse_dir
        self.evaluation_dir = evaluation_dir
        self.prediction_dir = prediction_dir
        self.vllm = vllm
        self.seeds = seeds
        self.rerun = rerun
        self.port = port
        self.system_prompt = system_prompt
        self.requests_per_minute = requests_per_minute
        self.sequential = sequential
        self.reparse = reparse
        self.openai_client = openai_client
        
        # Create output directory if it doesn't exist
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
        
        # Tracking statistics
        self.extraction_stats = {
            'success': 0,
            'failed': 0,
            'empty': 0
        }
    
    def load_problems(self) -> List[Dict[str, Any]]:
        """Load all problems from the problems directory."""
        problems = {}
        for year, contests in self.task_dict.items():
            for contest, tasks in contests.items():
                for task in tasks:
                    with open(self.problems_dir + f"/IOI/{year}/{contest}/{task}/{task}_prompt.txt", 'r') as f:
                        problems[year + "_" + contest + "_" + task] = {
                            "problem_id": year + "_" + contest + "_" + task,
                            "prompt": f.read(),
                            "task": task,
                        }
        return problems
    def discover_problems(self):
        """Discover problems based on command-line arguments"""
        problems = []
        total_tasks = 0
        
        for competition in self.competitions:
            for year in self.years:
                if self.rounds is None:
                    try:
                        rounds = os.listdir(os.path.join(self.problems_dir, competition, year))
                    except FileNotFoundError:
                        print(f"Warning: Directory not found: {os.path.join(self.problems_dir, competition, year)}")
                        continue
                for round_name in rounds:
                    problems_dir = os.path.join(self.problems_dir, competition, year, round_name)
                    
                    if not os.path.exists(problems_dir):
                        print(f"Warning: Directory not found: {problems_dir}")
                        continue
                    
                    try:
                        # Read meta_info.json to get task list
                        with open(os.path.join(problems_dir, "meta_info.json")) as f:
                            meta_info = json.load(f)

                        task_dirs = []
                        for split, tasks in meta_info.items():
                            for task in tasks:
                                if self.tasks and task not in self.tasks:
                                    continue
                                task_dir = os.path.join(problems_dir, task)
                                problem_json_path = os.path.join(task_dir, "problem.json")
                                
                                if os.path.exists(problem_json_path):
                                    with open(problem_json_path) as f:
                                        problem_config = json.load(f)
                                    task_type = problem_config.get("task_type", "unknown")
                                    if len(task_type) == 0:
                                        task_type = "batch"
                                else:
                                    task_type = "unknown"

                                if self.task_types and task_type.lower() not in self.task_types:
                                    continue
                                # Create problem info dictionary
                                problem_info = {
                                    "competition": competition,
                                    "year": year,
                                    "round": round_name,
                                    "task": task,
                                    "split": split,
                                    "dir": os.path.join(problems_dir, task),
                                    "id": f"{competition}-{year}-{round_name}-{task}",
                                    "task_type": task_type
                                }
                                problems.append(problem_info)
                                total_tasks += 1
                    
                    except Exception as e:
                        print(f"Error processing {problems_dir}: {str(e)}")
                        continue
        print(f"Found {total_tasks} tasks to evaluate.\n")
        return problems
    
    def generate_solution(self, problems, model, seeds=5) -> str:
        """Generate code solution for a given problem using the specified model."""
        print(f"Generating solution for model: {model}")
        
        query_prompts = []
        save_info = []
        
        # Track all problems for reparsing
        all_problems = []
        for problem in problems:
            parse_task_path = os.path.join(self.parse_dir, problem['competition'], problem['year'], problem['round'], problem['task'])
            problem_obj = Problem(problem['dir'], problem['task'], problem['year'], problem['competition'], problem['round'], problem['split'], parse_task_path)
            prompt = problem_obj.get_prompt()
            prediction_path = os.path.join(self.prediction_dir, model, problem['competition'], problem['year'], problem['round'], problem['task'])
            
            # Remember all problems for potential reparsing
            all_problems.append((problem['task'], prediction_path))
            
            # Only query for non-existing responses if not reparsing
            if not self.reparse:
                for i in range(seeds):
                    if os.path.exists(prediction_path + f"/raw/{problem['task']}_{model}_{i}.json") and not self.rerun:
                        continue
                    # Store prompt
                    query_prompts.append((problem['task'], i, prompt, prediction_path))
                    # Store saving information
                    save_info.append((problem['task'], i, prediction_path))
        
        # Process generation of new responses
        solutions = {}
        if not self.reparse:
            print(f"Total Problems to generate: {len(query_prompts)}")
            
            if len(query_prompts) == 0:
                print("No new problems to generate.")
            else:
                if self.system_prompt:
                    if "QwQ" in self.model_name:
                        system_prompt = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
                    elif "R1-Distill" in self.model_name:
                        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
                    else:
                        system_prompt = "You are a helpful assistant."
                    messages_list = [[{'role': 'system', 'content': system_prompt}, {"role": "user", "content": prompt[2]}] for prompt in query_prompts]
                else:
                    messages_list = [[{"role": "user", "content": prompt[2]}] for prompt in query_prompts]
                
                # Generate responses with immediate raw response saving
                responses, token_usage = asyncio.run(generate_from_chat_completion(
                    messages_list, 
                    model, 
                    vllm=self.vllm, 
                    port=self.port,
                    requests_per_minute=self.requests_per_minute,
                    save_info=save_info,
                    verbose=True,
                    sequential=self.sequential,  # Pass the sequential flag
                    openai_client=self.openai_client  # Pass the parameter
                ))

                # Save token usage to a JSON file with timestamp
                timestamp = int(time.time())
                token_usage_path = os.path.join(self.evaluation_dir, f"{model}_token_usage_{timestamp}.json")
                with open(token_usage_path, "w") as f:
                    json.dump(token_usage, f, indent=2)

                print(f"\nToken usage information saved to {token_usage_path}")
                
                # Process responses and extract code
                for i, response in enumerate(responses):
                    task = query_prompts[i][0]
                    seed = query_prompts[i][1]
                    prediction_path = query_prompts[i][3]
                    
                    self._extract_and_save_code(task, seed, model, prediction_path, response, solutions)
        
        # Reparse code from existing responses if requested
        if self.reparse:
            print("Reparsing code from existing responses...")
            for task, prediction_path in all_problems:
                if not os.path.exists(prediction_path + "/raw"):
                    continue
                    
                for file in os.listdir(prediction_path + "/raw"):
                    if file.startswith(f"{task}_{model}_") and file.endswith(".json"):
                        try:
                            seed = int(file.split('_')[-1].split('.')[0])
                            response_file = os.path.join(prediction_path, "raw", file)
                            
                            with open(response_file, 'r') as f:
                                response_data = json.load(f)
                            
                            # Create a mock response object similar to OpenAI API response
                            if isinstance(response_data, dict) and "choices" in response_data:
                                # This looks like an OpenAI-style response
                                response = type('MockResponse', (), {})()
                                response.choices = [type('MockChoice', (), {})()]
                                
                                # Handle different response formats
                                if "message" in response_data["choices"][0]:
                                    response.choices[0].message = type('MockMessage', (), {})()
                                    if isinstance(response_data["choices"][0]["message"], dict) and "content" in response_data["choices"][0]["message"]:
                                        response.choices[0].message.content = response_data["choices"][0]["message"]["content"]
                                    else:
                                        response.choices[0].message.content = str(response_data["choices"][0]["message"])
                                else:
                                    # Skip this file if it doesn't have the expected structure
                                    print(f"Skipping {file} - unexpected format")
                                    continue
                                    
                                self._extract_and_save_code(task, seed, model, prediction_path, response, solutions)
                            else:
                                print(f"Skipping {file} - not a valid response")
                                self.extraction_stats['failed'] += 1
                        except Exception as e:
                            print(f"Error reparsing {file}: {e}")
                            self.extraction_stats['failed'] += 1
        
        # Print extraction statistics
        print("\nCode Extraction Statistics:")
        print(f"  Successful extractions: {self.extraction_stats['success']}")
        print(f"  Failed extractions: {self.extraction_stats['failed']}")
        print(f"  Empty code blocks: {self.extraction_stats['empty']}")
        
        return solutions
    
    def _extract_and_save_code(self, task, seed, model, prediction_path, response, solutions):
        """Helper method to extract and save code from a response"""
        # Create codes directory if it doesn't exist
        os.makedirs(prediction_path + "/codes", exist_ok=True)
        
        if task not in solutions:
            solutions[task] = []
        
        # Skip processing if response is None or doesn't have choices
        if response is None or not hasattr(response, "choices") or len(response.choices) == 0:
            solutions[task].append("")
            self.extraction_stats['failed'] += 1
            with open(prediction_path + f"/codes/{task}_{model}_{seed}.cpp", "w") as f:
                f.write("")
            print(f"No valid response for {task} seed {seed}")
            return
        
        # Extract code from content using prioritized patterns
        content = response.choices[0].message.content
        try:
            patterns = [
                rf"```(?:{re.escape(task)}\.(?:cpp|c))\s*([\s\S]*?)```",  # 1) ```{task}.cpp|c ... ```
                r"```(?:cpp|c)\s*([\s\S]*?)```",                           # 2) ```cpp|c ... ```
                r"```([\s\S]*?)```",                                        # 3) ``` ... ```
            ]

            code = ""
            found = False
            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    code = max(matches, key=lambda x: len(x.split('\n'))).strip()
                    found = True
                    break

            if found:
                if code:
                    # Check and fix first line if it contains just "cpp" or ends with .cpp
                    lines = code.split('\n')
                    if len(lines) < 5:
                        print(f"Code block too short {prediction_path}")
                    first_line = lines[0].strip()

                    # Check if first line is just "cpp" or contains a filename ending with .cpp
                    if first_line.lower() == "cpp" or first_line.endswith(".cpp"):
                        lines[0] = "// " + first_line
                        code = '\n'.join(lines)

                    with open(prediction_path + f"/codes/{task}_{model}_{seed}.cpp", "w") as f:
                        f.write(code)
                    solutions[task].append(code)
                    self.extraction_stats['success'] += 1
                else:
                    # Empty code block
                    with open(prediction_path + f"/codes/{task}_{model}_{seed}.cpp", "w") as f:
                        f.write("")
                    solutions[task].append("")
                    self.extraction_stats['empty'] += 1
                    print(f"Empty code block found for {task} seed {seed}")
            else:
                # No code block found
                with open(prediction_path + f"/codes/{task}_{model}_{seed}.cpp", "w") as f:
                    f.write("")
                solutions[task].append("")
                self.extraction_stats['failed'] += 1
                print(f"No code block found for {task} seed {seed}")
        except Exception as e:
            with open(prediction_path + f"/codes/{task}_{model}_{seed}.cpp", "w") as f:
                f.write("")
            print(f"Failed to extract code for {task} seed {seed}: {e}")
            solutions[task].append("")
            self.extraction_stats['failed'] += 1
    
    def run_pipeline(self):
        """Run the complete evaluation pipeline."""
        problems = self.discover_problems()
        self.generate_solution(problems, self.model_name, seeds=self.seeds)
        

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Code Generation and Evaluation Pipeline')
    parser.add_argument('--model', type=str, required=True, help='Model name or API endpoint')
    parser.add_argument("--competitions", nargs="+", default=["IOI"], 
                        help="List of competitions to evaluate (e.g., IOI, CEOI)")
    parser.add_argument("--years", nargs="+", default=["2024"], 
                        help="List of years to evaluate")
    parser.add_argument("--rounds", nargs="+", default=None, 
                        help="List of competition rounds to evaluate (e.g., contest, practice)")
    parser.add_argument("--tasks", nargs="+", default=None, 
                        help="Specific tasks to evaluate (if omitted, all tasks are evaluated)")
    parser.add_argument('--problems_dir', type=str, default=DEFAULT_PROBLEMS_DIR, help='Directory containing problem definitions')
    parser.add_argument('--task_types', nargs="+", default=None, help='Task types to evaluate (e.g., batch, interactive)')
    parser.add_argument('--parse_dir', type=str, default=DEFAULT_PARSE_DIR, help='Directory for parsed problems')
    parser.add_argument('--evaluation_dir', type=str, default=DEFAULT_EVAL_DIR, help='Output directory for results')
    parser.add_argument('--prediction_dir', type=str, default=DEFAULT_PREDICTION_DIR, help='Output directory for predictions')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to use for each problem')
    parser.add_argument('--vllm', action='store_true', help='Use VLLM model for code generation')
    parser.add_argument('--rerun', action='store_true', help='Whether to rerun the pipeline')
    parser.add_argument('--reparse', action='store_true', help='Reparse code from existing responses')
    parser.add_argument("--system_prompt", action="store_true", help="Use system prompt for VLLM")
    parser.add_argument('--port', type=int, default=8080, help='Port for VLLM server')
    parser.add_argument('--requests_per_minute', type=int, default=200, 
                        help='Maximum API requests per minute (for rate limiting)')
    parser.add_argument('--sequential', action='store_true', 
                        help='Process requests sequentially (one at a time)')
    parser.add_argument('--openai_client', type=str, default='openai', help='Which OpenAI client to use (azure or openai)')
    
    args = parser.parse_args()

    
    pipeline = IoIEvaluation(
        model_name=args.model,
        competitions=args.competitions,
        years=args.years,
        rounds=args.rounds,
        tasks=args.tasks,
        task_types=args.task_types,
        problems_dir=args.problems_dir,
        parse_dir=args.parse_dir,
        evaluation_dir=args.evaluation_dir,
        prediction_dir=args.prediction_dir,
        vllm=args.vllm,
        seeds=args.seeds,
        rerun=args.rerun,
        port=args.port,
        system_prompt=args.system_prompt,
        requests_per_minute=args.requests_per_minute,
        sequential=args.sequential,
        reparse=args.reparse,
        openai_client=args.openai_client
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()