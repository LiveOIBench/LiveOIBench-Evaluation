import os
import json
import time
import subprocess
import argparse
from typing import Any, Dict, List, Tuple
import requests
import tempfile
import asyncio
import re
from pathlib import Path

from models import generate_from_chat_completion
from model_config import get_model_config
from code_extractor import CodeExtractor, ExtractionStats
from judges.problem import Problem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.getenv("LIVEOIBENCH_ROOT"))
DEFAULT_PROBLEMS_DIR = str(DEFAULT_ROOT / "data")
DEFAULT_PARSE_DIR = str(DEFAULT_ROOT / "data")
DEFAULT_EVAL_DIR = str(DEFAULT_ROOT / "evaluation")
DEFAULT_PREDICTION_DIR = str(DEFAULT_ROOT / "predictions")

class LiveOIBenchEvaluation:
    def __init__(
        self,
        model_name: str,
        competitions,
        years,
        rounds,
        tasks,
        task_types,
        problems_dir: str,
        parse_dir: str,
        evaluation_dir: str,
        prediction_dir: str,
        vllm: bool,
        seeds: int,
        rerun: bool = False,
        port: int = 8080,
        system_prompt: bool = False,
        requests_per_minute: int = 200,
        sequential: bool = False,
        reparse: bool = False,
        openai_client: str = "openai",
        mode: str = "folder",
    ):
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
        self.mode = mode
        self.json_output_dir = Path(self.prediction_dir).expanduser() / self.model_name
        
        # Create output directory if it doesn't exist
        os.makedirs(evaluation_dir, exist_ok=True)

        # Results storage
        self.results = {}
        self.code_records: Dict[str, Dict[str, str]] = {}
        self.raw_records: Dict[str, Dict[str, Any]] = {}

        # Tracking statistics
        self.extraction_stats = ExtractionStats()
    
    def load_problems(self) -> List[Dict[str, Any]]:
        """Load all problems from the problems directory."""
        problems = {}
        for year, contests in self.task_dict.items():
            for contest, tasks in contests.items():
                for task in tasks:
                    with open(self.problems_dir + f"/LiveOIBench/{year}/{contest}/{task}/{task}_prompt.txt", 'r') as f:
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
                            if split != "tasks":
                                continue
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
    
    def generate_solution(self, problems, model, seeds=5) -> Dict[str, List[str]]:
        """Generate code solution for a given problem using the specified model."""
        print(f"Generating solution for model: {model}")
        
        query_prompts = []
        save_info: List[Tuple[str, int, str]] = []
        solutions: Dict[str, List[str]] = {}
        
        # Track all problems for reparsing
        all_problems = []
        existing_keys: set[tuple[str, int]] = set()
        if self.mode == "json":
            existing_keys = self._load_existing_json_predictions(solutions)
        for problem in problems:
            parse_task_path = os.path.join(self.parse_dir, problem['competition'], problem['year'], problem['round'], problem['task'])
            problem_obj = Problem(problem['dir'], problem['task'], problem['year'], problem['competition'], problem['round'], problem['split'], parse_task_path)
            prompt = problem_obj.get_prompt()
            prediction_path = os.path.join(self.prediction_dir, model, problem['competition'], problem['year'], problem['round'], problem['task'])
            
            # Remember all problems for potential reparsing
            all_problems.append((problem, prediction_path))
            
            # Only query for non-existing responses if not reparsing
            if not self.reparse:
                for i in range(seeds):
                    problem_id = problem.get("id") or f"{problem['competition']}-{problem['year']}-{problem['round']}-{problem['task']}"
                    existing_key = (problem_id, i)
                    if self.mode == "json" and not self.rerun and existing_key in existing_keys:
                        continue
                    if self.mode == "folder" and os.path.exists(prediction_path + f"/raw/{problem['task']}_{model}_{i}.json") and not self.rerun:
                        continue
                    # Store prompt
                    query_prompts.append((problem, i, prompt, prediction_path))
                    # Store saving information
                    if self.mode == "folder":
                        save_info.append((problem['task'], i, prediction_path))
        
        # Process generation of new responses
        if not self.reparse:
            print(f"Total Problems to generate: {len(query_prompts)}")
            
            if len(query_prompts) == 0:
                print("No new problems to generate.")
            else:
                # Build messages list with optional system prompt
                messages_list = []
                model_config = get_model_config(self.model_name)

                for prompt in query_prompts:
                    messages = []

                    # Add system prompt if requested and available in config
                    if self.system_prompt and model_config.system_prompt:
                        messages.append({
                            'role': 'system',
                            'content': model_config.system_prompt
                        })

                    # Add user prompt
                    messages.append({
                        "role": "user",
                        "content": prompt[2]
                    })

                    messages_list.append(messages)
                
                # Generate responses with immediate raw response saving
                responses, token_usage = asyncio.run(generate_from_chat_completion(
                    messages_list, 
                    model, 
                    vllm=self.vllm, 
                    port=self.port,
                    requests_per_minute=self.requests_per_minute,
                    save_info=save_info if self.mode == "folder" else None,
                    verbose=True,
                    sequential=self.sequential,  # Pass the sequential flag
                    openai_client=self.openai_client  # Pass the parameter
                ))

                # Save token usage to a JSON file alongside predictions (timestamped)
                timestamp = int(time.time())
                token_usage_dir = Path(self.prediction_dir).expanduser() / model
                token_usage_dir.mkdir(parents=True, exist_ok=True)
                token_usage_path = token_usage_dir / f"{model}_token_usage_{timestamp}.json"
                with open(token_usage_path, "w") as f:
                    json.dump(token_usage, f, indent=2)

                print(f"\nToken usage information saved to {token_usage_path}")
                
                # Process responses and extract code
                for i, response in enumerate(responses):
                    problem_meta = query_prompts[i][0]
                    seed = query_prompts[i][1]
                    prediction_path = query_prompts[i][3]
                    
                    self._extract_and_save_code(problem_meta, seed, model, prediction_path, response, solutions)
        
        # Reparse code from existing responses if requested
        if self.mode == "folder" and self.reparse:
            print("Reparsing code from existing responses...")
            for problem_meta, prediction_path in all_problems:
                if not os.path.exists(prediction_path + "/raw"):
                    continue
                    
                for file in os.listdir(prediction_path + "/raw"):
                    if file.startswith(f"{problem_meta['task']}_{model}_") and file.endswith(".json"):
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
                                    
                                self._extract_and_save_code(problem_meta, seed, model, prediction_path, response, solutions)
                            else:
                                print(f"Skipping {file} - not a valid response")
                                self.extraction_stats.record('failed')
                        except Exception as e:
                            print(f"Error reparsing {file}: {e}")
                            self.extraction_stats.record('failed')
        elif self.mode == "json" and self.reparse:
            print("Reparse mode is only supported for folder output; skipping reparse.")
        
        if self.mode == "json":
            self._write_json_predictions()

        # Print extraction statistics
        self.extraction_stats.print_summary()

        return solutions
    
    def _extract_and_save_code(self, problem_meta: Dict[str, Any], seed: int, model: str, prediction_path: str, response: Any, solutions: Dict[str, List[str]]):
        """Helper method to extract and save code from a response"""
        task = problem_meta["task"]
        problem_id = problem_meta.get("id") or f"{problem_meta.get('competition')}-{problem_meta.get('year')}-{problem_meta.get('round')}-{task}"
        if task not in solutions:
            solutions[task] = []

        # Extract content from response
        content = CodeExtractor.extract_from_response(response)

        raw_response = self._response_to_json(response)

        if content is None:
            # No valid response
            code = ""
            status = "failed"
            if self.mode == "folder":
                CodeExtractor.save_code(None, prediction_path, task, model, seed)
            else:
                code_filename = f"{task}_{model}_{seed}.cpp"
                self.code_records.setdefault(problem_id, {})[code_filename] = code
                raw_filename = f"{task}_{model}_{seed}.json"
                self.raw_records.setdefault(problem_id, {})[raw_filename] = raw_response
        else:
            # Extract code from content
            code, status = CodeExtractor.extract_code(content, task)

            if self.mode == "folder":
                # Save the code
                CodeExtractor.save_code(code, prediction_path, task, model, seed)
            else:
                code_filename = f"{task}_{model}_{seed}.cpp"
                self.code_records.setdefault(problem_id, {})[code_filename] = code or ""
                raw_filename = f"{task}_{model}_{seed}.json"
                self.raw_records.setdefault(problem_id, {})[raw_filename] = raw_response

        # Update solutions and stats
        solutions[task].append(code if code else "")
        self.extraction_stats.record(status)

        # Print warnings
        if status == 'failed':
            print(f"No valid response for {task} seed {seed}")
        elif status == 'empty':
            print(f"Empty code block found for {task} seed {seed}")
        elif status == 'not_found':
            print(f"No code block found for {task} seed {seed}")
        elif code and len(code.split('\n')) < 5:
            print(f"Code block too short for {task} seed {seed}")

    @staticmethod
    def _response_to_json(response: Any) -> Any:
        """Convert API response objects to JSON-serializable payloads."""
        if response is None:
            return None
        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if hasattr(response, "model_dump_json"):
                return json.loads(response.model_dump_json())
            if hasattr(response, "to_dict"):
                return response.to_dict()
            if isinstance(response, dict):
                return response
            return json.loads(json.dumps(response, default=str))
        except Exception as exc:
            return {"error": f"Failed to serialize response: {exc}"}

    @staticmethod
    def _extract_seed_from_filename(filename: str) -> int | None:
        """Extract seed number from filenames like task_model_seed.ext."""
        try:
            base = Path(filename).stem
            seed_str = base.rsplit("_", 1)[-1]
            return int(seed_str)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _task_from_problem_id(problem_id: str) -> str:
        parts = problem_id.split("-")
        if len(parts) >= 4:
            return "-".join(parts[3:])
        return problem_id

    def _load_existing_json_predictions(self, solutions: Dict[str, List[str]]) -> set[tuple[str, int]]:
        """
        Load existing JSON predictions when running in json mode to reuse results.

        Returns a set of keys (problem_id, seed) already present.
        """
        keys: set[tuple[str, int]] = set()
        codes_path = self.json_output_dir / f"{self.model_name}_code.json"
        raw_path = self.json_output_dir / f"{self.model_name}_raw.json"

        if codes_path.exists():
            try:
                with codes_path.open("r", encoding="utf-8") as f:
                    self.code_records = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.code_records = {}

        if raw_path.exists():
            try:
                with raw_path.open("r", encoding="utf-8") as f:
                    self.raw_records = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.raw_records = {}

        for problem_id, files in self.code_records.items():
            for filename, code in files.items():
                seed = self._extract_seed_from_filename(filename)
                if seed is not None:
                    keys.add((problem_id, seed))
                task = self._task_from_problem_id(problem_id)
                solutions.setdefault(task, []).append(code or "")
                self.extraction_stats.record("success")

        return keys

    def _write_json_predictions(self) -> None:
        """Persist collected predictions as JSON when running in JSON mode."""
        self.json_output_dir.mkdir(parents=True, exist_ok=True)
        codes_path = self.json_output_dir / f"{self.model_name}_code.json"
        raw_path = self.json_output_dir / f"{self.model_name}_raw.json"

        with codes_path.open("w", encoding="utf-8") as codes_file:
            json.dump(self.code_records, codes_file, ensure_ascii=False, separators=(",", ":"))

        with raw_path.open("w", encoding="utf-8") as raw_file:
            json.dump(self.raw_records, raw_file, ensure_ascii=False, separators=(",", ":"))

        print(f"Wrote JSON code predictions to {codes_path}")
        print(f"Wrote JSON raw responses to {raw_path}")
    
    def run_pipeline(self):
        """Run the complete evaluation pipeline."""
        problems = self.discover_problems()
        self.generate_solution(problems, self.model_name, seeds=self.seeds)
        

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Code Generation and Evaluation Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Model name or API endpoint")
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=["LiveOIBench"],
        help="List of competitions to evaluate (e.g., LiveOIBench, CEOI)",
    )
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
    parser.add_argument(
        '--mode',
        type=str,
        choices=["folder", "json"],
        default="folder",
        help='Output mode: "folder" writes raw/codes to the prediction directory, "json" writes JSONL summaries.',
    )
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

    
    pipeline = LiveOIBenchEvaluation(
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
        openai_client=args.openai_client,
        mode=args.mode,
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
