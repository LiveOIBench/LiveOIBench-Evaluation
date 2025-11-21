"""
Script judge implementation for script-based problems.
"""
import os
import time
import subprocess
import threading
import math
import queue
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .base_judge import BaseJudge
from .problem import Problem


def set_limits(cpu_time_limit, memory_limit_bytes):
    """Apply CPU and memory limits before execution."""
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


class ScriptJudge(BaseJudge):
    """
    Judge implementation for script-based problems.

    This judge handles problems that use custom evaluation scripts:
    - setup.sh: Prepares the environment and compiles solution
    - evaluate.sh: Runs and evaluates individual test cases
    - Supports partial scoring and custom validation logic
    """

    def setup(self, problem: Problem, solution_file: str) -> Tuple[str, str]:
        """
        Set up the environment for script-based judging.

        Creates a working directory for the problem.

        Args:
            problem: Problem object
            solution_file: Path to solution source file

        Returns:
            Tuple of (solution_file_path, work_directory_path)
        """
        base_name = os.path.basename(solution_file).replace(".cpp", "")
        work_dir = os.path.join(self.work_path, problem.id, base_name)

        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)

        return solution_file, work_dir

    def compile(
        self,
        solution_file: str,
        work_dir: str,
        problem: Problem = None,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Run setup.sh script to prepare environment and compile solution.

        Args:
            solution_file: Path to solution source
            work_dir: Working directory (base directory, not prep directory)
            problem: Problem object
            verbose: Print compilation details

        Returns:
            Tuple of (success, message)
        """
        # This will be called by judge() which runs the full setup
        # For script-based problems, compilation happens in run_script_evaluation
        return True, "Script compilation deferred to evaluation phase"

    def run_test_case(
        self,
        problem: Problem,
        solution_basename: str,
        input_file: str,
        worker_dir: str,
        cpu_time_limit: float,
        memory_limit_mb: float,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single test case using evaluate.sh script.

        Args:
            problem: Problem object
            solution_basename: Base name of solution file (e.g., "solution.cpp")
            input_file: Path to input file
            worker_dir: Worker directory with evaluate.sh and compiled solution
            cpu_time_limit: CPU time limit in seconds
            memory_limit_mb: Memory limit in MB
            verbose: Print execution details

        Returns:
            Dictionary with test result including: test_case, correct, cpu_time, memory, exit_code, score
        """
        test_base_name = os.path.basename(input_file).replace('.in', '')
        abs_input_path = os.path.abspath(input_file)

        if verbose:
            print(f"Running evaluate.sh for test {test_base_name}")

        # Measure start time
        start_wall_time = time.time()

        # Use GNU time to capture resource usage
        time_format = "\\nuser %U\\nsystem %S\\nmaxmem %M"
        time_cmd = f"/usr/bin/time -f '{time_format}'"

        # Run evaluate.sh with input file
        cmd = f"cd {worker_dir} && {time_cmd} ./evaluate.sh {solution_basename} {abs_input_path}"

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            preexec_fn=lambda: set_limits(
                int(math.ceil(cpu_time_limit)),
                int(math.ceil(memory_limit_mb * 1024 * 1024))
            )
        )

        timeout = False
        try:
            stdout, stderr = process.communicate(timeout=cpu_time_limit)
        except subprocess.TimeoutExpired:
            print(f"Timeout reached for test {test_base_name}. Terminating process...")
            process.terminate()

            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                print("Process didn't terminate, sending SIGKILL...")
                process.kill()

                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    print("Failed to collect output even after SIGKILL. Process may be zombie.")
                    stdout = ""
                    stderr = ""
            timeout = True

        return_code = -11 if timeout else (process.returncode or 0)
        elapsed_wall_time = time.time() - start_wall_time

        # Parse resource usage from GNU time output
        user_match = re.search(r'user\s+([\d\.]+)', stderr)
        system_match = re.search(r'system\s+([\d\.]+)', stderr)
        mem_match = re.search(r'maxmem\s+(\d+)', stderr)

        user_time = float(user_match.group(1)) if user_match else 0.0
        system_time = float(system_match.group(1)) if system_match else 0.0
        cpu_time = user_time + system_time

        memory_kb = float(mem_match.group(1)) if mem_match else 0.0
        memory_mb = memory_kb / 1024

        # Look for common runtime error patterns
        error_patterns = [
            r'(Segmentation fault)',
            r'(Runtime error:.*)',
            r'(Floating point exception)',
            r'(Aborted)',
            r'(Stack overflow)',
            r'(std::bad_alloc)',
            r'(Exception in thread.*)',
            r'(java\.lang\..*Exception:.*)',
            r'(Error:.*)',
            r'(Fatal error:.*)'
        ]

        for pattern in error_patterns:
            if re.search(pattern, stderr):
                return_code = -1
                break

        # Read results file
        results_file = f"{test_base_name}_{solution_basename.replace('.cpp', '')}.results.txt"
        results_path = os.path.join(worker_dir, results_file)

        result_content = ""
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                result_content = f.read()
            os.remove(results_path)

        # Determine correctness and score
        correct = False
        score = 0.0
        try:
            score = float(result_content.strip())
            correct = score == 1.0
        except ValueError:
            score = 0.0

        result = {
            "test_case": test_base_name,
            "wall_time": elapsed_wall_time,
            "cpu_time": cpu_time,
            "memory": memory_mb,
            "exit_code": return_code,
            "correct": correct,
            "score": score,
            "log": result_content,
            "script_output": stdout,
            "script_error": stderr
        }

        if verbose:
            print(f"[{test_base_name}] Finished in {elapsed_wall_time:.3f}s | "
                  f"Score: {score} | Correct: {correct}")

        return result

    def evaluate(
        self,
        solution_file: str,
        work_dir: str,
        problem: Problem = None,
        verbose: bool = False,
        save_output: bool = False,
        generate_gold_output: bool = False,
        max_workers: int = 10,
        stop_on_failure: bool = False,
        keep_executables: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run evaluation using setup.sh + evaluate.sh scripts.

        Args:
            solution_file: Path to solution source
            work_dir: Base working directory
            problem: Problem object
            verbose: Print progress
            save_output: Unused for script-based
            generate_gold_output: Unused for script-based
            max_workers: Number of parallel workers
            stop_on_failure: Stop subtask on first failure

        Returns:
            List of test case results, or None if setup fails
        """
        setup_script = problem.get_setup_script()
        evaluate_script = problem.get_evaluation_script()
        inputs = problem.get_test_inputs()
        cpu_time_limit = problem.get_time_limit()
        memory_limit_mb = problem.get_memory_limit()

        # Special case: limit workers for certain problems
        if problem.task == "message":
            max_workers = 2

        # Set up preparation directory for running setup.sh once
        prep_dir = os.path.join(work_dir, "prep")
        os.makedirs(prep_dir, exist_ok=True)

        solution_basename = os.path.basename(solution_file)
        solution_dest = os.path.join(prep_dir, solution_basename)
        os.system(f"cp {solution_file} {solution_dest}")

        # Copy all grader files to prep directory
        for file in os.listdir(problem.grader_dir):
            source_path = os.path.join(problem.grader_dir, file)
            if os.path.isfile(source_path):
                dest_path = os.path.join(prep_dir, file)
                os.system(f"cp {source_path} {dest_path}")

        # Make scripts executable
        if setup_script:
            setup_dest = os.path.join(prep_dir, "setup.sh")
            os.system(f"chmod +x {setup_dest}")

        evaluate_dest = os.path.join(prep_dir, "evaluate.sh")
        os.system(f"chmod +x {evaluate_dest}")

        if verbose:
            print(f"Prepared evaluation directory at {prep_dir}")

        # Run setup.sh once if it exists
        if setup_script:
            if verbose:
                print(f"Running setup.sh for {solution_basename}...")

            setup_cmd = f"cd {prep_dir} && ./setup.sh {solution_basename}"
            try:
                setup_output = subprocess.run(
                    setup_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if setup_output.returncode != 0:
                    error_msg = f"Setup script failed with code {setup_output.returncode}:\n"
                    error_msg += f"STDOUT: {setup_output.stdout}\n"
                    error_msg += f"STDERR: {setup_output.stderr}"
                    if verbose:
                        print(error_msg)
                    return None

                if verbose:
                    print(f"Setup completed successfully")
                    if setup_output.stdout:
                        print(f"Setup output: {setup_output.stdout}")
            except Exception as e:
                if verbose:
                    print(f"Error running setup script: {str(e)}")
                return None

        # Create worker directories
        worker_dirs = self._create_worker_directories(
            prep_dir, solution_dest, work_dir, max_workers, verbose
        )

        # Worker pool management
        worker_pool = queue.Queue()
        for i in range(len(worker_dirs)):
            worker_pool.put(i)
        worker_pool_lock = threading.Lock()

        def run_single_test(idx, input_file):
            # Get an available worker
            with worker_pool_lock:
                worker_idx = worker_pool.get()

            worker_dir = worker_dirs[worker_idx]

            try:
                if verbose:
                    test_base_name = os.path.basename(input_file).replace('.in', '')
                    print(f"Running evaluate.sh for test {idx+1}/{len(inputs)}: "
                          f"{test_base_name} (worker {worker_idx})")

                result = self.run_test_case(
                    problem, solution_basename, input_file, worker_dir,
                    cpu_time_limit, memory_limit_mb, verbose=False
                )

                return result
            except Exception as e:
                if verbose:
                    print(f"Error running evaluate.sh for test: {str(e)}")

                test_base_name = os.path.basename(input_file).replace('.in', '')
                return {
                    "test_case": test_base_name,
                    "correct": False,
                    "error": f"Evaluation error: {str(e)}"
                }
            finally:
                # Return the worker to the pool
                with worker_pool_lock:
                    worker_pool.put(worker_idx)

        try:
            results = []

            if stop_on_failure:
                # Sequential evaluation with early stopping
                results = self._evaluate_with_early_stopping(
                    problem, inputs, run_single_test, verbose
                )
            else:
                # Parallel evaluation
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(run_single_test, idx, input_file)
                        for idx, input_file in enumerate(inputs)
                    ]

                    for future in futures:
                        results.append(future.result())

            if verbose:
                print("\n=== Script-Based Evaluation Summary ===")
                for result in results:
                    print(f"Test: {result['test_case']}, Correct: {result['correct']}, "
                          f"Wall Time: {result.get('wall_time', 'N/A'):.3f}s, "
                          f"CPU Time: {result.get('cpu_time', 'N/A'):.3f}s, "
                          f"Memory: {result.get('memory', 'N/A'):.2f} MB, "
                          f"Exit Code: {result['exit_code']}")

            return results
        finally:
            if not keep_executables:
                # Clean up all directories
                shutil.rmtree(prep_dir, ignore_errors=True)
                for worker_dir in worker_dirs:
                    shutil.rmtree(worker_dir, ignore_errors=True)
                if verbose:
                    print(f"Cleaned up preparation directory and {len(worker_dirs)} worker directories")

    def _create_worker_directories(
        self,
        prep_dir: str,
        solution_dest: str,
        work_dir: str,
        max_workers: int,
        verbose: bool
    ) -> List[str]:
        """
        Create worker directories for parallel execution.

        Args:
            prep_dir: Preparation directory with compiled files
            solution_dest: Path to solution file in prep directory
            work_dir: Base working directory
            max_workers: Number of workers to create
            verbose: Print progress

        Returns:
            List of worker directory paths
        """
        worker_dirs = []
        for i in range(max_workers):
            worker_dir = f"{work_dir}_worker_{i}"
            os.makedirs(worker_dir, exist_ok=True)

            # Copy the evaluate.sh script
            os.system(f"cp {prep_dir}/evaluate.sh {worker_dir}/")

            # Copy the solution file
            os.system(f"cp {solution_dest} {worker_dir}/")

            # Copy all compiled binaries and necessary files from prep_dir
            for file in os.listdir(prep_dir):
                source_path = os.path.join(prep_dir, file)
                if os.path.isfile(source_path) and not file.startswith("setup"):
                    dest_path = os.path.join(worker_dir, file)
                    os.system(f"cp {source_path} {dest_path}")
                    # Ensure executables keep their permissions
                    if os.access(source_path, os.X_OK):
                        os.system(f"chmod +x {dest_path}")

            worker_dirs.append(worker_dir)

        if verbose:
            print(f"Created {len(worker_dirs)} worker directories for parallel evaluation")

        return worker_dirs

    def _evaluate_with_early_stopping(
        self,
        problem: Problem,
        inputs: List[str],
        run_single_test,
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """
        Evaluate test cases with early stopping per subtask.

        Args:
            problem: Problem object
            inputs: List of input file paths
            run_single_test: Function to run a single test
            verbose: Print progress

        Returns:
            List of test case results
        """
        subtasks = problem.get_subtasks()
        test_case_map = {}
        for input_file in inputs:
            base_name = os.path.basename(input_file).replace(".in", "")
            test_case_map[base_name] = input_file

        already_run = set()
        results_dict = {}
        results = []

        for subtask_id in sorted(subtasks.keys(), key=lambda x: int(x)):
            subtask = subtasks[subtask_id]

            if verbose:
                print(f"Evaluating subtask {subtask_id}...")

            for test_case in subtask['testcases']:
                if test_case in already_run:
                    results.append(results_dict[test_case])
                    if not results_dict[test_case]['correct']:
                        self._mark_remaining_tests_as_failed(subtask, subtask_id, already_run, results, results_dict)
                        break
                    continue

                already_run.add(test_case)

                if test_case not in test_case_map:
                    if verbose:
                        print(f"Warning: Test case {test_case} not found in input files")
                    continue

                input_file = test_case_map[test_case]

                if verbose:
                    print(f"Running script evaluation for test case {test_case} from subtask {subtask_id}...")

                idx = list(inputs).index(input_file) if input_file in inputs else -1
                result = run_single_test(idx, input_file)
                results.append(result)
                results_dict[test_case] = result

                if not result['correct']:
                    if verbose:
                        print(f"Script evaluation for test case {test_case} failed. "
                              f"Skipping remaining tests in subtask {subtask_id}.")
                    self._mark_remaining_tests_as_failed(subtask, subtask_id, already_run, results, results_dict)
                    break

        return results

    def judge(
        self,
        problem: Problem,
        solution_file: str,
        verbose: bool = False,
        save_output: bool = False,
        generate_gold_output: bool = False,
        max_workers: int = 10,
        stop_on_failure: bool = False,
        keep_executables: bool = False
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Judge a solution for a script-based problem.

        Overrides base implementation to handle script-based workflow.

        Args:
            problem: Problem object
            solution_file: Path to solution source file
            verbose: Print progress
            save_output: Unused for script-based
            generate_gold_output: Unused for script-based
            max_workers: Number of parallel workers
            stop_on_failure: Stop subtask on first failure

        Returns:
            Tuple of (score_info, detailed_results)
        """
        if verbose:
            print(f"Judging script-based problem: {problem.id}")

        # Setup
        _, work_dir = self.setup(problem, solution_file)

        # Evaluate (compilation happens within evaluate for script-based)
        results = self.evaluate(
            solution_file, work_dir,
            problem=problem, verbose=verbose, save_output=save_output,
            generate_gold_output=generate_gold_output,
            max_workers=max_workers, stop_on_failure=stop_on_failure,
            keep_executables=keep_executables
        )

        if results is None:
            return {"ace": False, "tests_passed": 0, "subtasks": {}, "score": 0, "compile_output": "Error in setup.sh"}, []

        # Interpret results
        score_info = self.interprete_task_result(results, problem.get_subtasks())

        return score_info, results
