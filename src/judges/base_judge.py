"""
Abstract base class for all judge implementations.
Defines the interface that all judges must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import os
import subprocess
import math
from pathlib import Path


class BaseJudge(ABC):
    """
    Abstract base class for judges that evaluate competitive programming solutions.

    All judge implementations must inherit from this class and implement the
    required abstract methods.
    """

    def __init__(self, evaluation_path: str):
        """
        Initialize the judge with evaluation directory paths.

        Args:
            evaluation_path: Base directory for evaluation resources
        """
        self.evaluation_path = evaluation_path
        self.cpp_executable_path = os.path.join(evaluation_path, "executables")
        self.outputs_path = os.path.join(evaluation_path, "outputs")
        self.work_path = os.path.join(evaluation_path, "work")

        # Create necessary directories
        os.makedirs(self.work_path, exist_ok=True)
        os.makedirs(self.cpp_executable_path, exist_ok=True)
        os.makedirs(self.outputs_path, exist_ok=True)

    @abstractmethod
    def setup(self, problem: Any, solution_file: str) -> Tuple[Any, ...]:
        """
        Set up the environment for judging a solution.

        Args:
            problem: Problem object containing metadata and test cases
            solution_file: Path to the solution file

        Returns:
            Tuple containing setup results (implementation-specific)
        """
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Tuple[bool, str]:
        """
        Compile the solution and any necessary auxiliary files.

        Returns:
            Tuple of (success: bool, message: str)
        """
        pass

    @abstractmethod
    def run_test_case(self, problem: Any, *args, **kwargs) -> Dict[str, Any]:
        """
        Run a single test case and return the result.

        Args:
            problem: Problem object
            *args, **kwargs: Implementation-specific arguments

        Returns:
            Dictionary containing test result with keys like:
            - test_case: str
            - correct: bool
            - cpu_time: float
            - memory: float
            - exit_code: int
            - score: float (optional)
        """
        pass

    @abstractmethod
    def evaluate(self, problem: Any, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Run all test cases for a problem.

        Args:
            problem: Problem object
            *args, **kwargs: Implementation-specific arguments

        Returns:
            List of test case results
        """
        pass

    def judge(
        self,
        problem: Any,
        solution_file: str,
        verbose: bool = False,
        save_output: bool = False,
        generate_gold_output: bool = False,
        max_workers: int = 10,
        stop_on_failure: bool = False,
        keep_executables: bool = False
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Main entry point to judge a solution for a problem.

        This method orchestrates the setup, compilation, and evaluation process.
        It can be overridden by subclasses if needed, but provides a sensible
        default implementation.

        Args:
            problem: Problem object containing metadata and test cases
            solution_file: Path to the solution file
            verbose: Whether to print detailed progress information
            save_output: Whether to save solution outputs
            generate_gold_output: Whether to generate gold outputs
            max_workers: Number of parallel workers for test execution
            stop_on_failure: Stop subtask evaluation on first failure
            keep_executables: Preserve compiled artifacts and work directories

        Returns:
            Tuple of (score_info: dict, detailed_results: list)
        """
        # Setup
        setup_result = self.setup(problem, solution_file)

        # Compile
        success, message = self.compile(*setup_result, verbose=verbose)
        if not success:
            return {
                "ace": False,
                "tests_passed": 0,
                "subtasks": {},
                "score": 0,
                "compile_output": message
            }, []

        # Evaluate
        results = self.evaluate(
            problem,
            *setup_result,
            verbose=verbose,
            save_output=save_output,
            generate_gold_output=generate_gold_output,
            max_workers=max_workers,
            stop_on_failure=stop_on_failure,
            keep_executables=keep_executables
        )

        # Interpret results
        score_info = self.interprete_task_result(results, problem.get_subtasks())

        return score_info, results

    def interprete_task_result(self, results: List[Dict[str, Any]], subtasks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret test results and calculate scores for subtasks.

        This is a common implementation that works for most problem types.
        Subclasses can override if they need custom scoring logic.

        Args:
            results: List of test case results
            subtasks: Dictionary of subtask configurations

        Returns:
            Dictionary containing:
            - ace: bool - all tests passed
            - tests_passed: float - fraction of tests passed
            - score: int - total score
            - subtasks: dict - per-subtask scores
        """
        score = {"subtasks": {}}
        results_dict = {result['test_case']: result for result in results}
        test_cases_passed = [result['test_case'] for result in results if result['correct']]

        for i, subtask in subtasks.items():
            subtask_score = 0
            subtask_passed = 0

            for test_case in subtask['testcases']:
                if results_dict[test_case]['correct']:
                    subtask_passed += 1

            # Handle min-score grading
            if 'grading' in subtask and subtask['grading'] == "min-score":
                test_scores = [results_dict[test_case].get('score', 0) for test_case in subtask['testcases']]
                subtask_score = subtask['score'] * min(test_scores)

            # Full score if all tests passed
            if subtask_passed == len(subtask['testcases']):
                subtask_score = subtask['score']

            score['subtasks'][i] = {
                "testcases": len(subtask['testcases']),
                "score": subtask_score,
                "passed": subtask_passed / len(subtask['testcases'])
            }

        score['score'] = round(sum([subtask['score'] for subtask in score['subtasks'].values()]))
        score['tests_passed'] = len(test_cases_passed) / len(results) if results else 0
        score['ace'] = score['tests_passed'] == 1

        return score

    def compile_cpp(
        self,
        solution_file: str,
        grader_file: str,
        executable_path: str,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Compile a C++ solution file (with optional grader).

        This is a common utility method that can be used by multiple judge types.

        Args:
            solution_file: Path to solution source file
            grader_file: Path to grader file (None if not needed)
            executable_path: Output path for executable
            verbose: Print compilation details

        Returns:
            Tuple of (success: bool, output/error message: str)
        """
        if grader_file:
            compile_command = [
                "g++", "-std=gnu++17", "-Wall", "-O2", "-pipe", "-static", "-g",
                "-o", executable_path, grader_file, solution_file
            ]
        else:
            compile_command = [
                "g++", "-std=gnu++17", "-Wall", "-O2", "-pipe", "-static", "-g",
                "-o", executable_path, solution_file
            ]

        if verbose:
            print("Compiling with command:", " ".join(compile_command))

        try:
            result = subprocess.run(
                compile_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            if verbose:
                print("Compilation succeeded.")
                if result.stdout:
                    print("Compiler output:", result.stdout)
            return (True, result.stdout)
        except subprocess.CalledProcessError as e:
            print("Compilation failed!")
            print("Compiler output:", e.stdout)
            print("Compiler errors:", e.stderr)
            return (False, e.stderr)

    def compile_checker(
        self,
        checker_source: str,
        executable_path: str,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Compile a checker source file.

        Args:
            checker_source: Path to checker source file
            executable_path: Output path for checker executable
            verbose: Print compilation details

        Returns:
            Tuple of (success: bool, output/error message: str)
        """
        compile_command = [
            "g++", "-std=gnu++17", "-Wall", "-O2", "-pipe", "-static", "-g",
            "-o", executable_path, checker_source
        ]

        if verbose:
            print("Compiling checker with command:", " ".join(compile_command))

        try:
            result = subprocess.run(
                compile_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if verbose:
                print("Checker compilation succeeded.")
                if result.stdout:
                    print("Checker compiler output:", result.stdout)
            return (True, result.stdout)
        except subprocess.CalledProcessError as e:
            print("Checker compilation failed!")
            print("Checker compiler output:", e.stdout)
            print("Checker compiler errors:", e.stderr)
            return (False, e.stderr)

    def _mark_remaining_tests_as_failed(
        self,
        subtask: Dict[str, Any],
        subtask_id: str,
        already_run: set,
        results: List[Dict[str, Any]],
        results_dict: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Mark all remaining tests in a subtask as failed due to a previous failure.

        This is used when stop_on_failure is enabled.

        Args:
            subtask: Subtask configuration dictionary
            subtask_id: Identifier of the subtask
            already_run: Set of test cases already run
            results: List to append skipped results to
            results_dict: Dictionary to store skipped results by test case name
        """
        for remaining_case in subtask['testcases']:
            if remaining_case not in already_run:
                already_run.add(remaining_case)
                skipped_result = {
                    "test_case": remaining_case,
                    "wall_time": 0,
                    "cpu_time": 0,
                    "memory": 0,
                    "exit_code": -1,
                    "error": f"Skipped due to previous test failure in subtask {subtask_id}",
                    "correct": False,
                    "score": 0
                }
                results.append(skipped_result)
                results_dict[remaining_case] = skipped_result
