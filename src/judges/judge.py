"""
Main judge facade that dispatches to specialized judge implementations.

This module provides a unified Judge interface that automatically selects
the appropriate judge implementation (Batch, Interactive, or Script) based
on the problem type.
"""
from typing import Dict, List, Tuple, Any
from .problem import Problem
from .batch_judge import BatchJudge
from .interactive_judge import InteractiveJudge
from .script_judge import ScriptJudge


class Judge:
    """
    Main judge class that acts as a facade/factory for specialized judges.

    This class automatically determines the problem type and delegates to the
    appropriate specialized judge implementation:
    - BatchJudge: for standard I/O problems
    - InteractiveJudge: for interactive problems
    - ScriptJudge: for script-based problems
    """

    def __init__(self, evaluation_path: str):
        """
        Initialize the judge with evaluation directory paths.

        Creates instances of all specialized judges.

        Args:
            evaluation_path: Base directory for evaluation resources
        """
        self.evaluation_path = evaluation_path

        # Create specialized judge instances
        self.batch_judge = BatchJudge(evaluation_path)
        self.interactive_judge = InteractiveJudge(evaluation_path)
        self.script_judge = ScriptJudge(evaluation_path)

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
        Judge a solution for a problem.

        Automatically determines the problem type and dispatches to the
        appropriate specialized judge.

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
                score_info contains:
                - ace: bool - all tests passed
                - tests_passed: float - fraction of tests passed
                - score: int - total score
                - subtasks: dict - per-subtask scores
                detailed_results is a list of per-test-case results
        """
        # Determine problem type and dispatch to appropriate judge
        if problem.is_interactive_problem():
            if verbose:
                print(f"Using InteractiveJudge for problem {problem.id}")
            return self.interactive_judge.judge(
                problem, solution_file, verbose, save_output,
                generate_gold_output, max_workers, stop_on_failure,
                keep_executables=keep_executables
            )
        elif problem.is_script_based_problem():
            if verbose:
                print(f"Using ScriptJudge for problem {problem.id}")
            return self.script_judge.judge(
                problem, solution_file, verbose, save_output,
                generate_gold_output, max_workers, stop_on_failure,
                keep_executables=keep_executables
            )
        else:
            if verbose:
                print(f"Using BatchJudge for problem {problem.id}")
            return self.batch_judge.judge(
                problem, solution_file, verbose, save_output,
                generate_gold_output, max_workers, stop_on_failure,
                keep_executables=keep_executables
            )

    # Expose common utility methods from batch_judge for backward compatibility
    def compile_cpp(self, solution_file: str, grader_file: str, executable_path: str, verbose: bool = False):
        """Compile a C++ solution. Delegates to BatchJudge."""
        return self.batch_judge.compile_cpp(solution_file, grader_file, executable_path, verbose)

    def compile_checker(self, checker_source: str, executable_path: str, verbose: bool = False):
        """Compile a checker. Delegates to BatchJudge."""
        return self.batch_judge.compile_checker(checker_source, executable_path, verbose)


# For backward compatibility, keep the helper functions at module level
def set_limits(cpu_time_limit, memory_limit_bytes):
    """Apply CPU and memory limits before execution."""
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def monitor_process(proc, time_limit, monitor_data, interval=0.01):
    """Periodically checks and updates the CPU time and memory usage of the process."""
    import psutil
    import time

    while proc.is_running():
        try:
            cpu_time = proc.cpu_times().user + proc.cpu_times().system
            mem_usage = proc.memory_info().rss / (1024 * 1024)  # in MB
            monitor_data['max_cpu_time'] = max(monitor_data['max_cpu_time'], cpu_time)
            monitor_data['max_memory'] = max(monitor_data['max_memory'], mem_usage)
            if cpu_time > time_limit:
                proc.kill()
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(interval)


def remove_first_line(str):
    """Remove first line from a string."""
    return str[str.find('\n')+1:]


if __name__ == "__main__":
    print("Use the CLI entry points under src/evaluation or the helper scripts to run evaluations.")
