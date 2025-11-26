"""
Batch judge implementation for standard input/output problems.
"""
import os
import time
import subprocess
import threading
import math
import re
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any
from pathlib import Path

from .base_judge import BaseJudge
from .problem import Problem


def set_limits(cpu_time_limit, memory_limit_bytes):
    """Apply CPU and memory limits before execution."""
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def monitor_process(proc, time_limit, monitor_data, interval=0.01):
    """Periodically checks and updates the CPU time and memory usage of the process."""
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


class BatchJudge(BaseJudge):
    """
    Judge implementation for batch (standard I/O) problems.

    This judge handles traditional competitive programming problems where:
    - Solution reads from stdin and writes to stdout
    - Correctness is checked by comparing output with expected output
    - Custom checkers (testlib.h-based) are supported
    """

    def setup(self, problem: Problem, solution_file: str) -> Tuple[str, str, str, str, str, str]:
        """
        Set up the environment for batch judging.

        Creates a working directory, copies solution, grader, headers, and checker files.

        Args:
            problem: Problem object
            solution_file: Path to solution source file

        Returns:
            Tuple of (solution_file, grader_file, solution_folder, executable_path,
                     checker_file, checker_executable_path)
        """
        grader, header = problem.get_grader()
        language = self._detect_language(solution_file)

        base_name = Path(solution_file).stem
        if language == "java":
            maybe_class = self._extract_java_public_class(solution_file)
            if maybe_class:
                base_name = maybe_class

        solution_folder = os.path.join(self.cpp_executable_path, problem.id, base_name)

        if not os.path.exists(solution_folder):
            os.makedirs(solution_folder, exist_ok=True)

        testlib = problem.get_testlib()

        # Copy grader and header if available
        if grader is not None:
            os.system(f"cp {grader} {header} {testlib} {solution_folder}")
            new_grader = os.path.join(solution_folder, os.path.basename(grader))
        else:
            new_grader = None

        # Copy solution file
        os.system(f"cp {solution_file} {solution_folder}")
        new_solution_file = os.path.join(solution_folder, os.path.basename(solution_file))
        language = self._detect_language(new_solution_file)

        if language == "java":
            class_name = self._extract_java_public_class(new_solution_file)
            if class_name and class_name != Path(new_solution_file).stem:
                renamed_path = os.path.join(solution_folder, f"{class_name}.java")
                os.rename(new_solution_file, renamed_path)
                new_solution_file = renamed_path
                base_name = class_name
        if language == "cpp":
            executable_path = os.path.join(solution_folder, base_name)
        elif language == "java":
            executable_path = os.path.join(solution_folder, f"{base_name}.class")
        else:
            executable_path = new_solution_file

        # Get checker from the problem
        checker, checker_headers = problem.get_checker()
        if checker is not None:
            os.system(f"cp {checker} {solution_folder}")
            for header in checker_headers:
                os.system(f"cp {header} {solution_folder}")
            new_checker = os.path.join(solution_folder, os.path.basename(checker))
            checker_executable_path = os.path.join(
                solution_folder,
                os.path.basename(checker).replace(".cpp", "")
            )
        else:
            new_checker = None
            checker_executable_path = None

        return new_solution_file, new_grader, solution_folder, executable_path, new_checker, checker_executable_path

    def compile(
        self,
        solution_file: str,
        grader_file: str,
        solution_folder: str,
        executable_path: str,
        checker_file: str,
        checker_executable_path: str,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Compile solution and checker.

        Args:
            solution_file: Path to solution source
            grader_file: Path to grader (None if not needed)
            solution_folder: Working directory
            executable_path: Output path for solution executable
            checker_file: Path to checker source (None if not needed)
            checker_executable_path: Output path for checker executable
            verbose: Print compilation details

        Returns:
            Tuple of (success, message)
        """
        language = self._detect_language(solution_file)

        if language == "cpp":
            status, output = self.compile_cpp(solution_file, grader_file, executable_path, verbose)
        elif language == "python":
            status, output = True, "Python does not require compilation"
        elif language == "java":
            status, output = self._compile_java(solution_file, solution_folder, verbose)
        else:
            return (False, f"Unsupported language: {language}")

        if not status:
            return False, output

        # Compile checker if available
        if checker_file is not None:
            status, output = self.compile_checker(checker_file, checker_executable_path, verbose)
            if not status:
                return (False, output)

        return (True, "Compilation successful")

    def _compile_java(self, solution_file: str, solution_folder: str, verbose: bool = False) -> Tuple[bool, str]:
        """Compile a Java solution file to bytecode."""
        compile_command = ["javac", "-d", solution_folder, solution_file]

        if verbose:
            print("Compiling Java with command:", " ".join(compile_command))

        try:
            result = subprocess.run(
                compile_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            if verbose and result.stdout:
                print("Java compiler output:", result.stdout)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            if verbose:
                print("Java compilation failed!")
                print("Compiler output:", e.stdout)
                print("Compiler errors:", e.stderr)
            return False, e.stderr

    @staticmethod
    def _detect_language(solution_file: str) -> str:
        """Infer language from solution file extension."""
        suffix = Path(solution_file).suffix.lower()
        if suffix == ".cpp":
            return "cpp"
        if suffix == ".py":
            return "python"
        if suffix in (".java", ".class"):
            return "java"
        return "cpp"

    @staticmethod
    def _extract_java_public_class(solution_file: str) -> str:
        """Extract the public class name from a Java source file, if present."""
        try:
            with open(solution_file, "r") as src:
                content = src.read()
        except OSError:
            return None
        match = re.search(r"public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", content)
        if match:
            return match.group(1)
        return None

    def _build_execution_command(self, language: str, executable_path: str, memory_limit_mb: float = None) -> List[str]:
        """Build the execution command for the given language."""
        if language == "python":
            return ["python3", executable_path]
        if language == "java":
            class_dir = os.path.dirname(executable_path) or "."
            class_name = Path(executable_path).stem
            java_cmd = ["java"]
            if memory_limit_mb:
                # Keep JVM memory below RLIMIT_AS to avoid startup failures.
                heap_size = max(64, int(memory_limit_mb * 0.6))
                java_cmd.extend(["-Xms64m", f"-Xmx{heap_size}m", "-XX:ReservedCodeCacheSize=64m"])
            java_cmd.extend(["-cp", class_dir, class_name])
            return java_cmd
        return [executable_path]

    def run_test_case(
        self,
        problem: Problem,
        cpp_executable: str,
        input_file: str,
        gold_output_file: str,
        cpu_time_limit: float,
        memory_limit_mb: float,
        verbose: bool = False,
        save_output: bool = False,
        generate_gold_output: bool = False,
        checker_executable: str = None,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Run a single test case and check correctness.

        Args:
            problem: Problem object
            cpp_executable: Path to compiled solution
            input_file: Path to input file
            gold_output_file: Path to expected output file
            cpu_time_limit: CPU time limit in seconds
            memory_limit_mb: Memory limit in MB
            verbose: Print execution details
            save_output: Save solution output to file
            generate_gold_output: Generate gold output (for testing)
            checker_executable: Path to custom checker (None for default comparison)

        Returns:
            Dictionary with test result including: test_case, correct, cpu_time, memory, exit_code, etc.
        """
        base_name = os.path.basename(input_file).replace(".in", "")
        solution_name = Path(cpp_executable).stem
        output_file = os.path.join(self.evaluation_path, "outputs", problem.id, solution_name, f"{base_name}.out")

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        start_wall_time = time.time()
        cpu_time_limit *= 1.2  # Add 20% buffer
        memory_limit_mb *= 1.2

        # Read input data
        with open(input_file, "r") as f:
            input_data = f.read()

        language = language or self._detect_language(cpp_executable)
        execution_command = self._build_execution_command(language, cpp_executable, memory_limit_mb)

        # Launch the solution with resource limits
        preexec = None
        if language != "java":
            preexec = lambda: set_limits(
                int(math.ceil(cpu_time_limit)),
                int(math.ceil(memory_limit_mb * 1024 * 1024))
            )

        process = subprocess.Popen(
            execution_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec
        )

        # Monitor process resources
        proc = psutil.Process(process.pid)
        monitor_data = {'max_cpu_time': 0.0, 'max_memory': 0.0}
        monitor_thread = threading.Thread(target=monitor_process, args=(proc, cpu_time_limit, monitor_data))
        monitor_thread.start()

        try:
            stdout_data, stderr_data = process.communicate(input=input_data.encode(), timeout=120)
        except subprocess.TimeoutExpired:
            print("Timeout reached. Killing process...")
            process.kill()
            stdout_data, stderr_data = process.communicate()

        monitor_thread.join()
        end_wall_time = time.time()
        elapsed_wall_time = end_wall_time - start_wall_time

        cpu_time = monitor_data['max_cpu_time']
        memory_usage = monitor_data['max_memory']

        result = {
            "test_case": base_name,
            "wall_time": elapsed_wall_time,
            "cpu_time": cpu_time,
            "memory": memory_usage,
            "exit_code": process.returncode,
            "error": stderr_data.decode() if stderr_data else None
        }

        # Check correctness
        if result['exit_code'] == 0:
            stdout_decoded = stdout_data.decode() if stdout_data else ""
            stdout_decoded = stdout_decoded.strip()

            if checker_executable is not None and not generate_gold_output:
                # Use custom checker
                result['correct'], result['score'] = self._run_checker(
                    problem, checker_executable, input_file, gold_output_file,
                    output_file, stdout_decoded, base_name
                )
            else:
                # Direct output comparison
                result['correct'] = self._compare_outputs(gold_output_file, stdout_decoded)

            # Save output if required
            if save_output:
                with open(output_file, "w") as f:
                    f.write(stdout_decoded)
            if generate_gold_output:
                with open(gold_output_file, "w") as f:
                    f.write(stdout_decoded)
        else:
            result['correct'] = False

        if verbose:
            print(f"[{base_name}] Finished in {elapsed_wall_time:.3f}s | "
                  f"CPU Time: {cpu_time:.3f}s | Memory: {memory_usage:.2f} MB | "
                  f"Exit Code: {process.returncode}")

        return result

    def _run_checker(
        self,
        problem: Problem,
        checker_executable: str,
        input_file: str,
        gold_output_file: str,
        output_file: str,
        stdout_decoded: str,
        base_name: str
    ) -> Tuple[bool, float]:
        """
        Run custom checker to validate output.

        Args:
            problem: Problem object
            checker_executable: Path to checker executable
            input_file: Path to input file
            gold_output_file: Path to expected output
            output_file: Path to save solution output
            stdout_decoded: Solution's stdout
            base_name: Test case name

        Returns:
            Tuple of (is_correct, score)
        """
        # Write output to file
        with open(output_file, "w") as f:
            f.write(stdout_decoded)

        # Determine checker command based on competition format
        if "IOI" in problem.competition:
            checker_command = [checker_executable, input_file, gold_output_file, output_file]
            checker_proc = subprocess.Popen(checker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elif "EGOI" in problem.competition and (int(problem.year) == 2024 or int(problem.year) == 2023):
            checker_command = [checker_executable, input_file, gold_output_file, os.path.dirname(checker_executable)]
            temp_output = os.path.join(os.path.dirname(checker_executable), f"{base_name}.out")
            with open(temp_output, "w") as f:
                f.write(stdout_decoded)
            checker_proc = subprocess.Popen(
                checker_command,
                stdin=open(temp_output, "r"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        elif "IATI" in problem.competition:
            checker_command = [checker_executable, input_file, gold_output_file, output_file]
            checker_proc = subprocess.Popen(checker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            checker_command = [checker_executable, input_file, output_file, gold_output_file]
            checker_proc = subprocess.Popen(checker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            out_checker, err_checker = checker_proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            print("Checker process timed out. Killing it...")
            checker_proc.kill()
            out_checker = "Time Limit Exceeded"
            err_checker = "Time Limit Exceeded"
            checker_proc.returncode = -9

        combined_output = (out_checker.strip() + " " + err_checker.strip()).strip()

        # Parse checker output
        is_correct = None
        if re.search(r'(OK|Correct|Accepted|^YES$|^AC$|^100$|^1\.0$|^1$|Passed)', combined_output, re.IGNORECASE):
            is_correct = True
        elif re.search(r'(WA|Wrong|Incorrect|^NO$|^0$|^0\.0$|Failed|Error)', combined_output, re.IGNORECASE):
            is_correct = False

        if is_correct is None:
            is_correct = checker_proc.returncode == 0

        score = 1.0 if is_correct else 0.0

        # Check for score.txt (partial scoring)
        score_file = os.path.join(os.path.dirname(checker_executable), "score.txt")
        if os.path.exists(score_file):
            with open(score_file, "r") as f:
                score_content = f.read()
            with open(score_file, "w") as f:
                pass  # Clear the file
            if score_content:
                try:
                    score = float(score_content)
                    if score >= 1.0:
                        score = 1.0
                        is_correct = True
                    else:
                        is_correct = False
                except ValueError:
                    score = 0.0

        # Handle special case scoring
        if problem.id == "RMI-2023-contest-To_be_xor_not_to_be":
            m = re.search(r"Score ratio: ([0-9.]+)", err_checker)
            if m:
                score = float(m.group(1))
                is_correct = score == 1.0
            else:
                score = 0.0

        if problem.competition == "IATI":
            score = float(out_checker)
            is_correct = score == 1.0

        return is_correct, score

    def _compare_outputs(self, gold_output_file: str, stdout_decoded: str) -> bool:
        """
        Compare solution output with expected output.

        Handles various comparison modes:
        - Exact string match
        - Fuzzy numeric comparison
        - Line-by-line comparison

        Args:
            gold_output_file: Path to expected output
            stdout_decoded: Solution's output

        Returns:
            True if outputs match, False otherwise
        """
        with open(gold_output_file, "r") as f:
            gold_output = f.read()

        gold_output = gold_output.strip()
        stdout_decoded = stdout_decoded.strip()

        # Check if both outputs are single numbers - if so, use fuzzy comparison
        if self._is_single_number(gold_output) and self._is_single_number(stdout_decoded):
            return self._compare_numbers(gold_output, stdout_decoded)

        # Exact match
        if gold_output == stdout_decoded:
            return True

        # Line-by-line comparison (ignoring trailing whitespace)
        gold_lines = gold_output.splitlines()
        output_lines = stdout_decoded.splitlines()

        if len(gold_lines) != len(output_lines):
            return False

        for i in range(len(gold_lines)):
            if gold_lines[i].strip() != output_lines[i].strip():
                return False

        return True

    @staticmethod
    def _is_single_number(text: str) -> bool:
        """Check if text contains exactly one number."""
        number_re = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
        lines = text.splitlines()

        # Remove trailing blank lines
        while lines and lines[-1].strip() == '':
            lines.pop()

        if len(lines) != 1:
            return False

        return bool(number_re.fullmatch(lines[0].strip()))

    @staticmethod
    def _compare_numbers(gold: str, output: str) -> bool:
        """Compare two numeric outputs with tolerance."""
        m_gold = re.match(r'^[+\-]?(\d+(\.\d*)?|\.\d+)([eE][+\-]?\d+)?', gold.strip())
        m_out = re.match(r'^[+\-]?(\d+(\.\d*)?|\.\d+)([eE][+\-]?\d+)?', output.strip())

        if m_gold and m_out:
            gold_value = float(m_gold.group(0))
            output_value = float(m_out.group(0))
            return math.isclose(gold_value, output_value, rel_tol=1e-6, abs_tol=1e-6)

        return False

    def evaluate(
        self,
        problem: Problem,
        solution_file: str,
        grader_file: str,
        solution_folder: str,
        executable_path: str,
        checker_file: str,
        checker_executable_path: str,
        verbose: bool = False,
        save_output: bool = False,
        generate_gold_output: bool = False,
        max_workers: int = 10,
        stop_on_failure: bool = False,
        keep_executables: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run all test cases for a problem.

        Args:
            problem: Problem object
            solution_file: Path to solution source
            grader_file: Path to grader
            solution_folder: Working directory
            executable_path: Path to solution executable
            checker_file: Path to checker source
            checker_executable_path: Path to checker executable
            verbose: Print progress
            save_output: Save outputs
            generate_gold_output: Generate gold outputs
            max_workers: Number of parallel workers
            stop_on_failure: Stop subtask on first failure

        Returns:
            List of test case results
        """
        input_files = problem.get_test_inputs()
        output_files = problem.get_test_outputs()

        # Handle special cases
        if problem.id not in ["IOI-2018-contest-combo", "EGOI-2023-contest-guessinggame"]:
            assert len(input_files) == len(output_files), "Input and output files count mismatch"
        else:
            output_files = input_files

        cpu_time_limit = problem.get_time_limit()
        memory_limit_mb = problem.get_memory_limit()
        language = self._detect_language(solution_file)
        results = []

        if stop_on_failure:
            # Sequential evaluation with early stopping per subtask
            results = self._evaluate_with_early_stopping(
                problem, executable_path, checker_executable_path,
                input_files, output_files, cpu_time_limit, memory_limit_mb,
                verbose, save_output, generate_gold_output, language
            )
        else:
            # Parallel evaluation
            # Special case: some problems need serial execution
            if problem.competition == "EGOI" and problem.task == "makethemmeet":
                max_workers = 1

            test_cases = zip(input_files, output_files)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(
                    lambda f: self.run_test_case(
                        problem, executable_path, f[0], f[1],
                        cpu_time_limit, memory_limit_mb,
                        verbose, save_output, generate_gold_output, checker_executable_path,
                        language=language
                    ),
                    test_cases
                ))

        # Cleanup
        if not keep_executables:
            os.system(f"rm -rf {solution_folder}")

        if verbose:
            print("\n=== Batch Evaluation Summary ===")
            for result in results:
                print(f"Test: {result['test_case']}, Correct: {result['correct']}, "
                      f"Wall Time: {result['wall_time']:.3f}s, "
                      f"CPU Time: {result['cpu_time']:.3f}s, "
                      f"Exit Code: {result['exit_code']}, Error: {result.get('error')}")

        return results

    def _evaluate_with_early_stopping(
        self,
        problem: Problem,
        executable_path: str,
        checker_executable_path: str,
        input_files: List[str],
        output_files: List[str],
        cpu_time_limit: float,
        memory_limit_mb: float,
        verbose: bool,
        save_output: bool,
        generate_gold_output: bool,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate test cases with early stopping per subtask.

        When a test fails in a subtask, remaining tests in that subtask are skipped.

        Args:
            problem: Problem object
            executable_path: Path to solution executable
            checker_executable_path: Path to checker executable
            input_files: List of input file paths
            output_files: List of output file paths
            cpu_time_limit: CPU time limit
            memory_limit_mb: Memory limit
            verbose: Print progress
            save_output: Save outputs
            generate_gold_output: Generate gold outputs

        Returns:
            List of test case results
        """
        subtasks = problem.get_subtasks()
        test_case_map = {}
        for input_file, output_file in zip(input_files, output_files):
            base_name = os.path.basename(input_file).replace(".in", "")
            test_case_map[base_name] = (input_file, output_file)

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
                        print(f"Warning: Test case {test_case} not found in input/output files")
                    continue

                input_file, output_file = test_case_map[test_case]

                if verbose:
                    print(f"Running test case {test_case} from subtask {subtask_id}...")

                result = self.run_test_case(
                    problem, executable_path, input_file, output_file,
                    cpu_time_limit, memory_limit_mb, verbose,
                    save_output, generate_gold_output, checker_executable_path,
                    language=language
                )
                results.append(result)
                results_dict[test_case] = result

                if not result['correct']:
                    if verbose:
                        print(f"Test case {test_case} failed. Skipping remaining tests in subtask {subtask_id}.")
                    self._mark_remaining_tests_as_failed(subtask, subtask_id, already_run, results, results_dict)
                    break

        return results
