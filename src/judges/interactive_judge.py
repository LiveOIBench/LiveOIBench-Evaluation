"""
Interactive judge implementation for interactive problems.
"""
import os
import time
import subprocess
import threading
import math
import queue
import shutil
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


class InteractiveJudge(BaseJudge):
    """
    Judge implementation for interactive problems.

    This judge handles problems where:
    - Solution interacts with an interactor process via named pipes
    - Solution and interactor communicate bidirectionally
    - Correctness is determined by interactor's exit code
    """

    def setup(self, problem: Problem, solution_file: str) -> Tuple[str, str]:
        """
        Set up the environment for interactive judging.

        Creates a working directory and copies solution, interactor, headers, and testlib files.

        Args:
            problem: Problem object
            solution_file: Path to solution source file

        Returns:
            Tuple of (interactor_file_path, work_directory_path)
        """
        interactor_file = problem.get_interactor()
        testlib_file = problem.get_testlib()
        header_file = problem.get_header()

        base_name = os.path.basename(solution_file).replace(".cpp", "")
        work_dir = os.path.join(self.work_path, problem.id, base_name)

        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)

        # Copy solution file
        solution_dest = os.path.join(work_dir, os.path.basename(solution_file))
        os.system(f"cp {solution_file} {solution_dest}")

        # Copy interactor file
        interactor_dest = os.path.join(work_dir, os.path.basename(interactor_file))
        os.system(f"cp {interactor_file} {interactor_dest}")

        # Copy header file if available
        if header_file:
            header_dest = os.path.join(work_dir, os.path.basename(header_file))
            os.system(f"cp {header_file} {header_dest}")

        # Copy testlib.h if available
        if testlib_file:
            testlib_dest = os.path.join(work_dir, "testlib.h")
            os.system(f"cp {testlib_file} {testlib_dest}")

        return interactor_file, work_dir

    def compile(
        self,
        interactor_file: str,
        work_dir: str,
        solution_file: str,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Compile interactor and solution for interactive problem.

        Args:
            interactor_file: Path to interactor source
            work_dir: Working directory
            solution_file: Path to solution source (original path, not copied)
            verbose: Print compilation details

        Returns:
            Tuple of (success, message)
        """
        interactor_executable = os.path.join(work_dir, "interactor")
        solution_executable = os.path.join(work_dir, "solution")

        # Get paths to copied files
        interactor_dest = os.path.join(work_dir, os.path.basename(interactor_file))
        solution_dest = os.path.join(work_dir, os.path.basename(solution_file))

        # Compile interactor
        if verbose:
            print("Compiling interactor...")

        if interactor_file.endswith(".cpp"):
            interactor_compile_cmd = ["g++", "-std=gnu++17", "-O2", "-o", interactor_executable, interactor_dest]
            interactor_process = subprocess.run(interactor_compile_cmd, capture_output=True, text=True)

            if interactor_process.returncode != 0:
                if verbose:
                    print(f"Interactor compilation failed: {interactor_process.stderr}")
                return False, f"Interactor compilation error: {interactor_process.stderr}"
        elif interactor_file.endswith(".py"):
            # Python interactor - just make it executable
            os.system(f"chmod +x {interactor_dest}")
            if verbose:
                print(f"Using Python interactor: {interactor_dest}")

        # Compile solution
        if verbose:
            print("Compiling solution...")

        solution_compile_cmd = ["g++", "-std=gnu++17", "-O2", "-o", solution_executable, solution_dest]
        solution_process = subprocess.run(solution_compile_cmd, capture_output=True, text=True)

        if solution_process.returncode != 0:
            if verbose:
                print(f"Solution compilation failed: {solution_process.stderr}")
            return False, f"Solution compilation error: {solution_process.stderr}"

        return True, "Compilation successful"

    def run_test_case(
        self,
        problem: Problem,
        work_dir: str,
        input_file: str,
        gold_output_file: str,
        cpu_time_limit: float,
        memory_limit_mb: float,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single interactive test case using named pipes.

        Args:
            problem: Problem object
            work_dir: Working directory with compiled binaries
            input_file: Path to input file
            gold_output_file: Path to expected output (often unused for interactive)
            cpu_time_limit: CPU time limit in seconds
            memory_limit_mb: Memory limit in MB
            verbose: Print execution details

        Returns:
            Dictionary with test result including: test_case, correct, cpu_time, memory, exit_code, score
        """
        base_name = os.path.basename(input_file).replace(".in", "")
        dummy_output = os.path.join(work_dir, "dummy.txt")

        cpu_time_limit *= 1.2  # Add 20% buffer
        memory_limit_mb *= 1.2

        # Create named pipes for communication
        fifo_input = os.path.join(work_dir, "fifo_input")   # Judge to solution
        fifo_output = os.path.join(work_dir, "fifo_output")  # Solution to judge

        # Remove pipes if they already exist
        for pipe in [fifo_input, fifo_output]:
            if os.path.exists(pipe):
                os.unlink(pipe)

        # Create the pipes
        os.mkfifo(fifo_input)
        os.mkfifo(fifo_output)

        # Get absolute paths
        abs_fifo_input = os.path.abspath(fifo_input)
        abs_fifo_output = os.path.abspath(fifo_output)
        abs_input_dest = os.path.abspath(input_file)
        abs_dummy_output = os.path.abspath(dummy_output)

        if problem.get_interactor().endswith(".py"):
            abs_judge_path = os.path.join(os.path.abspath(work_dir), "interactor.py")
        else:
            abs_judge_path = os.path.join(os.path.abspath(work_dir), "interactor")

        abs_solution_path = os.path.join(os.path.abspath(work_dir), "solution")

        # Create a PID file to track processes for cleanup
        pid_file = os.path.join(work_dir, "pids.txt")

        monitor_data = {'max_cpu_time': 0.0, 'max_memory': 0.0}
        start_wall_time = time.time()

        fifo_in_fd = None
        fifo_out_fd = None
        interactor_returncode = -1

        try:
            # Open named pipes in non-blocking mode to prevent deadlock
            fifo_in_fd = os.open(abs_fifo_input, os.O_RDWR)
            fifo_out_fd = os.open(abs_fifo_output, os.O_RDWR)

            # Start the interactor process
            if verbose:
                print(f"Starting interactor for test {base_name}...")

            if problem.get_interactor().endswith(".py"):
                interactor_cmd = ["python3", abs_judge_path, abs_input_dest, abs_dummy_output, work_dir]
            else:
                interactor_cmd = [abs_judge_path, abs_input_dest, abs_dummy_output]

            interactor_process = subprocess.Popen(
                interactor_cmd,
                stdin=open(abs_fifo_output, 'r'),
                stdout=open(abs_fifo_input, 'w'),
                stderr=subprocess.PIPE
            )

            time.sleep(0.05)  # Give interactor time to start

            # Start the solution process with resource limits
            if verbose:
                print(f"Starting solution for test {base_name}...")

            solution_process = subprocess.Popen(
                [abs_solution_path],
                stdin=open(abs_fifo_input, 'r'),
                stdout=open(abs_fifo_output, 'w'),
                stderr=subprocess.PIPE,
                preexec_fn=lambda: set_limits(
                    int(math.ceil(cpu_time_limit)),
                    int(math.ceil(memory_limit_mb * 1024 * 1024))
                )
            )

            # Write PIDs to file for potential cleanup
            with open(pid_file, 'w') as f:
                f.write(f"{interactor_process.pid} {solution_process.pid}")

            # Monitor solution process for resources
            try:
                solution_psutil = psutil.Process(solution_process.pid)
                monitor_thread = threading.Thread(
                    target=monitor_process,
                    args=(solution_psutil, cpu_time_limit, monitor_data)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                if verbose:
                    print(f"Warning: Could not monitor solution process: {e}")

            # Wait for both processes with timeout
            try:
                interactor_returncode = interactor_process.wait(timeout=10)
                interactor_stderr = interactor_process.stderr.read()
            except subprocess.TimeoutExpired:
                if verbose:
                    print(f"Timeout expired for test {base_name}. Killing processes...")

                # Kill both processes
                try:
                    solution_process.kill()
                except:
                    pass

                try:
                    interactor_process.kill()
                    interactor_returncode = -9
                    interactor_stderr = interactor_process.stderr.read()
                except:
                    pass

            # Wait for monitor thread to finish
            if 'monitor_thread' in locals() and monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)

            end_wall_time = time.time()
            elapsed_wall_time = end_wall_time - start_wall_time

            # Get resource metrics
            cpu_time = monitor_data['max_cpu_time']
            memory_usage = monitor_data['max_memory']

            # Determine correctness from interactor exit code
            correct = interactor_returncode == 0
            score = 1.0 if correct else 0.0

            if not correct and verbose:
                print(f"Interactor error for {base_name}: {interactor_stderr.decode() if interactor_stderr else 'N/A'}")

            result = {
                "test_case": base_name,
                "wall_time": elapsed_wall_time,
                "cpu_time": cpu_time,
                "memory": memory_usage,
                "exit_code": interactor_returncode,
                "correct": correct,
                "score": score,
                "stderr": interactor_stderr.decode() if interactor_stderr else None,
            }

            if verbose:
                print(f"[{base_name}] Finished in {elapsed_wall_time:.3f}s | "
                      f"CPU Time: {cpu_time:.3f}s | Memory: {memory_usage:.2f} MB | "
                      f"Score: {score} | Correct: {correct}")

            return result

        finally:
            # Close the file descriptors
            for fd in [fifo_in_fd, fifo_out_fd]:
                if fd is not None:
                    try:
                        os.close(fd)
                    except:
                        pass

            # Kill any remaining processes
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        pids = f.read().split()
                        for pid in pids:
                            try:
                                os.kill(int(pid), 9)  # SIGKILL
                            except:
                                pass
                    os.unlink(pid_file)
                except:
                    pass

            # Clean up named pipes
            for pipe in [fifo_input, fifo_output]:
                if os.path.exists(pipe):
                    try:
                        os.unlink(pipe)
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Error cleaning up pipe {pipe}: {str(e)}")

    def evaluate(
        self,
        interactor_file: str,
        work_dir: str,
        solution_file: str,
        verbose: bool = False,
        save_output: bool = False,
        generate_gold_output: bool = False,
        max_workers: int = 10,
        stop_on_failure: bool = False,
        problem: Problem = None,
        keep_executables: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run all test cases for an interactive problem.

        Args:
            interactor_file: Path to interactor source
            work_dir: Working directory (base directory for workers)
            solution_file: Path to solution source
            verbose: Print progress
            save_output: Unused for interactive problems
            generate_gold_output: Unused for interactive problems
            max_workers: Number of parallel workers
            stop_on_failure: Stop subtask on first failure
            problem: Problem object

        Returns:
            List of test case results
        """
        input_files = problem.get_test_inputs()
        output_files = problem.get_test_outputs()

        if len(output_files) == 0:
            # Create dummy output file
            dummy_output = os.path.join(work_dir, "dummy.txt")
            with open(dummy_output, "w") as f:
                f.write("0\n")
            output_files = [dummy_output] * len(input_files)
        else:
            assert len(input_files) == len(output_files), "Input and output files count mismatch"

        cpu_time_limit = problem.get_time_limit()
        memory_limit_mb = problem.get_memory_limit()

        # Create worker directories
        worker_dirs = self._create_worker_directories(work_dir, max_workers, verbose)

        # Thread-safe worker directory allocation
        worker_queue = queue.Queue()
        for i in range(len(worker_dirs)):
            worker_queue.put(i)

        worker_lock = threading.Lock()
        results = []

        def run_test(idx, input_file, output_file):
            # Get an available worker directory
            with worker_lock:
                worker_idx = worker_queue.get()

            worker_dir = worker_dirs[worker_idx]

            try:
                if verbose:
                    print(f"Running interactive test {idx+1}/{len(input_files)}: "
                          f"{os.path.basename(input_file)} (using worker {worker_idx})")

                result = self.run_test_case(
                    problem, worker_dir, input_file, output_file,
                    cpu_time_limit, memory_limit_mb, verbose
                )

                return result
            finally:
                # Return the worker to the pool
                with worker_lock:
                    worker_queue.put(worker_idx)

        try:
            if stop_on_failure:
                # Sequential evaluation with early stopping
                results = self._evaluate_with_early_stopping(
                    problem, worker_dirs, worker_queue, worker_lock,
                    input_files, output_files, cpu_time_limit, memory_limit_mb, verbose
                )
            else:
                # Parallel evaluation
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(run_test, idx, input_file, output_file)
                        for idx, (input_file, output_file) in enumerate(zip(input_files, output_files))
                    ]

                    for future in futures:
                        results.append(future.result())

            if verbose:
                print("\n=== Interactive Evaluation Summary ===")
                for result in results:
                    print(f"Test: {result['test_case']}, Correct: {result['correct']}, "
                          f"Wall Time: {result.get('wall_time', 'N/A'):.3f}s, "
                          f"CPU Time: {result.get('cpu_time', 'N/A'):.3f}s, "
                          f"Memory: {result.get('memory', 'N/A'):.2f} MB")

            return results
        finally:
            if not keep_executables:
                # Clean up all worker directories
                for worker_dir in worker_dirs:
                    shutil.rmtree(worker_dir, ignore_errors=True)
                if verbose:
                    print(f"Cleaned up {len(worker_dirs)} worker directories")

    def _create_worker_directories(self, work_dir: str, max_workers: int, verbose: bool) -> List[str]:
        """
        Create worker directories for parallel execution.

        Each worker gets its own copy of all files to avoid conflicts.

        Args:
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

            # Copy all necessary files from the base work dir to each worker dir
            for file in os.listdir(work_dir):
                file_path = os.path.join(work_dir, file)
                if os.path.isfile(file_path):
                    dest_path = os.path.join(worker_dir, file)
                    shutil.copy2(file_path, dest_path)

            # Make sure executable permissions are preserved
            interactor_path = os.path.join(worker_dir, "interactor")
            solution_path = os.path.join(worker_dir, "solution")

            for exec_path in [interactor_path, solution_path]:
                if os.path.exists(exec_path):
                    os.chmod(exec_path, 0o755)  # rwx r-x r-x

            worker_dirs.append(worker_dir)

        if verbose:
            print(f"Created {len(worker_dirs)} worker directories for parallel evaluation")

        return worker_dirs

    def _evaluate_with_early_stopping(
        self,
        problem: Problem,
        worker_dirs: List[str],
        worker_queue: queue.Queue,
        worker_lock: threading.Lock,
        input_files: List[str],
        output_files: List[str],
        cpu_time_limit: float,
        memory_limit_mb: float,
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """
        Evaluate test cases with early stopping per subtask.

        Args:
            problem: Problem object
            worker_dirs: List of worker directory paths
            worker_queue: Queue of available worker indices
            worker_lock: Lock for worker queue access
            input_files: List of input file paths
            output_files: List of output file paths
            cpu_time_limit: CPU time limit
            memory_limit_mb: Memory limit
            verbose: Print progress

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
                    print(f"Running interactive test case {test_case} from subtask {subtask_id}...")

                # Get worker
                with worker_lock:
                    worker_idx = worker_queue.get()
                worker_dir = worker_dirs[worker_idx]

                try:
                    result = self.run_test_case(
                        problem, worker_dir, input_file, output_file,
                        cpu_time_limit, memory_limit_mb, verbose=verbose
                    )
                    results.append(result)
                    results_dict[test_case] = result
                finally:
                    # Return worker to pool
                    with worker_lock:
                        worker_queue.put(worker_idx)

                if not result['correct']:
                    if verbose:
                        print(f"Interactive test case {test_case} failed. "
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
        Judge a solution for an interactive problem.

        Overrides base implementation to pass problem object to evaluate().

        Args:
            problem: Problem object
            solution_file: Path to solution source file
            verbose: Print progress
            save_output: Unused for interactive
            generate_gold_output: Unused for interactive
            max_workers: Number of parallel workers
            stop_on_failure: Stop subtask on first failure
            keep_executables: Preserve compiled artifacts and worker directories

        Returns:
            Tuple of (score_info, detailed_results)
        """
        if verbose:
            print(f"Judging interactive problem: {problem.id}")

        # Setup
        interactor_file, work_dir = self.setup(problem, solution_file)

        # Compile
        success, message = self.compile(interactor_file, work_dir, solution_file, verbose)
        if not success:
            return {"ace": False, "tests_passed": 0, "subtasks": {}, "score": 0, "compile_output": message}, []

        # Evaluate
        results = self.evaluate(
            interactor_file, work_dir, solution_file,
            verbose=verbose, save_output=save_output, generate_gold_output=generate_gold_output,
            max_workers=max_workers, stop_on_failure=stop_on_failure,
            problem=problem, keep_executables=keep_executables  # Pass problem explicitly
        )

        # Interpret results
        score_info = self.interprete_task_result(results, problem.get_subtasks())

        return score_info, results
