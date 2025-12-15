# problem.py
import json, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
DEFAULT_TESTLIB = os.environ.get("TESTLIB_PATH", str(REPO_ROOT / "testlib.h"))

class Problem:
    def __init__(self, problem_dir, task, year, competition, round_name, split, parse_dir=None):
        self.task = task
        self.year = year
        self.competition = competition
        self.round_name = round_name
        self.split = split
        self.id = competition + "-" + year + "-" + round_name + "-" + task
        self.test_dir = os.path.join(problem_dir, "tests")
        self.grader_dir = os.path.join(problem_dir, "graders")
        self.solution_dir = os.path.join(problem_dir, "solutions")
        self.attachments_dir = os.path.join(problem_dir, "attachments")
        self.statements_dir = os.path.join(problem_dir, "statements")
        self.parse_dir = parse_dir
        self.checker = os.path.join(problem_dir, "checkers")
        # Load subtasks when available (IOI-Bench-Restructured only)
        subtasks_path = os.path.join(problem_dir, "subtasks.json")
        with open(subtasks_path) as f:
            self.subtasks = json.load(f)
        #Load problem config
        with open(os.path.join(problem_dir, "problem.json")) as f:
            self.problem_config = json.load(f)
        self.memory_limit = self.problem_config.get("memory_limit", 1024 * 1024 * 1024)  # default 1024 MB
        self.time_limit = self.problem_config.get("time_limit", 1.0 * 1024 * 1024)      # default 1s
        self.task_type = self.problem_config.get("task_type", "None")  # default normal
    
    def get_test_inputs(self):
        test_files = os.listdir(self.test_dir)
        test_files = [f for f in test_files if f.endswith(".in")]
        test_files.sort()
        test_files = [os.path.join(self.test_dir, f) for f in test_files]
        return test_files
    def get_test_outputs(self):
        test_files = os.listdir(self.test_dir)
        test_files = [f for f in test_files if f.endswith(".out")]
        test_files.sort()
        test_files = [os.path.join(self.test_dir, f) for f in test_files]
        return test_files
    def get_grader(self):
        grader = os.path.join(self.grader_dir, "grader.cpp")
        if "code" in self.problem_config:
            task = self.problem_config["code"]
        else:
            task = self.task
        header = os.path.join(self.grader_dir, f"{task}.h")
        if not os.path.exists(header):
            header = os.path.join(self.attachments_dir, f"{task}.h")
        if not os.path.exists(header):
            if  os.path.exists(self.grader_dir):
                header_files = [os.path.join(self.grader_dir, f) for f in os.listdir(self.grader_dir) if f.endswith(".h")]
                if len(header_files) > 0:
                    header = header_files[0]
        if not os.path.exists(grader):
            grader = os.path.join(self.attachments_dir, "grader.cpp")
        if os.path.exists(grader) and os.path.exists(header):
            print(f"Grader and header files found for {self.id}.")
            return grader, header
        else:
            return None, None
    def get_code_solution(self, lang="cpp", solution_type= None):
        def list_solution_files(directory):
            # List and sort all solution files with the specified extension in the given directory
            files = [f for f in os.listdir(directory) if f.endswith(f".{lang}")]
            files.sort()
            return [os.path.join(directory, f) for f in files]
    
        code_dir = os.path.join(self.solution_dir, "codes")
        if solution_type == 'incorrect':
            for sub in ("incorrect", "wrong"):
                sub_dir = os.path.join(code_dir, sub)
                if os.path.exists(sub_dir):
                    return list_solution_files(sub_dir)
        elif solution_type == 'time_limit':
            for sub in ("time_limit", "TLE"):
                sub_dir = os.path.join(code_dir, sub)
                if os.path.exists(sub_dir):
                    return list_solution_files(sub_dir)
        elif solution_type == 'memory_limit':
            for sub in ("time_limit_and_runtime_error", "runtime_error", "memory_limit", "MLE"):
                sub_dir = os.path.join(code_dir, sub)
                if os.path.exists(sub_dir):
                    return list_solution_files(sub_dir)
        # Check preferred subdirectories in order
        for sub in ("model_solution", "correct", "accepted"):
            sub_dir = os.path.join(code_dir, sub)
            if os.path.exists(sub_dir):
                return list_solution_files(sub_dir)
            if solution_type == "correct" or solution_type == "model_solution":
                return []
        
        # Fallback to the main codes directory
        return list_solution_files(code_dir)
    def get_edoitorial(self, convert_type="gemini-2.0-flash"):
        editorial = os.path.join(self.solution_dir, "editorial.md")
        if not os.path.exists(editorial):
            editorial = os.path.join(self.solution_dir, "editorial.txt")
        # if not os.path.exists(editorial):
        #     editorial = os.path.join(self.solution_dir, "editorial.tex")
        if not os.path.exists(editorial):
            editorial = os.path.join(self.parse_dir, "solutions", f"editorial_{convert_type}.md")
        if not os.path.exists(editorial):
            editorial = os.path.join(self.solution_dir, "editorial.pdf")
        if not os.path.exists(editorial):
            print(f"Warning: {editorial} not found")
            return None
        print(f"Editorial file found: {editorial}")
        return editorial
    def get_time_limit(self):
        return self.time_limit
    def get_memory_limit(self):
        return self.memory_limit
    def get_task_type(self):
        return self.task_type
    def get_subtasks(self):
        return self.subtasks
    def get_statement(self, converter_type="gemini-2.0-flash"):
        statement = os.path.join(self.statements_dir, "statement.md")
        if not os.path.exists(statement):
            statement = os.path.join(self.statements_dir, "statement.txt")
        if not os.path.exists(statement):
            statement = os.path.join(self.parse_dir, "statements", f"statement_{converter_type}.md")
        if not os.path.exists(statement):
            statement = os.path.join(self.statements_dir, "statement.tex")
        if not os.path.exists(statement):
            print(f"Warning: {statement} not found")
            return None
        print(f"Statement file found: {statement}")
        return statement
    def get_checker(self):
        if not os.path.exists(self.checker):
            return None, None
        files = os.listdir(self.checker)
        header_files = [file for file in files if file.endswith(".h")]
        if len(header_files) > 0:
            header_file = [os.path.join(self.checker, file) for file in header_files]
        else:
            header_file = [DEFAULT_TESTLIB]
        checker_files = [file for file in files if file.endswith(".cpp")]
        if "checker.cpp" in checker_files:
            checker_file = os.path.join(self.checker, "checker.cpp")
        elif "validator.cpp" in checker_files:
            checker_file = os.path.join(self.checker, "validator.cpp")
        elif len(checker_files) > 0:
            checker_file = os.path.join(self.checker, checker_files[0])
        else:
            checker_file = None
        return checker_file, header_file
    def get_total_points(self):
        total_points = 0
        for idx, subtask in self.subtasks.items():
            total_points += int(subtask["score"])
        return total_points
    def get_prompt(self):
        #create prompt
        # if os.path.exists(os.path.join(self.parse_dir, "prompt.txt")):
        #     with open(os.path.join(self.parse_dir, "prompt.txt"), 'r') as f:
        #         return f.read()
        task = self.task
        prompt = f"Given a competition problem below, write a solution in C++ that solves all the subtasks. Make sure to wrap your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
        statement = self.get_statement()
        with open(statement, "r") as f:
            prompt += "[BEGIN PROBLEM]\n" + f.read() + "[END PROBLEM]\n"
        grader = os.path.join(self.attachments_dir, "grader.cpp")
        if not os.path.exists(grader):
            grader = os.path.join(self.attachments_dir, "stub.cpp")
        header = os.path.join(self.attachments_dir, f"{self.task}.h")
        starter = os.path.join(self.attachments_dir, f"{self.task}.cpp")
        prompt += f"Time limit: {self.time_limit} seconds\n"
        prompt += f"Memory limit: {self.memory_limit} MB\n"
        if os.path.exists(grader) and os.path.exists(header) and os.path.exists(starter):
            prompt +=f"We are going to grade your solution using the following grader.cpp file and {task}.h file. You can write your solution by modifying the {task}.cpp and wrap your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
            with open(grader, "r") as f:
                prompt += "```grader.cpp\n" + f.read() + "```\n\n"
            with open(header, "r") as f:
                prompt += f"```{task}.h\n" + f.read() + "```\n\n"
            with open(starter, "r") as f:
                prompt += f"```{task}.cpp\n" + f.read() + "```\n\n"
            print(f"Grader and header files found for {self.id}.")
        else:
            prompt += f"Generate a solution in C++ that solves the task. Make sure to wrap your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
        # with open(os.path.join(self.parse_dir, "prompt.txt"), 'w') as f:
        #     f.write(prompt)
        return prompt

    def get_manager(self):
        """Get the manager.cpp file for interactive problems."""
        manager_path = os.path.join(self.grader_dir, "manager.cpp")
        return manager_path if os.path.exists(manager_path) else None

    def get_stub(self):
        """Get the stub.cpp file for interactive problems."""
        stub_path = os.path.join(self.grader_dir, "stub.cpp")
        return stub_path if os.path.exists(stub_path) else None
    
    def get_interactor(self):
        """Get the interactor.cpp file for interactive problems."""
        interactor_path = os.path.join(self.grader_dir, "interactor.cpp")
        if not os.path.exists(interactor_path):
            interactor_path = os.path.join(self.grader_dir, "interactor.py")
        return interactor_path if os.path.exists(interactor_path) else None
    def get_header(self):
        """Get the header file for interactive problems."""
        if "code" in self.problem_config:
            task = self.problem_config["code"]
        else:
            task = self.task.lower()
        header = os.path.join(self.grader_dir, f"{task}.h")
        if not os.path.exists(header):
            header = os.path.join(self.grader_dir, f"stub.h")
        if not os.path.exists(header):
            header = os.path.join(self.attachments_dir, f"{task}.h")
        if not os.path.exists(header):
            header = os.path.join(self.grader_dir, f"validate.h")
        if not os.path.exists(header):
            header_files = [os.path.join(self.grader_dir, f) for f in os.listdir(self.grader_dir) if f.endswith(".h")]
            if len(header_files) > 0:
                header = header_files[0]
        return header if os.path.exists(header) else None
    
    def get_testlib(self):
        """Get the testlib.h file for interactive problems."""
        testlib_path = os.path.join(self.grader_dir, "testlib.h")
        if not os.path.exists(testlib_path):
            testlib_path = DEFAULT_TESTLIB
        return testlib_path
    def get_evaluation_script(self):
        """Get the evaluation script path if it exists."""
        script_path = os.path.join(self.grader_dir, "evaluate.sh")
        if os.path.exists(script_path):
            return script_path
        return None
    def get_setup_script(self):
        """Get the setup script path if it exists."""
        script_path = os.path.join(self.grader_dir, "setup.sh")
        if os.path.exists(script_path):
            return script_path
        return None
    def is_script_based_problem(self):
        """Check if the problem has an evaluate.sh script."""
        script_path = os.path.join(self.grader_dir, "evaluate.sh")
        return os.path.exists(script_path)
    def is_interactive_problem(self):
        """Check if the problem is interactive based on the presence of specific files."""
        manager_path = os.path.join(self.grader_dir, "interactor.cpp")
        return os.path.exists(manager_path) or os.path.exists(os.path.join(self.grader_dir, "interactor.py"))
