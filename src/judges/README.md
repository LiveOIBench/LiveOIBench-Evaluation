# Judge Architecture

This directory contains the modular judge implementation for evaluating competitive programming solutions.

## Architecture Overview

The judge system uses a **Strategy pattern** with a **Facade** to provide a clean, extensible architecture:

```
┌──────────────────────────────────────┐
│           Judge (Facade)             │
│  Automatically dispatches to the     │
│  appropriate specialized judge       │
└────────┬─────────────────────────────┘
         │
         ├─────────┐
         │         │
         ▼         ▼
┌────────────┐   ┌─────────────────┐
│ BaseJudge  │◄──┤ BatchJudge      │
│ (Abstract) │   │ InteractiveJudge│
└────────────┘   │ ScriptJudge     │
                 └─────────────────┘
```

## Components

### 1. BaseJudge (base_judge.py)
**Abstract base class** defining the interface all judges must implement.

**Key methods:**
- `setup()` - Prepare environment for judging
- `compile()` - Compile solution and auxiliary files
- `run_test_case()` - Execute a single test case
- `evaluate()` - Run all test cases
- `judge()` - Main entry point (setup → compile → evaluate → score)
- `interprete_task_result()` - Calculate scores from results

**Common utilities:**
- `compile_cpp()` - Compile C++ with standard flags
- `compile_checker()` - Compile custom checkers
- `_mark_remaining_tests_as_failed()` - Support for early stopping

### 2. BatchJudge (batch_judge.py)
**For standard I/O problems** where solution reads stdin and writes stdout.

**Features:**
- Standard input/output execution
- Custom checker support (testlib.h-based)
- Fallback to exact/fuzzy output comparison
- Parallel test execution with ThreadPoolExecutor
- Early stopping per subtask (--stop_on_failure)

**Special handling:**
- Competition-specific checker command formats (IOI, EGOI, IATI, etc.)
- Numeric output comparison with tolerance
- Line-by-line comparison with whitespace normalization

### 3. InteractiveJudge (interactive_judge.py)
**For interactive problems** where solution communicates with an interactor.

**Features:**
- Bidirectional communication via named pipes (FIFOs)
- Interactor and solution run as separate processes
- Resource monitoring for solution process only
- Worker pool pattern for parallel execution

**Architecture:**
```
┌──────────┐  fifo_output  ┌────────────┐
│ Solution ├──────────────►│ Interactor │
│          │◄──────────────┤            │
└──────────┘  fifo_input   └────────────┘
```

**Cleanup:**
- PID tracking for process cleanup
- Named pipe cleanup in finally block
- Worker directory management

### 4. ScriptJudge (script_judge.py)
**For script-based problems** with custom evaluation logic.

**Features:**
- `setup.sh` - Run once to prepare environment
- `evaluate.sh` - Run for each test case
- Worker directories for parallel execution
- GNU time integration for resource measurement
- Results file parsing for partial scores

**Workflow:**
1. Create prep directory
2. Run setup.sh once (compile, prepare files)
3. Create worker directories (copy binaries)
4. Run evaluate.sh in parallel workers
5. Parse results from .results.txt files

### 5. Judge (judge.py)
**Main facade class** that automatically dispatches to specialized judges.

**Usage:**
```python
from src.evaluation.judges import Judge, Problem

judge = Judge(evaluation_path="./evaluation")
problem = Problem(...)
score_info, results = judge.judge(problem, solution_file, verbose=True)
```

**Automatic dispatch:**
- `problem.is_interactive_problem()` → InteractiveJudge
- `problem.is_script_based_problem()` → ScriptJudge
- Otherwise → BatchJudge

## Adding a New Judge Type

To add support for a new problem type:

### 1. Create new judge class inheriting from BaseJudge

```python
from .base_judge import BaseJudge
from .problem import Problem

class MyCustomJudge(BaseJudge):
    def setup(self, problem: Problem, solution_file: str):
        # Setup environment
        pass

    def compile(self, *args, **kwargs):
        # Compile solution
        pass

    def run_test_case(self, problem: Problem, *args, **kwargs):
        # Run single test
        pass

    def evaluate(self, problem: Problem, *args, **kwargs):
        # Run all tests
        pass
```

### 2. Add detection method to Problem class

```python
def is_my_custom_problem(self) -> bool:
    # Detection logic
    return self.task_type == "my_custom"
```

### 3. Update Judge facade to dispatch

```python
# In judge.py
def judge(self, problem, solution_file, ...):
    if problem.is_my_custom_problem():
        return self.my_custom_judge.judge(...)
    elif problem.is_interactive_problem():
        ...
```

### 4. Export from __init__.py

```python
from .my_custom_judge import MyCustomJudge

__all__ = [..., 'MyCustomJudge']
```

## Key Design Decisions

### Why Strategy Pattern?
- **Separation of Concerns**: Each judge type has distinct logic
- **Extensibility**: Easy to add new judge types without modifying existing code
- **Testability**: Each judge can be tested independently

### Why Facade Pattern?
- **Backward Compatibility**: Existing code using `Judge` continues to work
- **Simplified API**: Users don't need to know which judge to use
- **Centralized Dispatch**: Problem type detection in one place

### Why Abstract Base Class?
- **Type Safety**: Ensures all judges implement required methods
- **Code Reuse**: Common utilities in BaseJudge
- **Documentation**: Clear interface contract

## Migration Guide

### For Users
**No changes required!** The refactored `Judge` class is backward compatible:

```python
# Old code still works
from src.evaluation.judges.judge import Judge
judge = Judge(evaluation_path)
score, results = judge.judge(problem, solution_file)
```

### For Developers
**Direct access to specialized judges:**

```python
from src.evaluation.judges import BatchJudge, InteractiveJudge, ScriptJudge

# Use specific judge directly
batch_judge = BatchJudge(evaluation_path)
score, results = batch_judge.judge(problem, solution_file)
```

## Testing

Run basic import test:
```bash
python3 -c "from src.evaluation.judges import Judge, BatchJudge, InteractiveJudge, ScriptJudge; print('OK')"
```

Test Judge initialization:
```bash
python3 -c "from src.evaluation.judges import Judge; j = Judge('./evaluation'); print('Judge initialized')"
```

## Performance Considerations

1. **Worker Pool**: Each judge type manages its own worker pool for parallel execution
2. **Early Stopping**: `stop_on_failure` minimizes unnecessary test runs
3. **Resource Monitoring**: Per-process monitoring with psutil
4. **Worker Directories**: Avoids file conflicts in parallel execution

## File Structure

```
src/evaluation/judges/
├── __init__.py            # Module exports
├── README.md              # This file
├── base_judge.py          # Abstract base class
├── batch_judge.py         # Standard I/O judge
├── interactive_judge.py   # Interactive judge
├── script_judge.py        # Script-based judge
├── judge.py               # Main facade
└── problem.py             # Problem metadata class
```

## References

- Original monolithic implementation: 1398 lines
- Refactored modular implementation: ~2000 lines across 6 files
- Complexity: Reduced cyclomatic complexity per file
- Maintainability: Each file has a single responsibility
