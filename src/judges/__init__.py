"""
Judge module for evaluating competitive programming solutions.

This module provides a modular architecture for judging different types of problems:
- BatchJudge: Standard I/O problems
- InteractiveJudge: Interactive problems with interactors
- ScriptJudge: Script-based problems with custom evaluation logic

The main Judge class acts as a facade that automatically dispatches to the
appropriate specialized judge based on problem type.
"""

from .judge import Judge
from .base_judge import BaseJudge
from .batch_judge import BatchJudge
from .interactive_judge import InteractiveJudge
from .script_judge import ScriptJudge
from .problem import Problem
from .result_type import ResultType

__all__ = [
    'Judge',
    'BaseJudge',
    'BatchJudge',
    'InteractiveJudge',
    'ScriptJudge',
    'Problem',
    'ResultType'
]
