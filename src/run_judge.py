import argparse
import hashlib
import json
import os
import sys
import shutil
import tempfile
import threading
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from judges.judge import Judge
from judges.problem import Problem
from judges.result_type import ResultType


global_counter = 0
counter_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_rmtree(path: str) -> None:
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_judge_artifacts(evaluation_dir: str, keep_executables: bool, save_output: bool) -> None:
    """Remove working artifacts when not requested to keep them."""
    executables_dir = os.path.join(evaluation_dir, "executables")
    work_dir = os.path.join(evaluation_dir, "work")
    outputs_dir = os.path.join(evaluation_dir, "outputs")

    if not keep_executables:
        _safe_rmtree(executables_dir)
        _safe_rmtree(work_dir)
    if not save_output:
        _safe_rmtree(outputs_dir)


def normalize_solution_types(solution_types: Iterable[str]) -> List[str]:
    return ["llm" if st.lower() in ("llm", "llm_solutions") else st for st in solution_types]


def expand_years(years: Iterable[str], competitions: Iterable[str], data_dir: str) -> List[str]:
    raw_years = list(years)
    if any(y.lower() == "all" for y in raw_years):
        year_set = set()
        for comp in competitions:
            comp_dir = os.path.join(data_dir, comp)
            if not os.path.isdir(comp_dir):
                continue
            for entry in os.listdir(comp_dir):
                if entry.isdigit() and os.path.isdir(os.path.join(comp_dir, entry)):
                    year_set.add(entry)
        return sorted(year_set)

    expanded: List[str] = []
    for year in raw_years:
        if "-" in year:
            try:
                start, end = map(int, year.split("-"))
            except ValueError:
                expanded.append(year)
                continue
            expanded.extend(str(y) for y in range(start, end + 1))
        else:
            expanded.append(year)
    return expanded


def resolve_llm_models(llm_models: Optional[Iterable[str]], solutions_dir: Optional[str]) -> Optional[List[str]]:
    if llm_models is None:
        return None
    models = list(llm_models)
    if "all" not in {m.lower() for m in models}:
        return models
    if not solutions_dir:
        raise ValueError("--llm_solutions_dir is required when using 'all' for --llm_models")
    try:
        return [
            name
            for name in os.listdir(solutions_dir)
            if os.path.isdir(os.path.join(solutions_dir, name))
        ]
    except FileNotFoundError as exc:
        raise ValueError(f"LLM solutions directory not found: {solutions_dir}") from exc


def determine_result_type(score_info: Mapping[str, Any], detailed_results: List[Mapping[str, Any]]) -> ResultType:
    if "compile_output" in score_info:
        return ResultType.COMPILATION_ERROR
    if "exception" in score_info:
        return ResultType.UNKNOWN
    if score_info.get("ace", False):
        return ResultType.ACCEPTED
    if detailed_results:
        if any(r.get("exit_code", 0) == -9 for r in detailed_results):
            return ResultType.TIME_LIMIT_EXCEEDED
        if any(r.get("memory_limit_exceeded", False) or r.get("exit_code", 0) == -6 for r in detailed_results):
            return ResultType.MEMORY_LIMIT_EXCEEDED
        if any(r.get("exit_code", 0) != 0 for r in detailed_results):
            return ResultType.RUNTIME_ERROR
    return ResultType.WRONG_ANSWER


def discover_problems(args: argparse.Namespace) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    count = 0
    for comp in args.competitions:
        for year in args.years:
            rounds = list(args.rounds) if args.rounds else []
            base_year = os.path.join(args.data_dir, comp, year)
            if not args.rounds:
                try:
                    rounds = os.listdir(base_year)
                except FileNotFoundError:
                    continue
            for rnd in rounds:
                base = os.path.join(base_year, rnd)
                meta_file = os.path.join(base, "meta_info.json")
                if not os.path.exists(meta_file):
                    continue
                try:
                    meta = json.load(open(meta_file))
                except json.JSONDecodeError:
                    print(f"Warning: failed to parse {meta_file}, skipping", file=sys.stderr)
                    continue
                for split, tasks in meta.items():
                    for task in tasks:
                        if args.tasks and task not in args.tasks:
                            continue
                        task_dir = os.path.join(base, task)
                        if not os.path.isdir(task_dir):
                            continue
                        cfg_file = os.path.join(task_dir, "problem.json")
                        task_type = "batch"
                        if os.path.exists(cfg_file):
                            try:
                                cfg = json.load(open(cfg_file))
                                task_type = cfg.get("task_type") or "batch"
                            except json.JSONDecodeError:
                                pass
                        if args.task_types and task_type.lower() not in args.task_types:
                            continue
                        problems.append(
                            {
                                "competition": comp,
                                "year": year,
                                "round": rnd,
                                "split": split,
                                "task": task,
                                "dir": task_dir,
                                "id": f"{comp}-{year}-{rnd}-{task}",
                            }
                        )
                        count += 1
    print(f"Found {count} tasks.")
    return problems


SUPPORTED_EXTENSIONS = {".cpp", ".py", ".java"}

# Global cache for JSON solutions: model -> {problem_id: {filename: code}}
_JSON_SOLUTIONS_CACHE: Dict[str, Dict[str, Dict[str, str]]] = {}


def load_llm_json_solutions(json_dir: str, models: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load LLM solutions from JSON files into cache.

    JSON file structure: {problem_id: {filename: code_content}}
    Example: {"APIO-2023-contest-abc": {"abc_model_0.cpp": "code..."}}
    """
    global _JSON_SOLUTIONS_CACHE

    for model in models:
        if model in _JSON_SOLUTIONS_CACHE:
            continue

        json_file = os.path.join(json_dir, model, f"{model}_code.json")
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found for model {model}: {json_file}", file=sys.stderr)
            _JSON_SOLUTIONS_CACHE[model] = {}
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            _JSON_SOLUTIONS_CACHE[model] = data
            print(f"Loaded {len(data)} problems from {json_file}")
        except (json.JSONDecodeError, IOError) as exc:
            print(f"Warning: Failed to load JSON for model {model}: {exc}", file=sys.stderr)
            _JSON_SOLUTIONS_CACHE[model] = {}

    return _JSON_SOLUTIONS_CACHE


def get_solution_files_from_json(
    problem_info: Mapping[str, str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Get solution files from JSON cache for a specific problem.

    Returns list of dicts with: model, name, code (in-memory content).
    """
    solutions: List[Dict[str, Any]] = []
    problem_id = problem_info["id"]

    if not args.llm_models:
        raise ValueError("--llm_models is required when evaluating LLM solutions")

    for model in args.llm_models:
        model_data = _JSON_SOLUTIONS_CACHE.get(model, {})
        problem_solutions = model_data.get(problem_id, {})

        for filename, code_content in problem_solutions.items():
            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            seed = _extract_seed(filename)
            if seed >= args.max_solutions:
                continue
            solutions.append({
                "model": model,
                "name": filename,
                "code": code_content,  # In-memory code content
            })

    return solutions


def _extract_seed(filename: str, default: int = 0) -> int:
    """Extract trailing integer seed from filenames like name_<seed>.<ext>."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if not parts:
        return default
    try:
        return int(parts[-1])
    except ValueError:
        return default


def get_solution_files(problem_info: Mapping[str, str], solution_type: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Get solution files for a problem.

    For LLM solutions, supports two modes:
    - JSON mode: --llm_json_dir (solutions stored in {model}/{model}_code.json)
    - Directory mode: --llm_solutions_dir (solutions in directory tree)
    """
    prob = Problem(
        problem_info["dir"],
        problem_info["task"],
        problem_info["year"],
        problem_info["competition"],
        problem_info["round"],
        problem_info["split"],
    )
    solutions: List[Dict[str, Any]] = []

    if solution_type == "llm":
        if not args.llm_models:
            raise ValueError("--llm_models is required when evaluating LLM solutions")

        # Use JSON mode if llm_json_dir is specified
        if getattr(args, "llm_json_dir", None):
            return get_solution_files_from_json(problem_info, args)

        # Otherwise use directory mode
        if not getattr(args, "llm_solutions_dir", None):
            raise ValueError("--llm_solutions_dir or --llm_json_dir is required when evaluating LLM solutions")

        for model in args.llm_models:
            model_dir = os.path.join(
                args.llm_solutions_dir,
                model,
                problem_info["competition"],
                problem_info["year"],
                problem_info["round"],
                problem_info["task"],
                "codes",
            )
            if not os.path.isdir(model_dir):
                continue
            for filename in os.listdir(model_dir):
                if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                seed = _extract_seed(filename)
                if seed >= args.max_solutions:
                    continue
                solutions.append(
                    {
                        "path": os.path.join(model_dir, filename),
                        "model": model,
                        "name": filename,
                    }
                )
        return solutions

    for lang in ("cpp", "py", "java"):
        for path in prob.get_code_solution(lang, solution_type):
            solutions.append({"path": path, "model": "original", "name": os.path.basename(path)})
            if len(solutions) >= args.max_solutions:
                return solutions[: args.max_solutions]

    return solutions[: args.max_solutions]


def get_cache_key(problem_id: str, solution_path_or_code: str, is_code: bool = False) -> str:
    """Generate cache key from problem ID and solution content.

    Args:
        problem_id: The problem identifier
        solution_path_or_code: Either a file path or code content string
        is_code: If True, solution_path_or_code is code content, not a path
    """
    if is_code:
        digest = hashlib.md5(solution_path_or_code.encode("utf-8")).hexdigest()
    else:
        with open(solution_path_or_code, "rb") as handle:
            digest = hashlib.md5(handle.read()).hexdigest()
    return f"{problem_id}_{digest}"


def get_cached_result(cache_key: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def save_to_cache(cache_key: str, result: Mapping[str, Any], cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"{cache_key}.json"), "w") as handle:
        json.dump(result, handle)


def evaluate_solution(
    judge: Judge,
    problem_info: Mapping[str, Any],
    solution_info: Mapping[str, Any],
    args: argparse.Namespace,
    total: int,
) -> Dict[str, Any]:
    """Evaluate a single solution.

    Supports both file-based solutions (with 'path') and in-memory solutions (with 'code').
    """
    global global_counter
    with counter_lock:
        global_counter += 1
        counter_value = global_counter

    pid = problem_info["id"]
    model_name = solution_info["model"]
    sol_filename = solution_info["name"]

    print(
        f"[{counter_value}/{total}] {problem_info['competition']} {problem_info['year']} "
        f"{problem_info['round']} | Task: {problem_info['task']} | Solution: {sol_filename} | Model: {model_name}"
    )

    # Handle in-memory code vs file path
    is_in_memory = "code" in solution_info
    if is_in_memory:
        cache_key = get_cache_key(pid, solution_info["code"], is_code=True)
    else:
        cache_key = get_cache_key(pid, solution_info["path"], is_code=False)

    if args.use_cache and not args.reeval:
        cached = get_cached_result(cache_key, args.cache_dir)
        if cached:
            print(
                f"  → Result: {cached['status']} | Score: {cached['score']} | "
                f"Tests: {cached['tests_passed'] * 100:.2f}% (cached)"
            )
            return cached

    temp_file_path = None
    try:
        # If in-memory code, write to temp file
        if is_in_memory:
            suffix = Path(sol_filename).suffix or ".cpp"
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
                tmp.write(solution_info["code"])
                temp_file_path = tmp.name
            solution_path = temp_file_path
        else:
            solution_path = solution_info["path"]

        problem = Problem(
            problem_info["dir"],
            problem_info["task"],
            problem_info["year"],
            problem_info["competition"],
            problem_info["round"],
            problem_info["split"],
        )
        score_info, details = judge.judge(
            problem,
            solution_path,
            verbose=args.verbose,
            save_output=args.save_output,
            generate_gold_output=False,
            max_workers=args.workers,
            stop_on_failure=args.stop_on_failure,
            keep_executables=args.keep_executables,
        )
        times = [r.get("cpu_time", 0) for r in details if r.get("cpu_time")]
        mems = [r.get("memory", 0) for r in details if r.get("memory")]
        max_time = max(times) if times else 0
        max_mem = max(mems) if mems else 0
        status = determine_result_type(score_info, details)
        result = {
            "problem_id": pid,
            "solution_file": sol_filename,
            "model": model_name,
            "status": status.name,
            "status_code": int(status),
            "score": score_info.get("score", 0),
            "tests_passed": score_info.get("tests_passed", 0),
            "execution_time": max_time,
            "memory_usage": max_mem,
            "subtasks": score_info.get("subtasks", {}),
            "details": details,
        }
        if "compile_output" in score_info:
            result["compile_output"] = score_info["compile_output"]
        save_to_cache(cache_key, result, args.cache_dir)
        print(
            f"  → Result: {result['status']} | Score: {result['score']} | "
            f"Tests: {result['tests_passed'] * 100:.2f}%"
        )
        return result
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error evaluating {sol_filename}: {exc}", file=sys.stderr)
        fallback = {
            "problem_id": pid,
            "solution_file": sol_filename,
            "model": model_name,
            "status": ResultType.UNKNOWN.name,
            "status_code": int(ResultType.UNKNOWN),
            "score": 0,
            "tests_passed": 0,
            "execution_time": 0,
            "memory_usage": 0,
            "subtasks": {},
        }
        print("  → Result: UNKNOWN | Score: 0 | Tests: 0.00%")
        return fallback
    finally:
        # Clean up temp file if created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


def print_problem_summaries(results: Mapping[str, List[Mapping[str, Any]]], ps_map: Mapping[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("PROBLEM SUMMARIES")
    print("=" * 80)

    for pid, problem_results in sorted(results.items()):
        problem_info = ps_map[pid]["problem_info"]
        print(
            f"\n== Problem: {problem_info['task']} ({problem_info['competition']} "
            f"{problem_info['year']} {problem_info['round']}) =="
        )
        model_results: Dict[str, List[Mapping[str, Any]]] = {}
        for res in problem_results:
            model_results.setdefault(res["model"], []).append(res)

        print(f"Total solutions evaluated: {len(problem_results)}")
        for model, solutions in sorted(model_results.items()):
            best = sorted(
                solutions,
                key=lambda s: (
                    -s["score"],
                    -s["tests_passed"],
                    s["execution_time"],
                ),
            )[0]
            print(f"  Model: {model}")
            print(f"    Best solution: {best['solution_file']}")
            print(f"    Status: {best['status']}")
            print(f"    Score: {best['score']}")
            print(f"    Tests passed: {best['tests_passed'] * 100:.2f}%")
            print(f"    Time: {best['execution_time']:.3f}s | Memory: {best['memory_usage']} KB")
            if len(solutions) > 1:
                status_counts: Dict[str, int] = {}
                for sol in solutions:
                    status_counts[sol["status"]] = status_counts.get(sol["status"], 0) + 1
                joined = ", ".join(f"{status}: {count}" for status, count in sorted(status_counts.items()))
                print(f"    Solution statuses: {joined}")
    print("\n" + "=" * 80)


def print_evaluation_summary(results: Mapping[str, List[Mapping[str, Any]]], llm_models: List[str]) -> None:
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    model_results: Dict[str, List[Mapping[str, Any]]] = {}
    for solution_results in results.values():
        for result in solution_results:
            model_results.setdefault(result["model"], []).append(result)

    for model, items in sorted(model_results.items()):
        if model not in llm_models:
            continue
        status_counts: Dict[str, int] = {}
        total_score = 0
        total_tests = 0.0
        max_score = 0
        for res in items:
            status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
            total_score += res["score"]
            total_tests += res["tests_passed"]
            max_score += 100
        print(f"\n== Model: {model} ==")
        print(f"Solutions evaluated: {len(items)}")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count} ({count / len(items) * 100:.1f}%)")
        pct_score = total_score / max_score * 100 if max_score else 0
        print(f"Total score: {total_score} / {max_score} ({pct_score:.2f}%)")
        print(f"Average tests passed: {total_tests / len(items) * 100:.2f}%")
        accepted_rate = status_counts.get("ACCEPTED", 0) / len(items) * 100 if items else 0
        print(f"Solution acceptance rate: {accepted_rate:.2f}%")
    print("\n" + "=" * 80)


def _result_ranking_key(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -result.get("score", 0),
        -result.get("tests_passed", 0.0),
        result.get("execution_time", float("inf")),
        result.get("memory_usage", float("inf")),
        result.get("status", ""),
    )


def write_per_model_result_files(
    results: Mapping[str, List[Mapping[str, Any]]],
    ps_map: Mapping[str, Any],
    args: argparse.Namespace,
    metadata_args: Mapping[str, Any],
    generated_at: datetime,
    duration_seconds: float,
    timestamp: str,
    llm_models: List[str],
):
    per_model_submissions: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    problem_max_scores: Dict[str, int] = {}

    for pid, recs in results.items():
        if not recs:
            continue
        problem_info = ps_map.get(pid, {}).get("problem_info", {})
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for rec in recs:
            grouped[rec["model"]].append(rec)
        for model, entries in grouped.items():
            if pid not in problem_max_scores:
                try:
                    prob = Problem(
                        problem_info["dir"],
                        problem_info["task"],
                        problem_info["year"],
                        problem_info["competition"],
                        problem_info["round"],
                        problem_info["split"],
                    )
                    problem_max_scores[pid] = prob.get_total_points()
                except Exception:  # pylint: disable=broad-except
                    problem_max_scores[pid] = 0
            max_score = problem_max_scores.get(pid, 0)

            for entry in entries:
                score = entry.get("score", 0) or 0
                tests_passed = entry.get("tests_passed", 0.0) or 0.0
                relative_score = (score / max_score * 100) if max_score else 0.0
                tests_passed_pct = tests_passed * 100
                execution_time = entry.get("execution_time", 0.0) or 0.0
                memory_usage = entry.get("memory_usage", 0) or 0
                raw_subtasks = entry.get("subtasks") or {}
                subtask_scores = {
                    str(key): (val.get("score", 0) if isinstance(val, dict) else val)
                    for key, val in raw_subtasks.items()
                }
                trimmed_entry = {
                    "problem": pid,
                    "solution": entry.get("solution_file"),
                    "status": entry.get("status"),
                    "score": score,
                    "relative_score": round(relative_score, 2),
                    "tests_passed_pct": round(tests_passed_pct, 2),
                    "time": execution_time,
                    "memory": memory_usage,
                    "subtasks": subtask_scores,
                }
                trimmed_entry["model"] = model
                solution_name = trimmed_entry["solution"]
                if not solution_name:
                    continue
                key = f"{pid}+{solution_name}"
                problem_bucket = per_model_submissions[model].setdefault(pid, {})
                existing = problem_bucket.get(solution_name)
                if existing is None:
                    problem_bucket[solution_name] = trimmed_entry
                # elif existing != trimmed_entry:
                #     print(key)
                #     continue
                #     raise ValueError(f"Conflicting results for submission key '{key}'")
    models = sorted(per_model_submissions.keys())
    for model in models:
        if model not in llm_models:
            continue
        problem_map = per_model_submissions.get(model, {})
        submissions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for problem in sorted(problem_map.keys()):
            solution_map = problem_map[problem]
            submissions[problem] = {
                solution: solution_map[solution]
                for solution in sorted(solution_map.keys())
            }
        if len(models) == 1 and args.results_file:
            model_path = args.results_file
        else:
            # Write to submission_results/<model>/<model>_<timestamp>.json
            model_dir = os.path.join(args.output_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model}_{timestamp}.json")
        with open(model_path, "w") as handle:
            json.dump(submissions, handle, indent=2)
        print(f"Wrote submissions map to {model_path}")


# ---------------------------------------------------------------------------
# Batch workflow
# ---------------------------------------------------------------------------


def run_batch(args: argparse.Namespace) -> None:
    global global_counter
    global_counter = 0
    args.solution_types = normalize_solution_types(args.solution_types)
    args.years = expand_years(args.years, args.competitions, args.data_dir)

    # Resolve LLM models - can use either llm_solutions_dir or llm_json_dir
    llm_source_dir = getattr(args, "llm_json_dir", None) or getattr(args, "llm_solutions_dir", None)
    args.llm_models = resolve_llm_models(args.llm_models, llm_source_dir)

    # Validate LLM solution source is provided when evaluating LLM solutions
    if "llm" in args.solution_types:
        if not getattr(args, "llm_solutions_dir", None) and not getattr(args, "llm_json_dir", None):
            raise ValueError("--llm_solutions_dir or --llm_json_dir is required when evaluating llm solutions")

    # Load JSON solutions if using JSON mode
    if getattr(args, "llm_json_dir", None) and args.llm_models:
        load_llm_json_solutions(args.llm_json_dir, args.llm_models)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    judge = Judge(args.evaluation_dir)
    start = time.time()
    print(f"=== Start at {datetime.now()} ===")
    problems = discover_problems(args)

    all_solutions: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    ps_map: Dict[str, Dict[str, Any]] = {}
    total = 0

    for problem in problems:
        solutions: List[Dict[str, Any]] = []
        for solution_type in args.solution_types:
            solutions.extend(get_solution_files(problem, solution_type, args))
        ps_map[problem["id"]] = {"problem_info": problem, "solutions": solutions}
        if not solutions:
            print(f"Warning: No solutions found for {problem['id']}")
            continue
        for sol in solutions:
            # Handle both file-based and in-memory solutions
            if "code" in sol:
                # In-memory solution from JSON - check if code is non-empty
                if not sol["code"] or not sol["code"].strip():
                    continue
            else:
                # File-based solution - check file exists and is non-empty
                try:
                    if os.path.getsize(sol["path"]) == 0:
                        continue
                except OSError:
                    continue
            all_solutions.append((problem, sol))
            total += 1

    print(f"Total non-empty solutions to evaluate: {total}")
    results: Dict[str, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            (problem_info["id"], executor.submit(evaluate_solution, judge, problem_info, sol_info, args, total))
            for problem_info, sol_info in all_solutions
        ]
        for pid, future in futures:
            res = future.result()
            results.setdefault(pid, []).append(res)

    print_problem_summaries(results, ps_map)
    print_evaluation_summary(results, args.llm_models)

    duration = time.time() - start
    generated_at = datetime.utcnow()
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    metadata_args = {k: v for k, v in vars(args).items() if k not in {"handler"}}
    write_per_model_result_files(results, ps_map, args, metadata_args, generated_at, duration, timestamp, args.llm_models)
    print(f"=== Done in {duration:.2f}s ===")
    cleanup_judge_artifacts(args.evaluation_dir, args.keep_executables, args.save_output)


# ---------------------------------------------------------------------------
# Single-solution workflow
# ---------------------------------------------------------------------------


def run_single(args: argparse.Namespace) -> None:
    problem_dir = os.path.join(
        args.problem_folder,
        args.competition,
        args.year,
        args.round,
        args.task,
    )
    problem = Problem(problem_dir, args.task, args.year, args.competition, args.round, args.split)
    judge = Judge(args.evaluation_folder)
    score_info, details = judge.judge(
        problem,
        args.solution_file,
        verbose=args.verbose,
        save_output=args.save_output,
        generate_gold_output=args.generate_gold_output,
        max_workers=args.max_workers,
        stop_on_failure=args.stop_on_failure,
        keep_executables=args.keep_executables,
    )
    status = determine_result_type(score_info, details)
    print(f"Result: {status.name}")
    print(json.dumps(score_info, indent=2))
    cleanup_judge_artifacts(args.evaluation_folder, args.keep_executables, args.save_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge solutions for IOI-style benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("batch", help="Evaluate many solutions and persist JSON results.")
    batch.add_argument("--competitions", nargs="+", default=["IOI"], help="Competitions to evaluate")
    batch.add_argument("--years", nargs="+", default=["2024"], help="Years or ranges (e.g. 2023-2025 or all)")
    batch.add_argument("--rounds", nargs="+", default=None, help="Specific rounds to include")
    batch.add_argument("--tasks", nargs="+", default=None, help="Filter to specific tasks")
    batch.add_argument("--task_types", nargs="+", default=None, help="Filter by task types")
    batch.add_argument("--solution_types", nargs="+", required=True, help="Solution categories to evaluate")
    batch.add_argument("--llm_models", nargs="+", default=None, help="LLM model names (or 'all')")
    batch.add_argument("--llm_solutions_dir", type=str, help="Directory containing generated LLM solutions (directory mode)")
    batch.add_argument("--llm_json_dir", type=str, help="Directory containing LLM solution JSON files (JSON mode: {model}/{model}_code.json)")
    batch.add_argument("--workers", type=int, default=6, help="Thread pool size")
    batch.add_argument("--verbose", action="store_true", help="Verbose judging output")
    batch.add_argument("--stop_on_failure", action="store_true", help="Stop evaluation on first failing test")
    batch.add_argument("--save_output", action="store_true", help="Persist judge outputs under evaluation resources directory")
    batch.add_argument("--keep_executables", action="store_true", help="Preserve compiled executables and work folders after judging")
    batch.add_argument("--work_dir", type=str, default=None, help="Custom working directory for judge runs (defaults to a temp dir under evaluation_dir).")
    batch.add_argument("--data_dir", type=str, required=True, help="Benchmark data root")
    batch.add_argument("--evaluation_dir", type=str, required=True, help="Evaluation resources root")
    batch.add_argument("--output_dir", type=str, required=True, help="Directory for evaluation artifacts")
    batch.add_argument("--cache_dir", type=str, required=True, help="Cache directory for per-solution results")
    batch.add_argument("--max_solutions", type=int, default=1, help="Maximum solutions per task/model")
    batch.add_argument("--use_cache", dest="use_cache", action="store_true", default=True, help="Reuse cached results when available")
    batch.add_argument("--no-cache", dest="use_cache", action="store_false", help="Ignore cached results")
    batch.add_argument("--reeval", action="store_true", help="Force re-evaluation even if cached")
    batch.add_argument("--results_file", type=str, help="Path for the JSON results file")
    batch.set_defaults(handler=run_batch)

    single = subparsers.add_parser("single", help="Evaluate a single solution file.")
    single.add_argument("--competition", type=str, default="IOI")
    single.add_argument("--year", type=str, default="2024")
    single.add_argument("--round", type=str, default="contest")
    single.add_argument("--split", type=str, default="contest")
    single.add_argument("--task", type=str, required=True)
    single.add_argument("--solution_file", type=str, required=True)
    single.add_argument("--problem_folder", type=str, required=True)
    single.add_argument("--evaluation_folder", type=str, required=True)
    single.add_argument("--stop_on_failure", action="store_true")
    single.add_argument("--verbose", action="store_true")
    single.add_argument("--save_output", action="store_true")
    single.add_argument("--generate_gold_output", action="store_true")
    single.add_argument("--max_workers", type=int, default=4)
    single.add_argument("--keep_executables", action="store_true")
    single.add_argument("--work_dir", type=str, default=None, help="Custom working directory for judge runs (defaults to a temp dir under evaluation_folder).")
    single.set_defaults(handler=run_single)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
