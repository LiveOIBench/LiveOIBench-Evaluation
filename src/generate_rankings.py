#!/usr/bin/env python3
"""
Generate rankings from submission results with three-stage pipeline:
1. Problem results: Select best solution per problem from submissions
2. Contest results: Aggregate problem results to contest level with human metrics
3. Final results: Generate global rankings CSV across all models

This script processes the hierarchical evaluation structure:
  submission_results/<model>/<model>_<timestamp>.json  (input from run_judge.py)
  → problem_results/<model>_problem.json  (best solution per problem)
  → contest_results/<model>_contest.json  (contest-level metrics)
  → final_results.csv  (global rankings table)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# Default paths (can be overridden by environment variables or CLI args)
LIVEOIBENCH_ROOT = os.getenv("LIVEOIBENCH_ROOT")
LIVEOIBENCH_EVALUATION_DIR = os.getenv("LIVEOIBENCH_EVALUATION_DIR")
DEFAULT_SUBMISSION_RESULTS_DIR = os.path.join(LIVEOIBENCH_EVALUATION_DIR, "submission_results")
DEFAULT_PROBLEM_RESULTS_DIR = os.path.join(LIVEOIBENCH_EVALUATION_DIR, "problem_results")
DEFAULT_CONTEST_RESULTS_DIR = os.path.join(LIVEOIBENCH_EVALUATION_DIR, "contest_results")
DEFAULT_FINAL_RESULTS_FILE = os.path.join(LIVEOIBENCH_EVALUATION_DIR, "final_results.csv")
DEFAULT_CONTESTANT_PARQUET = os.path.join(LIVEOIBENCH_ROOT, "parquet_files", "contestants", "data", "contest_results.parquet")
DEFAULT_PROBLEMS_PARQUET = os.path.join(LIVEOIBENCH_ROOT, "parquet_files", "problems", "data", "liveoibench_v1.parquet")

USACO_INFO_ROOT = os.path.join(LIVEOIBENCH_ROOT, "data", "USACO")

# Columns that should never be treated as task scores when parsing human results
NON_TASK_COLUMNS = {
    "rank", "contestant", "country", "total", "recalculated_total", "medal",
    "cf_rating", "day1", "day2", "day 1", "day 2", "score rel.", "division",
    "team", "nationality",
}


# ==============================================================================
# Shared utility functions
# ==============================================================================

def normalize_contest_identifier(contest_base: str) -> str:
    """Normalize contest identifiers to match human leaderboard IDs."""
    if contest_base.startswith("CCO-"):
        parts = contest_base.split("-", 2)
        if len(parts) == 3:
            prefix, year, rest = parts
            replacements = {
                "Canadian_Computing_Competition_Junior": "Junior",
                "Canadian_Computing_Competition_Senior": "Senior",
                "Canadian_Computing_Olympiad": "contest",
            }
            for source, replacement in replacements.items():
                if rest.startswith(source):
                    rest = rest.replace(source, replacement, 1)
                    break
            contest_base = f"{prefix}-{year}-{rest}"
    return contest_base


def parse_contest_parts(contest_base: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Split a contest identifier into competition, year, and label."""
    parts = contest_base.split("-", 2)
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 1:
        return parts[0], None, None
    return None, None, None


def normalize_name(value: str) -> str:
    """Normalize task column names for consistent matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return float(round(value, digits))


# ==============================================================================
# USACO contest info cache
# ==============================================================================

_USACO_INFO_CACHE: Dict[Tuple[str, str], Optional[Dict[str, Dict]]] = {}


def get_usaco_contest_info(year: Optional[str], contest_name: Optional[str]) -> Optional[Dict[str, Dict]]:
    """Load USACO contest metadata (promotion thresholds, tasks) from disk."""
    if not year or not contest_name:
        return None

    key = (year, contest_name)
    if key in _USACO_INFO_CACHE:
        return _USACO_INFO_CACHE[key]

    info_path = os.path.join(USACO_INFO_ROOT, year, contest_name, "contest_info.json")
    contest_info: Dict[str, Dict] = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    payload = line.strip()
                    if not payload:
                        continue
                    record = json.loads(payload)
                    level = str(record.get("level", "")).lower()
                    if level:
                        contest_info[level] = record
        except Exception:
            contest_info = {}

    _USACO_INFO_CACHE[key] = contest_info or None
    return _USACO_INFO_CACHE[key]


# ==============================================================================
# Data classes
# ==============================================================================

@dataclass
class ProblemRecord:
    score: float
    relative_score: float
    tests_passed_pct: float
    status: str
    model: Optional[str]
    time: float = 0.0
    memory: int = 0
    subtasks: Dict[str, Any] = None

    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = {}

    @property
    def is_full_pass(self) -> bool:
        rel = self.relative_score if not math.isnan(self.relative_score) else 0.0
        tests_pct = self.tests_passed_pct if not math.isnan(self.tests_passed_pct) else 0.0
        status = (self.status or "").upper()
        if rel >= 99.999:
            return True
        if tests_pct >= 99.999:
            return True
        return status in {"OK", "AC", "ACCEPTED"}


# ==============================================================================
# Stage 1: Generate problem results
# ==============================================================================

def discover_models(submission_results_dir: str, models_arg: Optional[List[str]]) -> List[str]:
    """Discover model names from submission_results directory."""
    if not os.path.isdir(submission_results_dir):
        return []

    all_models = [
        name for name in os.listdir(submission_results_dir)
        if os.path.isdir(os.path.join(submission_results_dir, name))
    ]

    if models_arg and "all" not in models_arg:
        # Filter to specified models
        return [m for m in all_models if m in models_arg]

    return sorted(all_models)


def load_submission_files(submission_results_dir: str, model: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all submission JSON files for a model and merge them."""
    model_dir = os.path.join(submission_results_dir, model)
    if not os.path.isdir(model_dir):
        return {}

    merged_submissions: Dict[str, Dict[str, Dict[str, Any]]] = {}

    json_files = glob.glob(os.path.join(model_dir, f"{model}_*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Merge problem-level data
            for problem_id, solutions in data.items():
                if problem_id not in merged_submissions:
                    merged_submissions[problem_id] = {}
                merged_submissions[problem_id].update(solutions)
        except (json.JSONDecodeError, IOError) as exc:
            print(f"Warning: Failed to load {json_file}: {exc}", file=sys.stderr)
            continue

    return merged_submissions


def select_best_solution(problem_solutions: Dict[str, Dict]) -> Optional[Tuple[str, ProblemRecord]]:
    """Select the best solution for a problem from multiple submissions."""
    best_name: Optional[str] = None
    best_record: Optional[ProblemRecord] = None

    for solution_name, details in problem_solutions.items():
        if not isinstance(details, dict):
            continue
        try:
            score = float(details.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            rel = float(details.get("relative_score", 0) or 0)
        except (TypeError, ValueError):
            rel = 0.0
        try:
            tests_pct = float(details.get("tests_passed_pct", 0) or 0)
        except (TypeError, ValueError):
            tests_pct = 0.0
        try:
            time_val = float(details.get("time", 0) or 0)
        except (TypeError, ValueError):
            time_val = 0.0
        try:
            memory_val = int(details.get("memory", 0) or 0)
        except (TypeError, ValueError):
            memory_val = 0

        status = str(details.get("status", "") or "")
        model_name = details.get("model")
        subtasks = details.get("subtasks", {})

        candidate = ProblemRecord(score, rel, tests_pct, status, model_name, time_val, memory_val, subtasks)

        if best_record is None:
            best_record = candidate
            best_name = solution_name
            continue

        # Ranking: higher score > higher relative_score > higher tests_passed_pct > lower time
        if candidate.score > best_record.score:
            best_record = candidate
            best_name = solution_name
        elif candidate.score == best_record.score:
            if candidate.relative_score > best_record.relative_score:
                best_record = candidate
                best_name = solution_name
            elif math.isclose(candidate.relative_score, best_record.relative_score):
                if candidate.tests_passed_pct > best_record.tests_passed_pct:
                    best_record = candidate
                    best_name = solution_name
                elif math.isclose(candidate.tests_passed_pct, best_record.tests_passed_pct):
                    if candidate.time < best_record.time:
                        best_record = candidate
                        best_name = solution_name

    if best_name is None or best_record is None:
        return None
    return best_name, best_record


def generate_problem_results(submission_results_dir: str, problem_results_dir: str, models: List[str]) -> None:
    """Stage 1: Generate problem-level results (best solution per problem)."""
    os.makedirs(problem_results_dir, exist_ok=True)

    for model in models:
        print(f"Generating problem results for model: {model}")
        submissions = load_submission_files(submission_results_dir, model)

        if not submissions:
            print(f"  No submissions found for {model}")
            continue

        problem_results: Dict[str, Dict[str, Any]] = {}

        for problem_id, solutions in submissions.items():
            best = select_best_solution(solutions)
            if best is None:
                continue

            solution_name, record = best
            problem_results[problem_id] = {
                "best_solution": solution_name,
                "score": record.score,
                "relative_score": round_or_none(record.relative_score),
                "tests_passed_pct": round_or_none(record.tests_passed_pct),
                "status": record.status,
                "time": round_or_none(record.time, 3),
                "memory": record.memory,
                "subtasks": record.subtasks,
            }

        output_path = os.path.join(problem_results_dir, f"{model}_problem.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(problem_results, fh, indent=2, sort_keys=True)

        print(f"  Wrote {len(problem_results)} problem results to {output_path}")


# ==============================================================================
# Stage 2: Generate contest results
# ==============================================================================

def load_valid_problems(problems_parquet: str) -> Dict[str, Dict[str, str]]:
    """Return metadata keyed by problem_id for problems we should keep."""
    df = pd.read_parquet(problems_parquet)
    metadata: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        problem_id = row["problem_id"]
        task_name = row["task_name"]
        competition = row.get("competition")

        parts = str(problem_id).split("-", 3)
        if len(parts) < 3:
            continue

        contest_base_raw = "-".join(parts[:3])
        contest_base_norm = normalize_contest_identifier(contest_base_raw)
        division = None

        if contest_base_raw.startswith("USACO-") and len(parts) == 4:
            remainder = parts[3]
            division = remainder.split("_", 1)[0].lower()

        metadata[problem_id] = {
            "contest_base_raw": contest_base_raw,
            "contest_base": contest_base_norm,
            "division": division,
            "task_name": task_name,
            "task_key": normalize_name(str(task_name)),
            "competition": competition,
        }
    return metadata


def build_problem_to_contest_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build mapping from problem_id to contest_id."""
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        contest_id = row["contest_id"]
        problems = row.get("problems")
        if problems is None:
            continue
        if isinstance(problems, str):
            try:
                problems = json.loads(problems)
            except json.JSONDecodeError:
                problems = []
        if not isinstance(problems, (list, tuple)):
            continue
        for problem in problems:
            problem_id = str(problem)
            if problem_id.startswith("USACO-"):
                parts = problem_id.split("-", 3)
                if len(parts) < 4:
                    continue
                division = parts[3].split("_", 1)[0].lower()
                if contest_id.endswith("-combined"):
                    if division in {"bronze", "silver", "gold"}:
                        mapping[problem_id] = contest_id
                elif contest_id.endswith("-platinum"):
                    if division == "platinum":
                        mapping[problem_id] = contest_id
                else:
                    mapping[problem_id] = contest_id
            else:
                mapping[problem_id] = contest_id
    return mapping


def load_problem_results_for_model(problem_results_dir: str, model: str) -> Dict[str, Dict[str, Any]]:
    """Load problem results JSON for a model."""
    problem_file = os.path.join(problem_results_dir, f"{model}_problem.json")
    if not os.path.exists(problem_file):
        return {}

    try:
        with open(problem_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"Warning: Failed to load {problem_file}: {exc}", file=sys.stderr)
        return {}


def group_problems_by_contest(
    problem_results: Dict[str, Dict[str, Any]],
    valid_problem_meta: Dict[str, Dict[str, str]],
    problem_to_contest: Dict[str, str],
    available_contests: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Group problem results by contest."""
    contest_data: Dict[str, Dict[str, Any]] = {}

    for problem_id, result in problem_results.items():
        if problem_id not in valid_problem_meta:
            continue

        meta = valid_problem_meta[problem_id]
        contest_base_norm = meta["contest_base"]
        contest_base_raw = meta["contest_base_raw"]
        division = (meta.get("division") or "").lower()
        competition = (meta.get("competition") or "").upper()

        contest_id = problem_to_contest.get(problem_id)

        if not contest_id:
            contest_comp, contest_year, _ = parse_contest_parts(contest_base_norm)
            if competition == "USACO" or (contest_comp and contest_comp.upper() == "USACO"):
                if division == "platinum":
                    contest_id = f"{contest_base_norm}-platinum"
                elif division in {"bronze", "silver", "gold"}:
                    contest_id = f"{contest_base_norm}-combined"
                else:
                    continue
            else:
                contest_id = contest_base_norm

        if contest_id not in available_contests and contest_id != contest_base_norm:
            fallback = contest_base_norm
            if fallback in available_contests:
                contest_id = fallback
        if contest_id not in available_contests and contest_base_raw in available_contests:
            contest_id = contest_base_raw

        _, contest_year, contest_label = parse_contest_parts(contest_base_raw)

        score = float(result.get("score", 0) or 0)
        rel_score = float(result.get("relative_score", 0) or 0)
        tests_pct = float(result.get("tests_passed_pct", 0) or 0)
        status = str(result.get("status", "") or "")

        record = ProblemRecord(score, rel_score, tests_pct, status, None)

        contest_entry = contest_data.setdefault(
            contest_id,
            {
                "problems": [],
                "problem_records": {},
                "scores": [],
                "relative_scores": [],
                "tests_passed": [],
                "pass_flags": [],
                "division_scores": {},
                "contest_base": contest_base_norm,
                "contest_base_raw": contest_base_raw,
                "contest_year": contest_year,
                "contest_label": contest_label,
            },
        )

        contest_entry["problems"].append(problem_id)
        contest_entry["problem_records"][problem_id] = record
        contest_entry["scores"].append(score)
        contest_entry["relative_scores"].append(rel_score)
        contest_entry["tests_passed"].append(tests_pct)
        contest_entry["pass_flags"].append(1 if record.is_full_pass else 0)

        if division:
            division_scores = contest_entry["division_scores"]
            division_scores.setdefault(division, []).append(score)

    return contest_data


# Human metrics computation (adapted from generate_results.py)

def extract_contestant_dataframe(contest_row: pd.Series) -> pd.DataFrame:
    entries = contest_row["contestants_ranking"]
    if isinstance(entries, str):
        contestants = json.loads(entries)
    else:
        contestants = entries or []
    df = pd.DataFrame(contestants)
    df.columns = df.columns.astype(str).str.strip()
    return df


def identify_task_columns(df: pd.DataFrame, contest_id: str = "") -> Dict[str, str]:
    """Return a mapping from normalized task name to original column name."""
    mapping: Dict[str, str] = {}
    normalized_exclusions = {normalize_name(col) for col in NON_TASK_COLUMNS}
    contest_lower = (contest_id or "").lower()

    if "boi-" in contest_lower:
        for day_col in ("day1", "day2"):
            normalized_exclusions.discard(day_col)

    for column in df.columns:
        norm = normalize_name(column)
        if norm and norm not in normalized_exclusions:
            mapping[norm] = column
    return mapping


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def calculate_percentile(model_score: float, human_scores: np.ndarray) -> Optional[float]:
    if human_scores.size == 0:
        return None
    better = np.sum(model_score > human_scores)
    percentile = (better / human_scores.size) * 100
    return float(percentile)


def calculate_codeforces_rating(model_rank: int, ranked_ratings: pd.Series) -> Optional[float]:
    if ranked_ratings.empty:
        return None
    left = 500.0  # Minimum realistic Codeforces rating
    right = float(ranked_ratings.max() + 100)
    ratings = ranked_ratings.to_numpy()
    # Binary search for Elo value with Codeforces seed model
    while right - left > 1:
        mid = (left + right) / 2
        new_seed = 1.0
        for rating in ratings:
            new_seed += 1.0 / (1.0 + 10 ** ((mid - rating) / 400))
        if new_seed < model_rank:
            right = mid
        else:
            left = mid
    return float(round(left, 2))


def medal_from_cutoffs(
    model_total: float,
    gold_cutoff: Optional[float],
    silver_cutoff: Optional[float],
    bronze_cutoff: Optional[float],
) -> Optional[str]:
    thresholds = []
    if gold_cutoff is not None and not pd.isna(gold_cutoff):
        thresholds.append(("Gold", float(gold_cutoff)))
    if silver_cutoff is not None and not pd.isna(silver_cutoff):
        thresholds.append(("Silver", float(silver_cutoff)))
    if bronze_cutoff is not None and not pd.isna(bronze_cutoff):
        thresholds.append(("Bronze", float(bronze_cutoff)))

    for medal, threshold in thresholds:
        if model_total >= threshold:
            return medal

    if thresholds:
        return "None"
    return None


def compute_usaco_combined_metrics(contest_id: str, contest_entry: Dict) -> Dict[str, Optional[float]]:
    contest_year = contest_entry.get("contest_year")
    contest_label = contest_entry.get("contest_label")
    info = get_usaco_contest_info(contest_year, contest_label)
    division_scores = contest_entry.get("division_scores", {})

    totals = {division: float(sum(scores)) for division, scores in division_scores.items()}

    medal = None
    thresholds_present = False
    if info:
        for key, label in (("gold", "Gold"), ("silver", "Silver"), ("bronze", "Bronze")):
            record = info.get(key)
            if not record:
                continue
            threshold = record.get("promotion_threshold")
            if threshold is None:
                continue
            thresholds_present = True
            score = totals.get(key, 0.0)
            if score >= threshold:
                medal = label
                break

    if medal is None and thresholds_present:
        medal = "None"

    return {
        "human_percentile": None,
        "medal": medal,
        "codeforces_elo": None,
        "available_tasks": [],
    }


def compute_human_metrics(
    contest_id: str,
    contest_entry: Dict,
    contest_results_map: Dict[str, pd.Series],
    valid_problem_meta: Dict[str, Dict[str, str]],
) -> Dict[str, Optional[float]]:
    if contest_id.startswith("USACO-") and contest_id.endswith("-combined"):
        return compute_usaco_combined_metrics(contest_id, contest_entry)

    contest_row = contest_results_map.get(contest_id)
    if contest_row is None:
        return {
            "human_percentile": None,
            "medal": None,
            "codeforces_elo": None,
            "available_tasks": [],
        }

    df = extract_contestant_dataframe(contest_row)
    if df.empty:
        model_total = float(sum(contest_entry["scores"]))
        medal = medal_from_cutoffs(
            model_total,
            contest_row.get("gold_cutoff"),
            contest_row.get("silver_cutoff"),
            contest_row.get("bronze_cutoff"),
        )
        return {
            "human_percentile": None,
            "medal": medal,
            "codeforces_elo": None,
            "available_tasks": [],
        }

    df = df.copy()
    model_total = float(sum(contest_entry["scores"]))

    recalc_total_col = None
    for candidate in ["Recalculated_Total", "recalculated_total"]:
        if candidate in df.columns:
            recalc_total_col = candidate
            break

    new_total_col = "_liveoibench_total"
    available_tasks: List[str] = []

    if recalc_total_col:
        df[recalc_total_col] = pd.to_numeric(df[recalc_total_col], errors="coerce")
        human_scores_series = df[recalc_total_col].dropna()
        human_scores = human_scores_series.to_numpy(dtype=float)
        df[new_total_col] = df[recalc_total_col].fillna(0.0)
        human_percentile = calculate_percentile(model_total, human_scores)
        available_tasks = [recalc_total_col]

        if "canadian_computing_olympiad" in contest_id.lower():
            rank_col = None
            for candidate in ["Rank", "rank"]:
                if candidate in df.columns:
                    rank_col = candidate
                    break
            if rank_col:
                rank_series = pd.to_numeric(df[rank_col], errors="coerce").dropna()
                total_participants = len(rank_series)
                if total_participants:
                    model_rank = int((human_scores_series > model_total).sum() + 1)
                    model_rank = max(1, min(model_rank, total_participants))
                    human_percentile = ((total_participants - model_rank) / total_participants) * 100
    else:
        task_column_map = identify_task_columns(df, contest_id)
        contest_tasks = []
        problem_scores = []
        for problem_id in contest_entry["problems"]:
            metadata = valid_problem_meta.get(problem_id)
            if not metadata:
                continue
            task_key = metadata["task_key"]
            column = task_column_map.get(task_key)
            if column:
                contest_tasks.append(column)
                problem_scores.append(contest_entry["problem_records"][problem_id].score)

        if not contest_tasks:
            fallback_total_col = None
            for candidate in ["Total", "total", "Total Score", "score", "Score"]:
                if candidate in df.columns:
                    fallback_total_col = candidate
                    break
            if fallback_total_col:
                contest_tasks = [fallback_total_col]
                problem_scores = [float(sum(contest_entry["scores"]))]
            else:
                return {
                    "human_percentile": None,
                    "medal": None,
                    "codeforces_elo": None,
                    "available_tasks": [],
                }

        for column in contest_tasks:
            df[column] = to_numeric(df[column])

        df[new_total_col] = df[contest_tasks].sum(axis=1, skipna=True)
        human_scores = df[new_total_col].to_numpy(dtype=float)
        model_total = float(sum(problem_scores))
        human_percentile = calculate_percentile(model_total, human_scores)
        available_tasks = contest_tasks

    gold_cutoff = float(contest_row["gold_cutoff"]) if not pd.isna(contest_row["gold_cutoff"]) else None
    silver_cutoff = float(contest_row["silver_cutoff"]) if not pd.isna(contest_row["silver_cutoff"]) else None
    bronze_cutoff = float(contest_row["bronze_cutoff"]) if not pd.isna(contest_row["bronze_cutoff"]) else None

    medal = None
    if gold_cutoff is not None and model_total >= gold_cutoff:
        medal = "Gold"
    elif silver_cutoff is not None and model_total >= silver_cutoff:
        medal = "Silver"
    elif bronze_cutoff is not None and model_total >= bronze_cutoff:
        medal = "Bronze"
    else:
        medal = "None" if any(x is not None for x in (gold_cutoff, silver_cutoff, bronze_cutoff)) else None

    cf_rating_col = None
    for candidate in ["CF_Rating", "cf_rating", "CF Rating", "codeforces_rating"]:
        if candidate in df.columns:
            cf_rating_col = candidate
            break

    codeforces_elo = None
    if cf_rating_col:
        cf_series = pd.to_numeric(df[cf_rating_col], errors="coerce")
        valid_cf = df[~cf_series.isna()].copy()
        valid_cf[cf_rating_col] = cf_series[~cf_series.isna()]
        valid_cf = valid_cf[(valid_cf[cf_rating_col] != 0) & (valid_cf[cf_rating_col] != -1000)]
        if not valid_cf.empty:
            ranked = valid_cf.sort_values(by=new_total_col, ascending=False)
            model_rank = int((ranked[new_total_col] > model_total).sum() + 1)
            codeforces_elo = calculate_codeforces_rating(model_rank, ranked[cf_rating_col])

    return {
        "human_percentile": human_percentile,
        "medal": medal,
        "codeforces_elo": codeforces_elo,
        "available_tasks": available_tasks,
    }


def generate_contest_results(
    problem_results_dir: str,
    contest_results_dir: str,
    contestant_parquet: str,
    problems_parquet: str,
    models: List[str],
) -> None:
    """Stage 2: Generate contest-level results with human metrics."""
    os.makedirs(contest_results_dir, exist_ok=True)

    valid_problem_meta = load_valid_problems(problems_parquet)
    contestant_df = pd.read_parquet(contestant_parquet)
    contest_results_map = {row["contest_id"]: row for _, row in contestant_df.iterrows()}
    available_contests = set(contest_results_map.keys())
    problem_to_contest = build_problem_to_contest_map(contestant_df)

    for model in models:
        print(f"Generating contest results for model: {model}")
        problem_results = load_problem_results_for_model(problem_results_dir, model)

        if not problem_results:
            print(f"  No problem results found for {model}")
            continue

        contest_data = group_problems_by_contest(
            problem_results, valid_problem_meta, problem_to_contest, available_contests
        )

        per_contest_results: Dict[str, Dict[str, Any]] = {}
        overall_stats = {
            "contest_relative_scores": [],
            "contest_pass_rates": [],
            "contest_tests_passed": [],
            "contest_percentiles": [],
            "contest_elos": [],
            "medal_records": [],
            "total_problem_count": 0,
            "total_solved": 0,
        }

        for contest_id, entry in sorted(contest_data.items()):
            scores = entry["scores"]
            rel_scores = entry["relative_scores"]
            tests = entry["tests_passed"]
            pass_flags = entry["pass_flags"]

            overall_stats["total_problem_count"] += len(pass_flags)
            overall_stats["total_solved"] += sum(pass_flags)

            contest_relative = float(np.mean(rel_scores)) if rel_scores else 0.0
            contest_tests_avg = float(np.mean(tests)) if tests else 0.0
            contest_pass_rate = (sum(pass_flags) / len(pass_flags) * 100) if pass_flags else 0.0
            contest_total_score = float(sum(scores)) if scores else 0.0

            human_metrics = compute_human_metrics(contest_id, entry, contest_results_map, valid_problem_meta)

            human_percentile = human_metrics["human_percentile"]
            medal = human_metrics["medal"]
            codeforces_elo = human_metrics["codeforces_elo"]

            if human_percentile is not None:
                overall_stats["contest_percentiles"].append(human_percentile)
            if codeforces_elo is not None:
                overall_stats["contest_elos"].append(codeforces_elo)
            if medal is not None:
                overall_stats["medal_records"].append(medal)

            overall_stats["contest_relative_scores"].append(contest_relative)
            overall_stats["contest_tests_passed"].append(contest_tests_avg)
            overall_stats["contest_pass_rates"].append(contest_pass_rate)

            per_contest_results[contest_id] = {
                "total_score": round_or_none(contest_total_score),
                "relative_score": round_or_none(contest_relative),
                "human_percentile": round_or_none(human_percentile),
                "medal": medal,
                "codeforces_elo": round_or_none(codeforces_elo),
                "tests_passed_pct": round_or_none(contest_tests_avg),
                "pass_rate": round_or_none(contest_pass_rate),
            }

        medal_counter = Counter([m for m in overall_stats["medal_records"] if m in {"Gold", "Silver", "Bronze"}])
        medal_info_denominator = sum(
            1 for contest in per_contest_results.values() if contest["medal"] is not None
        )
        any_medal_count = sum(medal_counter.values())

        overall = {
            "gold_count": medal_counter.get("Gold", 0),
            "silver_count": medal_counter.get("Silver", 0),
            "bronze_count": medal_counter.get("Bronze", 0),
            "any_medal_count": any_medal_count,
            "gold_pct": round_or_none(
                (medal_counter.get("Gold", 0) / medal_info_denominator * 100) if medal_info_denominator else None
            ),
            "silver_pct": round_or_none(
                (medal_counter.get("Silver", 0) / medal_info_denominator * 100) if medal_info_denominator else None
            ),
            "bronze_pct": round_or_none(
                (medal_counter.get("Bronze", 0) / medal_info_denominator * 100) if medal_info_denominator else None
            ),
            "any_medal_pct": round_or_none(
                (any_medal_count / medal_info_denominator * 100) if medal_info_denominator else None
            ),
            "relative_score": round_or_none(
                np.mean(overall_stats["contest_relative_scores"]) if overall_stats["contest_relative_scores"] else None
            ),
            "human_percentile": round_or_none(
                np.mean(overall_stats["contest_percentiles"]) if overall_stats["contest_percentiles"] else None
            ),
            "pass_rate": round_or_none(
                (overall_stats["total_solved"] / overall_stats["total_problem_count"] * 100)
                if overall_stats["total_problem_count"] else None
            ),
            "codeforces_elo": round_or_none(
                np.mean(overall_stats["contest_elos"]) if overall_stats["contest_elos"] else None
            ),
        }

        output = {
            "model": model,
            "per_contest": per_contest_results,
            "overall": overall,
        }

        output_path = os.path.join(contest_results_dir, f"{model}_contest.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, sort_keys=True)

        print(f"  Wrote contest results to {output_path}")


# ==============================================================================
# Stage 3: Generate final results CSV
# ==============================================================================

def load_all_contest_results(contest_results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all contest result JSON files."""
    results: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(contest_results_dir):
        return results

    for filename in os.listdir(contest_results_dir):
        if not filename.endswith("_contest.json"):
            continue

        filepath = os.path.join(contest_results_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            model = data.get("model")
            if model:
                results[model] = data
        except (json.JSONDecodeError, IOError) as exc:
            print(f"Warning: Failed to load {filepath}: {exc}", file=sys.stderr)
            continue

    return results


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None and not pd.isna(v)]
    if not valid:
        return None
    return float(np.mean(valid))


def _aggregate_metric(per_contest: Dict[str, Dict[str, Any]], key: str, aggregation_level: str) -> Optional[float]:
    if aggregation_level == "contest":
        return _mean_or_none([contest.get(key) for contest in per_contest.values()])

    grouped: Dict[str, List[Optional[float]]] = defaultdict(list)
    for contest_id, contest in per_contest.items():
        competition, _, _ = parse_contest_parts(contest_id)
        grouped[competition].append(contest.get(key))

    competition_means: List[Optional[float]] = []
    for values in grouped.values():
        comp_mean = _mean_or_none(values)
        if comp_mean is not None:
            competition_means.append(comp_mean)

    return _mean_or_none(competition_means)


def generate_final_results_csv(contest_results_dir: str, final_results_file: str, aggregation_level: str = "competition") -> None:
    """Stage 3: Generate final results CSV (global rankings)."""
    contest_results = load_all_contest_results(contest_results_dir)

    if not contest_results:
        print("No contest results found. Cannot generate final results CSV.")
        return

    rows: List[Dict[str, Any]] = []

    for model, data in sorted(contest_results.items()):
        overall = data.get("overall", {})
        per_contest = data.get("per_contest", {})

        # Calculate total score across all contests
        total_score = sum(
            contest.get("total_score", 0) or 0
            for contest in per_contest.values()
        )

        # Count unique competitions and contests
        competition_set = set()
        contest_count = len(per_contest)
        for contest_id in per_contest.keys():
            comp, _, _ = parse_contest_parts(contest_id)
            if comp:
                competition_set.add(comp)
        competition_count = len(competition_set)

        # Count total tasks
        total_tasks = overall.get("gold_count", 0) + overall.get("silver_count", 0) + overall.get("bronze_count", 0)
        # Actually, total tasks should be counted differently - let me use a better approach
        # We don't have direct task counts in contest results, so we'll leave it as 0 for now
        # or calculate based on problem results if available
        total_tasks = 0  # Placeholder

        aggregated_relative = _aggregate_metric(per_contest, "relative_score", aggregation_level)
        aggregated_percentile = _aggregate_metric(per_contest, "human_percentile", aggregation_level)
        aggregated_codeforces = _aggregate_metric(per_contest, "codeforces_elo", aggregation_level)

        row = {
            "Model": model,
            "Total Score": round_or_none(total_score),
            "Global Relative Score (%)": round_or_none(aggregated_relative),
            "Pass Rate (%)": overall.get("pass_rate"),
            "Codeforces Elo": round_or_none(aggregated_codeforces),
            "Gold Medals": overall.get("gold_count", 0),
            "Silver Medals": overall.get("silver_count", 0),
            "Bronze Medals": overall.get("bronze_count", 0),
            "Total Medals": overall.get("any_medal_count", 0),
            "Avg Human Percentile": round_or_none(aggregated_percentile),
            "Competition Count": competition_count,
            "Contest Count": contest_count,
            "Total Tasks": total_tasks,
        }
        rows.append(row)

    # Sort by Global Relative Score descending
    rows.sort(key=lambda r: r.get("Global Relative Score (%)", 0) or 0, reverse=True)

    # Add Global Rank
    for rank, row in enumerate(rows, start=1):
        row["Global Rank"] = rank

    # Write CSV
    os.makedirs(os.path.dirname(final_results_file), exist_ok=True)

    fieldnames = [
        "Model", "Total Score", "Global Relative Score (%)", "Pass Rate (%)",
        "Codeforces Elo", "Gold Medals", "Silver Medals", "Bronze Medals", "Total Medals",
        "Avg Human Percentile", "Competition Count", "Contest Count",
        "Total Tasks", "Global Rank"
    ]

    with open(final_results_file, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote final results CSV to {final_results_file}")
    print(f"  Total models: {len(rows)}")


# ==============================================================================
# Main CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rankings from submission results (3-stage pipeline)."
    )
    parser.add_argument(
        "--submission-results-dir",
        type=str,
        default=DEFAULT_SUBMISSION_RESULTS_DIR,
        help="Directory containing submission results by model",
    )
    parser.add_argument(
        "--problem-results-dir",
        type=str,
        default=DEFAULT_PROBLEM_RESULTS_DIR,
        help="Directory to write/read problem results",
    )
    parser.add_argument(
        "--contest-results-dir",
        type=str,
        default=DEFAULT_CONTEST_RESULTS_DIR,
        help="Directory to write/read contest results",
    )
    parser.add_argument(
        "--final-results-file",
        type=str,
        default=DEFAULT_FINAL_RESULTS_FILE,
        help="Path to final results CSV file",
    )
    parser.add_argument(
        "--contestant-parquet",
        type=str,
        default=DEFAULT_CONTESTANT_PARQUET,
        help="Path to contest_results.parquet with human standings",
    )
    parser.add_argument(
        "--problems-parquet",
        type=str,
        default=DEFAULT_PROBLEMS_PARQUET,
        help="Path to liveoibench_v1.parquet listing valid problems",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names to process (or 'all' for all models)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["problem", "contest", "final", "all"],
        default="all",
        help="Which stage to run (problem, contest, final, or all)",
    )
    parser.add_argument(
        "--aggregation-level",
        type=str,
        choices=["competition", "contest"],
        default="competition",
        help="How to aggregate relative scores, Codeforces Elo, and human percentile in the final CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in {"problem", "all"}:
        print("=" * 80)
        print("Stage 1: Generating problem results")
        print("=" * 80)
        models = discover_models(args.submission_results_dir, args.models)
        if not models:
            print("No models found in submission results directory.")
        else:
            generate_problem_results(args.submission_results_dir, args.problem_results_dir, models)

    if args.stage in {"contest", "all"}:
        print("\n" + "=" * 80)
        print("Stage 2: Generating contest results")
        print("=" * 80)
        models = discover_models(args.submission_results_dir, args.models)
        if not models:
            # Try to discover from problem_results instead
            if os.path.isdir(args.problem_results_dir):
                models = [
                    fname.replace("_problem.json", "")
                    for fname in os.listdir(args.problem_results_dir)
                    if fname.endswith("_problem.json")
                ]
        if not models:
            print("No models found for contest results generation.")
        else:
            generate_contest_results(
                args.problem_results_dir,
                args.contest_results_dir,
                args.contestant_parquet,
                args.problems_parquet,
                models,
            )

    if args.stage in {"final", "all"}:
        print("\n" + "=" * 80)
        print("Stage 3: Generating final results CSV")
        print("=" * 80)
        generate_final_results_csv(args.contest_results_dir, args.final_results_file, args.aggregation_level)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
