#!/usr/bin/env python3
"""
Generate contest-level performance summaries for a single model run.

The script consumes a model evaluation JSON file produced by the LiveOIBench
pipeline and aligns those results with human contest standings stored in
`contest_results.parquet`. Only problems listed in the LiveOIBench v1 dataset
are considered. For each contest, we keep the highest-scoring solution per
problem, aggregate contest metrics, compute human-relative statistics, infer
medal tiers, and estimate a Codeforces rating analogue. A consolidated JSON
report is written to `/data2/kai/LiveOIBench/model_rankings` by default.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd


DEFAULT_MODEL_RESULTS = "/data2/kai/LiveOIBench/results/gpt-5_20251031T161818Z.json"
DEFAULT_CONTESTANT_PARQUET = (
    "/data2/kai/huggingface/LiveOIBench_contestants/data/contest_results.parquet"
)
DEFAULT_PROBLEMS_PARQUET = (
    "/data2/kai/huggingface/LiveOIBench/data/liveoibench_v1.parquet"
)
DEFAULT_OUTPUT_DIR = "/data2/kai/LiveOIBench/model_rankings"
USACO_INFO_ROOT = "/data2/kai/IOI-Bench-Restructured/USACO"

# Columns that should never be treated as task scores when parsing human results.
NON_TASK_COLUMNS = {
    "rank",
    "contestant",
    "country",
    "total",
    "recalculated_total",
    "medal",
    "cf_rating",
    "day1",
    "day2",
    "day 1",
    "day 2",
    "score rel.",
    "division",
    "team",
    "nationality",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate contest-level statistics for a model evaluation run."
    )
    parser.add_argument(
        "--model-results",
        type=str,
        required=True,
        help="Path to the model evaluation JSON file.",
    )
    parser.add_argument(
        "--contestant-parquet",
        type=str,
        default=DEFAULT_CONTESTANT_PARQUET,
        help="Path to contest_results.parquet with human standings.",
    )
    parser.add_argument(
        "--problems-parquet",
        type=str,
        default=DEFAULT_PROBLEMS_PARQUET,
        help="Path to liveoibench_v1.parquet listing valid problems.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the aggregated JSON report.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional custom filename for the output JSON (without directory).",
    )
    return parser.parse_args()


def build_problem_to_contest_map(df: pd.DataFrame) -> Dict[str, str]:
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


@dataclass
class ProblemRecord:
    score: float
    relative_score: float
    tests_passed_pct: float
    status: str
    model: Optional[str]

    @property
    def is_full_pass(self) -> bool:
        rel = self.relative_score if not math.isnan(self.relative_score) else 0.0
        tests_pct = (
            self.tests_passed_pct if not math.isnan(self.tests_passed_pct) else 0.0
        )
        status = (self.status or "").upper()
        if rel >= 99.999:
            return True
        if tests_pct >= 99.999:
            return True
        return status in {"OK", "AC", "ACCEPTED"}


def normalize_name(value: str) -> str:
    """Normalize task column names for consistent matching."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def load_valid_problems(path: str) -> Dict[str, Dict[str, str]]:
    """Return metadata keyed by problem_id for problems we should keep."""
    df = pd.read_parquet(path)
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


def select_best_solution(problem_solutions: Dict[str, Dict]) -> Optional[ProblemRecord]:
    best: Optional[ProblemRecord] = None
    for details in problem_solutions.values():
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
        status = str(details.get("status", "") or "")
        model = details.get("model")
        candidate = ProblemRecord(score, rel, tests_pct, status, model)
        if best is None:
            best = candidate
            continue
        if candidate.score > best.score:
            best = candidate
            continue
        if candidate.score == best.score and candidate.relative_score > best.relative_score:
            best = candidate
            continue
        if (
            candidate.score == best.score
            and math.isclose(candidate.relative_score, best.relative_score)
            and candidate.tests_passed_pct > best.tests_passed_pct
        ):
            best = candidate
    return best


def load_model_results(
    model_results_path: str,
    valid_problem_meta: Dict[str, Dict[str, str]],
    problem_to_contest: Dict[str, str],
    available_contests: Set[str],
) -> Tuple[Dict[str, Dict], Optional[str]]:
    with open(model_results_path, "r", encoding="utf-8") as fh:
        raw_results = json.load(fh)

    contest_data: Dict[str, Dict] = {}
    model_name: Optional[str] = None

    for problem_id, submissions in raw_results.items():
        if problem_id not in valid_problem_meta:
            continue
        best_solution = select_best_solution(submissions)
        if best_solution is None:
            continue

        if model_name is None and best_solution.model:
            model_name = str(best_solution.model)

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
        contest_entry["problem_records"][problem_id] = best_solution
        contest_entry["scores"].append(best_solution.score)
        contest_entry["relative_scores"].append(best_solution.relative_score)
        contest_entry["tests_passed"].append(best_solution.tests_passed_pct)
        contest_entry["pass_flags"].append(1 if best_solution.is_full_pass else 0)
        if division:
            division_scores = contest_entry["division_scores"]
            division_scores.setdefault(division, []).append(best_solution.score)

    if model_name is None:
        basename = os.path.basename(model_results_path)
        model_name = basename.split("_", 1)[0]

    return contest_data, model_name


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


def compute_scaled_cutoff(
    df: pd.DataFrame,
    original_total_col: Optional[str],
    new_total_col: str,
    original_cutoff: Optional[float],
) -> Optional[float]:
    if original_cutoff is None or original_total_col is None:
        return None
    original_totals = pd.to_numeric(df[original_total_col], errors="coerce")
    eligible = df[original_totals >= original_cutoff]
    if eligible.empty:
        return None
    return float(eligible[new_total_col].min())


def calculate_percentile(model_score: float, human_scores: np.ndarray) -> Optional[float]:
    if human_scores.size == 0:
        return None
    better = np.sum(model_score > human_scores)
    percentile = (better / human_scores.size) * 100
    return float(percentile)


def calculate_codeforces_rating(model_rank: int, ranked_ratings: pd.Series) -> Optional[float]:
    if ranked_ratings.empty:
        return None
    left = 0.0
    right = float(ranked_ratings.max() + 100)
    ratings = ranked_ratings.to_numpy()
    n = len(ratings)
    # Binary search for Elo value with Codeforces seed model.
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


def compute_usaco_combined_metrics(
    contest_id: str, contest_entry: Dict
) -> Dict[str, Optional[float]]:
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
        "human_rank": None,
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
            "human_rank": None,
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
            "human_rank": None,
            "medal": medal,
            "codeforces_elo": None,
            "available_tasks": [],
        }

    df = df.copy()
    model_total = float(sum(contest_entry["scores"]))
    human_rank: Optional[int] = None

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
        if human_scores.size:
            human_rank = int(np.sum(human_scores > model_total) + 1)
        available_tasks = [recalc_total_col]
        original_total_col = recalc_total_col

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
                    human_rank = model_rank
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
                    "human_rank": None,
                    "medal": None,
                    "codeforces_elo": None,
                    "available_tasks": [],
                }

        original_total_col = None
        for candidate in ["Total", "total", "Total Score", "score", "Score"]:
            if candidate in df.columns:
                original_total_col = candidate
                break
        original_total_lower = {col.lower(): col for col in df.columns}
        if original_total_col is None and "recalculated_total" in original_total_lower:
            original_total_col = original_total_lower["recalculated_total"]

        for column in contest_tasks:
            df[column] = to_numeric(df[column])

        df[new_total_col] = df[contest_tasks].sum(axis=1, skipna=True)
        human_scores_series = df[new_total_col].dropna()
        human_scores = human_scores_series.to_numpy(dtype=float)
        model_total = float(sum(problem_scores))
        human_percentile = calculate_percentile(model_total, human_scores)
        if human_scores.size:
            human_rank = int(np.sum(human_scores > model_total) + 1)
        available_tasks = contest_tasks

    gold_cutoff = float(contest_row["gold_cutoff"]) if not pd.isna(contest_row["gold_cutoff"]) else None
    silver_cutoff = (
        float(contest_row["silver_cutoff"]) if not pd.isna(contest_row["silver_cutoff"]) else None
    )
    bronze_cutoff = (
        float(contest_row["bronze_cutoff"]) if not pd.isna(contest_row["bronze_cutoff"]) else None
    )

    scaled_gold = gold_cutoff
    scaled_silver = silver_cutoff
    scaled_bronze = bronze_cutoff

    medal = None
    if scaled_gold is not None and model_total >= scaled_gold:
        medal = "Gold"
    elif scaled_silver is not None and model_total >= scaled_silver:
        medal = "Silver"
    elif scaled_bronze is not None and model_total >= scaled_bronze:
        medal = "Bronze"
    else:
        medal = "None" if any(x is not None for x in (scaled_gold, scaled_silver, scaled_bronze)) else None

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
            codeforces_elo = calculate_codeforces_rating(
                model_rank, ranked[cf_rating_col]
            )

    return {
        "human_percentile": human_percentile,
        "human_rank": human_rank,
        "medal": medal,
        "codeforces_elo": codeforces_elo,
        "available_tasks": available_tasks,
    }


def build_output_filename(args: argparse.Namespace, model_results_path: str) -> str:
    if args.output_name:
        filename = args.output_name
    else:
        base = os.path.splitext(os.path.basename(model_results_path))[0]
        filename = f"{base}_contest_results.json"
    if not filename.lower().endswith(".json"):
        filename += ".json"
    return os.path.join(args.output_dir, filename)


def round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return float(round(value, digits))


def main() -> None:
    args = parse_args()

    valid_problem_meta = load_valid_problems(args.problems_parquet)
    contestant_df = pd.read_parquet(args.contestant_parquet)
    contest_results_map = {
        row["contest_id"]: row for _, row in contestant_df.iterrows()
    }
    available_contests = set(contest_results_map.keys())
    problem_to_contest = build_problem_to_contest_map(contestant_df)

    contest_data, model_name = load_model_results(
        args.model_results, valid_problem_meta, problem_to_contest, available_contests
    )

    if not contest_data:
        raise ValueError("No valid contest data found for the provided model results.")

    per_contest_results: Dict[str, Dict[str, Optional[float]]] = {}
    contest_relative_scores: List[float] = []
    contest_pass_rates: List[float] = []
    contest_tests_passed: List[float] = []
    contest_percentiles: List[float] = []
    contest_elos: List[float] = []
    medal_records: List[str] = []
    total_problem_count = 0
    total_solved = 0

    for contest_id, entry in sorted(contest_data.items()):
        scores = entry["scores"]
        rel_scores = entry["relative_scores"]
        tests = entry["tests_passed"]
        pass_flags = entry["pass_flags"]

        total_problem_count += len(pass_flags)
        total_solved += sum(pass_flags)

        contest_relative = float(np.mean(rel_scores)) if rel_scores else 0.0
        contest_tests_avg = float(np.mean(tests)) if tests else 0.0
        contest_pass_rate = (sum(pass_flags) / len(pass_flags) * 100) if pass_flags else 0.0
        contest_total_score = float(sum(scores)) if scores else 0.0

        human_metrics = compute_human_metrics(
            contest_id, entry, contest_results_map, valid_problem_meta
        )

        human_percentile = human_metrics["human_percentile"]
        human_rank = human_metrics["human_rank"]
        medal = human_metrics["medal"]
        codeforces_elo = human_metrics["codeforces_elo"]

        if human_percentile is not None:
            contest_percentiles.append(human_percentile)
        if codeforces_elo is not None:
            contest_elos.append(codeforces_elo)
        if medal is not None:
            medal_records.append(medal)

        contest_relative_scores.append(contest_relative)
        contest_tests_passed.append(contest_tests_avg)
        contest_pass_rates.append(contest_pass_rate)

        per_contest_results[contest_id] = {
            "total_score": round_or_none(contest_total_score),
            "relative_score": round_or_none(contest_relative),
            "human_percentile": round_or_none(human_percentile),
             "human_rank": human_rank,
            "medal": medal,
            "codeforces_elo": round_or_none(codeforces_elo),
            "tests_passed_pct": round_or_none(contest_tests_avg),
            "pass_rate": round_or_none(contest_pass_rate),
        }

    total_contests = len(per_contest_results)
    medal_counter = Counter([m for m in medal_records if m in {"Gold", "Silver", "Bronze"}])
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
            (medal_counter.get("Gold", 0) / medal_info_denominator * 100)
            if medal_info_denominator
            else None
        ),
        "silver_pct": round_or_none(
            (medal_counter.get("Silver", 0) / medal_info_denominator * 100)
            if medal_info_denominator
            else None
        ),
        "bronze_pct": round_or_none(
            (medal_counter.get("Bronze", 0) / medal_info_denominator * 100)
            if medal_info_denominator
            else None
        ),
        "any_medal_pct": round_or_none(
            (any_medal_count / medal_info_denominator * 100)
            if medal_info_denominator
            else None
        ),
        "relative_score": round_or_none(np.mean(contest_relative_scores) if contest_relative_scores else None),
        "human_percentile": round_or_none(np.mean(contest_percentiles) if contest_percentiles else None),
        "pass_rate": round_or_none(
            (total_solved / total_problem_count * 100) if total_problem_count else None
        ),
        "codeforces_elo": round_or_none(np.mean(contest_elos) if contest_elos else None),
    }

    output = {
        "model": model_name,
        "input_file": args.model_results,
        "per_contest": per_contest_results,
        "overall": overall,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_filename(args, args.model_results)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)

    print(f"Wrote contest summary to {output_path}")


if __name__ == "__main__":
    main()
