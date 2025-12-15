import argparse
import json
import logging
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_LIVEOIBENCH_ROOT = Path(os.getenv("LIVEOIBENCH_ROOT", "/data2/kai/LiveOIBench")).expanduser()
DEFAULT_PARQUET_FILES_DIR = DEFAULT_LIVEOIBENCH_ROOT / "parquet_files"
DEFAULT_OUTPUT_DIR = DEFAULT_LIVEOIBENCH_ROOT / "data"
DEFAULT_PROBLEM_PARQUET = ["liveoibench_v1.parquet"]
DEFAULT_CONTESTANT_PARQUET = ["contest_results.parquet"]
DEFAULT_TEST_PARQUET = [
    "liveoibench_testcases_v1_2023.parquet",
    "liveoibench_testcases_v1_2024.parquet",
    "liveoibench_testcases_v1_2025.parquet",
]

DEFAULT_PROBLEM_REPO_ID = "LiveOIBench/LiveOIBench"
DEFAULT_CONTESTANT_REPO_ID = "LiveOIBench/LiveOIBench_contestants"
DEFAULT_TEST_REPO_ID = "LiveOIBench/LiveOIBench_tests"


LOGGER = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the downloaded dataset is missing required content."""


def parse_problem_id(problem_id: str) -> Tuple[str, str, str, str]:
    """Split a compound identifier into competition/year/round/task."""
    parts = problem_id.split("-")
    if len(parts) < 4:
        raise DatasetError(f"Invalid problem_id '{problem_id}'; expected at least 4 tokens.")
    competition, year = parts[0], parts[1]
    round_name = parts[2]
    task = "-".join(parts[3:])
    return competition, year, round_name, task


def download_parquet_files(
    repo_id: str,
    download_dir: Path,
    include_files: Sequence[str] | None = None,
) -> List[Path]:
    """Download every parquet artifact from the HuggingFace dataset repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = sorted(f for f in files if f.endswith(".parquet"))
    if not parquet_files:
        raise DatasetError(f"No parquet files found in dataset repo {repo_id}.")

    include_set = {Path(f).name for f in include_files} if include_files else None
    if include_set is not None:
        available = {Path(f).name for f in parquet_files}
        missing = sorted(include_set - available)
        if missing:
            raise DatasetError(
                f"Requested parquet files not found in dataset repo {repo_id}: {', '.join(missing)}"
            )
        parquet_files = [f for f in parquet_files if Path(f).name in include_set]

    local_paths: List[Path] = []
    for filename in parquet_files:
        target_path = download_dir / filename
        if target_path.exists():
            local_paths.append(target_path)
            continue
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )
        local_paths.append(Path(local_path))
    return local_paths


def iter_dataset_rows(parquet_paths: Sequence[Path]) -> Iterable[Dict[str, object]]:
    """Yield problem rows stored in the downloaded parquet files."""
    desired_columns = ["problem_id", "tests", "subtasks", "grader_codes", "starter_codes"]
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = [col for col in desired_columns if col in parquet_file.schema.names]
        if "problem_id" not in available_columns:
            raise DatasetError(f"Parquet file {parquet_path} missing required 'problem_id' column.")
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=available_columns)
            column_data = {col: table.column(col).to_pylist() for col in available_columns}
            num_rows = len(column_data["problem_id"])
            for idx in range(num_rows):
                row: Dict[str, object] = {"problem_id": column_data["problem_id"][idx]}
                for col in desired_columns[1:]:
                    if col in column_data:
                        row[col] = column_data[col][idx]
                    else:
                        row[col] = None
                row["tests"] = row.get("tests") or "{}"
                row["subtasks"] = row.get("subtasks") or "{}"
                yield row


def maybe_filter(value: str, filters: Sequence[str] | None) -> bool:
    """Return True when the given value matches the provided whitelist."""
    return not filters or value in filters


def ensure_parquet_file(parquet_path: Path, repo_id: str, download_dir: Path, label: str) -> Path:
    """
    Ensure the target parquet exists locally, downloading it from the given repo if missing.

    Returns the path to the available parquet file.
    """
    if parquet_path.exists():
        return parquet_path

    download_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "%s %s not found; downloading from %s into %s",
        label.capitalize(),
        parquet_path.name,
        repo_id,
        download_dir,
    )
    parquet_files = download_parquet_files(
        repo_id=repo_id,
        download_dir=download_dir,
        include_files=[parquet_path.name],
    )
    downloaded_path = next((path for path in parquet_files if path.name == parquet_path.name), None)
    if not downloaded_path:
        raise DatasetError(f"Unable to download {label} {parquet_path.name} from {repo_id}.")
    return downloaded_path


def write_problem_assets(problem_row: Dict[str, str], output_root: Path, overwrite: bool) -> None:
    """Materialize tests plus subtasks.json for a single problem."""
    competition, year, round_name, task = parse_problem_id(problem_row["problem_id"])
    tests_dir = output_root / competition / year / round_name / task / "tests"
    subtasks_path = tests_dir.parent / "subtasks.json"

    if tests_dir.exists():
        if not overwrite:
            print(f"Skipping existing tests: {problem_row['problem_id']}")
            return
        for path in tests_dir.glob("*"):
            if path.is_file():
                path.unlink()
        for path in tests_dir.glob("*"):
            if path.is_dir():
                shutil.rmtree(path)
    else:
        tests_dir.mkdir(parents=True, exist_ok=True)

    tests = json.loads(problem_row.get("tests") or "{}")
    if not isinstance(tests, dict):
        raise DatasetError(f"Malformed tests payload for {problem_row['problem_id']}")

    for test_name, payload in tests.items():
        payload = payload or {}
        input_path = tests_dir / f"{test_name}.in"
        output_path = tests_dir / f"{test_name}.out"
        input_path.write_text(payload.get("input", ""), encoding="utf-8")
        output_path.write_text(payload.get("output", ""), encoding="utf-8")

    raw_subtasks = problem_row.get("subtasks") or "{}"
    try:
        subtasks_obj = json.loads(raw_subtasks)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Malformed subtasks payload for {problem_row['problem_id']}") from exc

    subtasks_path.parent.mkdir(parents=True, exist_ok=True)
    subtasks_path.write_text(json.dumps(subtasks_obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_code_mapping(raw_value: object, context: str) -> Dict[str, str]:
    """Convert parquet payloads to filename->contents dictionaries."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return {}
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse %s payload: %s", context, exc)
            return {}
        return _normalize_code_mapping(parsed, context)
    if isinstance(raw_value, dict):
        return {str(filename): content for filename, content in raw_value.items() if isinstance(content, str) and content}
    if isinstance(raw_value, list):
        mapping: Dict[str, str] = {}
        for entry in raw_value:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename") or entry.get("name")
            content = entry.get("content") or entry.get("code")
            if isinstance(filename, str) and isinstance(content, str) and content:
                mapping[filename] = content
        return mapping
    LOGGER.warning("Unsupported %s payload type: %s", context, type(raw_value))
    return {}


def _select_code_payload(problem_row: Mapping[str, object], keys: Sequence[str]) -> object | None:
    """Return the first non-empty payload available for the provided keys."""
    for key in keys:
        if key not in problem_row:
            continue
        value = problem_row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _write_problem_codes(problem_row: Mapping[str, object], problem_dir: Path) -> None:
    """Write grader/starter bundles from either metadata or test parquet rows."""
    problem_id = problem_row.get("problem_id", "unknown")

    grader_payload = _select_code_payload(problem_row, ("grader_codes", "grader_code"))
    grader_codes = _normalize_code_mapping(grader_payload, f"{problem_id} grader_code")
    if grader_codes:
        graders_dir = problem_dir / "graders"
        graders_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in grader_codes.items():
            (graders_dir / filename).write_text(content, encoding="utf-8")

    starter_payload = _select_code_payload(problem_row, ("starter_codes", "starter_code"))
    starter_codes = _normalize_code_mapping(starter_payload, f"{problem_id} starter_code")

    if starter_codes:
        attachments_dir = problem_dir / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in starter_codes.items():
            (attachments_dir / filename).write_text(content, encoding="utf-8")


class ParquetReconstructor:
    """Rebuild LiveOIBench-style structure from a metadata parquet file."""

    def __init__(
        self,
        parquet_path: Path,
        output_dir: Path,
        include_competitions: Sequence[str] | None = None,
        include_years: Sequence[str] | None = None,
        include_rounds: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.include_competitions = include_competitions
        self.include_years = include_years
        self.include_rounds = include_rounds
        self.dry_run = dry_run
        self.stats: Dict[str, int] = {
            "problems_processed": 0,
            "files_created": 0,
            "directories_created": 0,
        }
        self.errors: List[str] = []
        self.contest_tasks: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)

    def create_directory(self, path: Path) -> None:
        if self.dry_run:
            LOGGER.debug("[DRY RUN] Would create directory: %s", path)
            return
        path.mkdir(parents=True, exist_ok=True)
        self.stats["directories_created"] += 1

    def write_file(self, path: Path, content: str) -> None:
        if self.dry_run:
            LOGGER.debug("[DRY RUN] Would write file: %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.stats["files_created"] += 1

    def write_json(self, path: Path, payload: Dict[str, object]) -> None:
        self.write_file(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def should_process(self, competition: str, year: str, round_name: str) -> bool:
        if not maybe_filter(competition, self.include_competitions):
            return False
        if not maybe_filter(year, self.include_years):
            return False
        if not maybe_filter(round_name, self.include_rounds):
            return False
        return True

    def process_problem(self, row: pd.Series) -> None:
        competition, year, round_name, task = parse_problem_id(row["problem_id"])
        if not self.should_process(competition, year, round_name):
            return

        problem_dir = self.output_dir / competition / year / round_name / task
        try:
            self.create_directory(problem_dir)

            problem_json: Dict[str, object] = {
                "task_name": row.get("task_name"),
                "task_type": row.get("task_type"),
                "time_limit": row.get("time_limit"),
                "memory_limit": row.get("memory_limit"),
            }
            if pd.notna(row.get("difficulty")):
                problem_json["difficulty"] = row.get("difficulty")
            if pd.notna(row.get("algorithms")):
                try:
                    problem_json["algorithms"] = json.loads(row["algorithms"])
                except (json.JSONDecodeError, TypeError):
                    problem_json["algorithms"] = []
            self.write_json(problem_dir / "problem.json", problem_json)

            statement_path: Path | None = None
            if pd.notna(row.get("problem_statement")):
                statements_dir = problem_dir / "statements"
                self.create_directory(statements_dir)
                statement_path = statements_dir / "statement.md"
                self.write_file(statement_path, row["problem_statement"])

            if row.get("task_type") == "interactive":
                graders_dir = problem_dir / "graders"
                setup_script = row.get("setup_script")
                evaluation_script = row.get("evaluation_script")
                if pd.notna(setup_script) and setup_script:
                    self.create_directory(graders_dir)
                    self.write_file(graders_dir / "setup.sh", setup_script)
                if pd.notna(evaluation_script) and evaluation_script:
                    self.create_directory(graders_dir)
                    self.write_file(graders_dir / "evaluate.sh", evaluation_script)

            if not self.dry_run:
                _write_problem_codes(row.to_dict(), problem_dir)
                self._write_prompt(
                    problem_dir=problem_dir,
                    task=task,
                    time_limit=problem_json.get("time_limit"),
                    memory_limit=problem_json.get("memory_limit"),
                )

            self.stats["problems_processed"] += 1
            self.contest_tasks[(competition, year, round_name)].add(task)
        except Exception as exc:
            msg = f"Error processing {row['problem_id']}: {exc}"
            LOGGER.error(msg)
            self.errors.append(msg)

    def create_meta_info_files(self) -> None:
        for (competition, year, round_name), tasks in self.contest_tasks.items():
            contest_dir = self.output_dir / competition / year / round_name
            meta_info = {
                "contest": round_name,
                "tasks": sorted(tasks),
            }
            self.write_json(contest_dir / "meta_info.json", meta_info)

    def _write_prompt(
        self,
        problem_dir: Path,
        task: str,
        time_limit: object,
        memory_limit: object,
    ) -> None:
        """Generate a prompt file similar to judges.problem.Problem.get_prompt."""
        statement_path = problem_dir / "statements" / "statement.md"
        attachments_dir = problem_dir / "attachments"
        prompt = (
            f"Given a competition problem below, write a solution in C++ that solves all the subtasks. "
            f"Make sure to wrap your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
        )
        if statement_path.exists():
            prompt += "[BEGIN PROBLEM]\n"
            prompt += statement_path.read_text(encoding="utf-8")
            prompt += "[END PROBLEM]\n"
        time_text = time_limit if time_limit is not None else "unknown"
        memory_text = memory_limit if memory_limit is not None else "unknown"
        prompt += f"Time limit: {time_text} seconds\n"
        prompt += f"Memory limit: {memory_text} MB\n"

        grader_candidates = [
            attachments_dir / "grader.cpp",
            attachments_dir / "stub.cpp",
        ]
        grader_path = next((path for path in grader_candidates if path.exists()), None)
        header_path = attachments_dir / f"{task}.h"
        starter_path = attachments_dir / f"{task}.cpp"

        if grader_path and header_path.exists() and starter_path.exists():
            prompt += (
                "We are going to grade your solution using the following grader.cpp file and "
                f"{task}.h file. You can write your solution by modifying the {task}.cpp and wrap "
                f"your code in '```{task}.cpp' and '```' Markdown delimiters.\n\n"
            )
            prompt += "```grader.cpp\n" + grader_path.read_text(encoding="utf-8") + "```\n\n"
            prompt += f"```{task}.h\n" + header_path.read_text(encoding="utf-8") + "```\n\n"
            prompt += f"```{task}.cpp\n" + starter_path.read_text(encoding="utf-8") + "```\n\n"
        else:
            prompt += (
                f"Generate a solution in C++ that solves the task. Make sure to wrap your code in "
                f"'```{task}.cpp' and '```' Markdown delimiters.\n\n"
            )
        self.write_file(problem_dir / "prompt.txt", prompt)

    def reconstruct(self) -> None:
        LOGGER.info("Reconstructing metadata from %s", self.parquet_path)
        if not self.parquet_path.exists():
            raise DatasetError(f"Metadata parquet file not found: {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        LOGGER.info("Loaded %d problems from metadata parquet", len(df))

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            if idx % 50 == 0:
                LOGGER.info("Processed %d metadata rows", idx)
            self.process_problem(row)

        self.create_meta_info_files()
        LOGGER.info(
            "Metadata reconstruction complete: %d problems, %d files, %d directories",
            self.stats["problems_processed"],
            self.stats["files_created"],
            self.stats["directories_created"],
        )
        if self.errors:
            LOGGER.warning("Encountered %d errors during metadata reconstruction", len(self.errors))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and materialize LiveOIBench test cases.")
    parser.add_argument(
        "--problem-repo-id",
        default=DEFAULT_PROBLEM_REPO_ID,
        help="HuggingFace dataset repo id containing the problem parquet.",
    )
    parser.add_argument(
        "--contestant-repo-id",
        default=DEFAULT_CONTESTANT_REPO_ID,
        help="HuggingFace dataset repo id containing the contestant parquet.",
    )
    parser.add_argument(
        "--test-repo-id",
        default=DEFAULT_TEST_REPO_ID,
        help="HuggingFace dataset repo id containing the test parquet shards.",
    )
    parser.add_argument(
        "--problem-parquet",
        nargs="+",
        default=None,
        help="Problem parquet filename(s) containing metadata (problem statements, graders, etc.). Defaults to liveoibench_v1.parquet; will reuse local files under parquet_files/problems or download when missing.",
    )
    parser.add_argument(
        "--contestant-parquet",
        nargs="+",
        default=None,
        help="Contestant standings parquet filename(s). Defaults to contest_results.parquet; will reuse local files under parquet_files/contestants or download when missing.",
    )
    parser.add_argument(
        "--test-parquet",
        nargs="+",
        default=None,
        help="Specific test parquet filename to download/process (default: download all available shards).",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_PARQUET_FILES_DIR),
        help="Directory containing (or caching) the HuggingFace parquet files. Defaults to LIVEOIBENCH_ROOT/parquet_files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory where the LiveOIBench layout should be rebuilt (defaults to LIVEOIBENCH_ROOT/data).",
    )
    parser.add_argument(
        "--include-competitions",
        nargs="+",
        default=None,
        help="Subset of competitions to import (default: all).",
    )
    parser.add_argument(
        "--include-years",
        nargs="+",
        default=None,
        help="Subset of years to import (default: all).",
    )
    parser.add_argument(
        "--include-rounds",
        nargs="+",
        default=None,
        help="Subset of round names to import (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after processing this many problems (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate tests folders even if they already exist.",
    )
    parser.add_argument(
        "--skip-problems",
        action="store_true",
        help="Skip problem metadata reconstruction (statements, graders, attachments, etc.).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test case reconstruction.",
    )
    parser.add_argument(
        "--metadata-dry-run",
        action="store_true",
        help="Only simulate metadata reconstruction (still runs tests stage unless skipped).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for metadata reconstruction.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(message)s")
    problem_repo_id = args.problem_repo_id
    contestant_repo_id = args.contestant_repo_id
    test_repo_id = args.test_repo_id

    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    download_dir = Path(args.download_dir).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)

    problem_download_dir = download_dir / "problems"
    contestant_download_dir = download_dir / "contestants"
    test_download_dir = download_dir / "tests"
    problem_download_dir.mkdir(parents=True, exist_ok=True)
    contestant_download_dir.mkdir(parents=True, exist_ok=True)
    test_download_dir.mkdir(parents=True, exist_ok=True)

    # Resolve parquet files, preferring local copies under parquet_files/<type> before downloading
    problem_files = args.problem_parquet or DEFAULT_PROBLEM_PARQUET
    contestant_files = args.contestant_parquet or DEFAULT_CONTESTANT_PARQUET
    test_files = args.test_parquet or DEFAULT_TEST_PARQUET

    def resolve_files(filenames: Sequence[str], local_dir: Path, cache_dir: Path, repo_id: str, label: str) -> List[Path]:
        resolved: List[Path] = []
        for name in filenames:
            name_path = Path(name)
            if name_path.is_absolute() and name_path.exists():
                resolved.append(name_path)
                continue
            local_candidate = local_dir / name
            if local_candidate.exists():
                resolved.append(local_candidate)
                continue
            target_path = cache_dir / name
            resolved.append(
                ensure_parquet_file(
                    parquet_path=target_path,
                    repo_id=repo_id,
                    download_dir=cache_dir,
                    label=label,
                )
            )
        return resolved

    problem_local_dir = DEFAULT_PARQUET_FILES_DIR / "problems"
    contestant_local_dir = DEFAULT_PARQUET_FILES_DIR / "contestants"
    test_local_dir = DEFAULT_PARQUET_FILES_DIR / "tests"

    problem_parquet_paths = resolve_files(
        problem_files, problem_local_dir, problem_download_dir, problem_repo_id, "problem parquet"
    )
    contestant_parquet_paths = resolve_files(
        contestant_files, contestant_local_dir, contestant_download_dir, contestant_repo_id, "contestant parquet"
    )
    test_parquet_paths = resolve_files(
        test_files, test_local_dir, test_download_dir, test_repo_id, "test parquet"
    )
    problem_parquet_path = problem_parquet_paths[0]
    contestant_parquet_path = contestant_parquet_paths[0]
    LOGGER.debug("Contestant parquet available at %s", contestant_parquet_path)

    if not args.skip_problems:
        reconstructor = ParquetReconstructor(
            parquet_path=problem_parquet_path,
            output_dir=output_root,
            include_competitions=args.include_competitions,
            include_years=args.include_years,
            include_rounds=args.include_rounds,
            dry_run=args.metadata_dry_run,
        )
        reconstructor.reconstruct()
    else:
        LOGGER.info("Skipping problem reconstruction per CLI flag.")

    if args.skip_tests:
        LOGGER.info("Skipping test case reconstruction per CLI flag.")
        return
    parquet_paths = test_parquet_paths
    LOGGER.info("Found %d parquet files to import.", len(parquet_paths))

    processed = 0
    for row in iter_dataset_rows(parquet_paths):
        competition, year, round_name, _task = parse_problem_id(row["problem_id"])
        if not maybe_filter(competition, args.include_competitions):
            continue
        if not maybe_filter(year, args.include_years):
            continue
        if not maybe_filter(round_name, args.include_rounds):
            continue

        write_problem_assets(row, output_root, overwrite=args.overwrite)
        processed += 1
        if processed % 25 == 0:
            LOGGER.info("Processed %d problems during test reconstruction", processed)
        if args.limit and processed >= args.limit:
            break

    LOGGER.info("Finished reconstructing tests for %d problems into %s", processed, output_root)


if __name__ == "__main__":
    main()
