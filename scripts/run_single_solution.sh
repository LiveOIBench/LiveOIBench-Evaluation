#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${LIVE_OIBENCH_DATA_DIR}"
EVAL_DIR="${LIVE_OIBENCH_EVAL_RESOURCE_DIR}"

python src/run_judge.py single \
  --competition COCI \
  --year 2024 \
  --round CONTEST_#5 \
  --task bitovi \
  --solution_file /data2/kai/IOI-Bench-Restructured/COCI/2024/CONTEST_#5/bitovi/solutions/codes/bitovi.cpp \
  --problem_folder "${DATA_DIR}" \
  --evaluation_folder "${EVAL_DIR}" \
  --max_workers 1 \
  --verbose
