#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${IOI_BENCH_DATA_DIR:-./data}"
EVAL_DIR="${IOI_EVAL_RESOURCE_DIR:-./evaluation}"
RESULTS_DIR="${IOI_EVAL_RESULTS_DIR:-${EVAL_DIR}/results}"

python src/evaluation/test_solutions.py \
  --competition RMI \
  --years 2023 \
  --tasks To_be_xor_not_to_be \
  --solution_types all \
  --data_dir "${DATA_DIR}" \
  --evaluation_dir "${EVAL_DIR}" \
  --output_dir "${RESULTS_DIR}" \
  --max_solutions 3 \
  --reeval \
  --workers 10 \
  --output_format csv
