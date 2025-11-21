#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${IOI_BENCH_DATA_DIR:-./data}"
EVAL_DIR="${IOI_EVAL_RESOURCE_DIR:-./evaluation}"
RESULTS_DIR="${IOI_EVAL_RESULTS_DIR:-${EVAL_DIR}/results}"
CACHE_DIR="${IOI_EVAL_CACHE_DIR:-${RESULTS_DIR}/cache}"
RUN_TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RESULTS_JSON="${RESULTS_DIR}/results_${RUN_TIMESTAMP}.json"

python src/run_judge.py batch \
  --competitions RMI \
  --years 2023 \
  --tasks To_be_xor_not_to_be \
  --solution_types all \
  --data_dir "${DATA_DIR}" \
  --evaluation_dir "${EVAL_DIR}" \
  --output_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --max_solutions 3 \
  --reeval \
  --workers 10 \
  --results_file "${RESULTS_JSON}"

python src/generate_results.py \
  --results_file "${RESULTS_JSON}" \
  --output_dir "${RESULTS_DIR}" \
  --skip_comparisons
