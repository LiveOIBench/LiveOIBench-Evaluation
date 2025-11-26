#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${LIVEOIBENCH_DATA_DIR}"
EVAL_DIR="${LIVEOIBENCH_EVAL_RESOURCE_DIR}"
RESULTS_DIR="${LIVEOIBENCH_RESULTS_DIR}"
PREDICTIONS_DIR="${LIVEOIBENCH_PREDICTIONS_DIR}"
CACHE_DIR="${LIVEOIBENCH_CACHE_DIR}"
MODEL_NAME="${LIVEOIBENCH_MODEL_NAME:-gpt-oss-120b-medium}"

LLM_SOLUTIONS_DIR="${PREDICTIONS_DIR}"
if [[ -d "${LLM_SOLUTIONS_DIR}/predictions" ]]; then
  LLM_SOLUTIONS_DIR="${LLM_SOLUTIONS_DIR}/predictions"
fi

if [[ ! -d "${LLM_SOLUTIONS_DIR}" ]]; then
  echo "Error: LLM solutions directory '${LLM_SOLUTIONS_DIR}' not found." >&2
  exit 1
fi

RUN_TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RESULTS_JSON="${RESULTS_DIR}/${MODEL_NAME}/${MODEL_NAME}_${RUN_TIMESTAMP}.json"

python src/run_judge.py batch \
  --competitions USACO \
  --years 2023-2025 \
  --llm_models "${MODEL_NAME}" \
  --solution_types llm \
  --llm_solutions_dir "${LLM_SOLUTIONS_DIR}" \
  --data_dir "${DATA_DIR}" \
  --evaluation_dir "${EVAL_DIR}" \
  --output_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --stop_on_failure \
  --workers 6 \
  --max_solutions 8
#IOI BOI CEOI CCO COCI EGOI EJOI IATI OOI USACO RMI APIO JOI NOINordic