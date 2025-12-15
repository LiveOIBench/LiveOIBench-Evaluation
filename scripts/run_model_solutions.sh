DATA_DIR="${LIVEOIBENCH_ROOT}/data"
EVAL_DIR="${LIVEOIBENCH_ROOT}/evaluation"
RESULTS_DIR="${EVAL_DIR}/submission_results"
LLM_SOLUTIONS_DIR="${LIVEOIBENCH_ROOT}/predictions"
CACHE_DIR="${EVAL_DIR}/cache"
MODEL_NAME="all"

python src/run_judge.py batch \
  --competitions IOI BOI CEOI CCO COCI EGOI EJOI IATI OOI USACO RMI APIO JOI NOINordic \
  --years 2023-2025 \
  --llm_models "${MODEL_NAME}" \
  --solution_types llm \
  --llm_json_dir "${LLM_SOLUTIONS_DIR}" \
  --data_dir "${DATA_DIR}" \
  --evaluation_dir "${EVAL_DIR}" \
  --output_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --stop_on_failure \
  --workers 6 \
  --reeval \
  --max_solutions 8