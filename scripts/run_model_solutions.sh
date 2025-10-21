DATA_DIR="${LIVEOIBENCH_DATA_DIR}"
EVAL_DIR="${LIVEOIBENCH_EVAL_RESOURCE_DIR}"
RESULTS_DIR="${LIVEOIBENCH_RESULTS_DIR}"
PREDICTIONS_DIR="${LIVEOIBENCH_PREDICTIONS_DIR}"
CACHE_DIR="${LIVEOIBENCH_CACHE_DIR}"

python src/test_solutions.py \
  --competitions IOI BOI CEOI CCO COCI EGOI EJOI IATI OOI USACO RMI APIO JOI NOINordic \
  --years 2023-2025 \
  --llm_models gpt-5 \
  --solution_types llm_solutions \
  --llm_solutions_dir "${PREDICTIONS_DIR}" \
  --data_dir "${DATA_DIR}" \
  --evaluation_dir "${EVAL_DIR}" \
  --output_dir "${RESULTS_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --stop_on_failure \
  --workers 6 \
  --max_solutions 8
