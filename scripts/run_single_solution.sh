DATA_DIR="${LIVEOIBENCH_ROOT}/data"
EVAL_DIR="${LIVEOIBENCH_ROOT}/evaluation"

python src/run_judge.py single \
  --competition IOI \
  --year 2024 \
  --round contest \
  --task closing \
  --solution_file <solution_file> \
  --problem_folder "${DATA_DIR}" \
  --evaluation_folder "${EVAL_DIR}" \
  --max_workers 1 \
  --verbose
