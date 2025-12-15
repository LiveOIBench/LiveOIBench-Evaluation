export LIVEOIBENCH_EVALUATION_DIR=${LIVEOIBENCH_ROOT}/evaluation

python src/generate_rankings.py \
  --submission-results-dir ${LIVEOIBENCH_EVALUATION_DIR}/submission_results \
  --problem-results-dir ${LIVEOIBENCH_EVALUATION_DIR}/problem_results \
  --contest-results-dir ${LIVEOIBENCH_EVALUATION_DIR}/contest_results \
  --final-results-file ${LIVEOIBENCH_EVALUATION_DIR}/final_results.csv \
  --contestant-parquet ${LIVEOIBENCH_ROOT}/parquet_files/contestants/data/contest_results.parquet \
  --problems-parquet ${LIVEOIBENCH_ROOT}/parquet_files/problems/data/liveoibench_v1.parquet \
  --models all \
  --stage all
