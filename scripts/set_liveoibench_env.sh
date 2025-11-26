# Paths follow repository guidelines.
export LIVEOIBENCH_DATA_DIR="/data2/kai/LiveOIBench/data"
export LIVEOIBENCH_PARSE_DIR="/data2/kai/LiveOIBench/data"
export LIVEOIBENCH_EVAL_RESOURCE_DIR="/data2/kai/LiveOIBench"
export LIVEOIBENCH_PREDICTIONS_DIR="/data2/kai/IOI-Evaluation/ioi-predictions/old"
export LIVEOIBENCH_RESULTS_DIR="/data2/kai/LiveOIBench/evaluation/submission_results"
export TESTLIB_PATH="/data2/kai/IOI-Evaluation/evaluation/testlib.h"

#Reorganized evaluation directory structure
export LIVEOIBENCH_EVAL_DIR="${LIVEOIBENCH_EVAL_RESOURCE_DIR}/evaluation"
export LIVEOIBENCH_CACHE_DIR="${LIVEOIBENCH_EVAL_DIR}/cache"
export LIVEOIBENCH_SUBMISSION_RESULTS_DIR="${LIVEOIBENCH_EVAL_DIR}/submission_results"
export LIVEOIBENCH_PROBLEM_RESULTS_DIR="${LIVEOIBENCH_EVAL_DIR}/problem_results"
export LIVEOIBENCH_CONTEST_RESULTS_DIR="${LIVEOIBENCH_EVAL_DIR}/contest_results"
export LIVEOIBENCH_FINAL_RESULTS="${LIVEOIBENCH_EVAL_DIR}/final_results.csv"

# Print values safely (will error if unset because of -u, which is desired here)
printf "%s\n" "LiveOIBench environment configured:"
printf "  LIVEOIBENCH_DATA_DIR=%s\n"                  "${LIVEOIBENCH_DATA_DIR}"
printf "  LIVEOIBENCH_EVAL_RESOURCE_DIR=%s\n"          "${LIVEOIBENCH_EVAL_RESOURCE_DIR}"
printf "  LIVEOIBENCH_PREDICTIONS_DIR=%s\n"            "${LIVEOIBENCH_PREDICTIONS_DIR}"
printf "  LIVEOIBENCH_RESULTS_DIR=%s\n"                "${LIVEOIBENCH_RESULTS_DIR}"
printf "  TESTLIB_PATH=%s\n"                           "${TESTLIB_PATH}"
printf "  LIVEOIBENCH_EVAL_DIR=%s\n"                   "${LIVEOIBENCH_EVAL_DIR}"
printf "  LIVEOIBENCH_CACHE_DIR=%s\n"                  "${LIVEOIBENCH_CACHE_DIR}"
printf "  LIVEOIBENCH_SUBMISSION_RESULTS_DIR=%s\n"     "${LIVEOIBENCH_SUBMISSION_RESULTS_DIR}"
printf "  LIVEOIBENCH_PROBLEM_RESULTS_DIR=%s\n"        "${LIVEOIBENCH_PROBLEM_RESULTS_DIR}"
printf "  LIVEOIBENCH_CONTEST_RESULTS_DIR=%s\n"        "${LIVEOIBENCH_CONTEST_RESULTS_DIR}"
printf "  LIVEOIBENCH_FINAL_RESULTS=%s\n"              "${LIVEOIBENCH_FINAL_RESULTS}"