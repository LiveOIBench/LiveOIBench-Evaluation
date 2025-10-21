# Paths follow repository guidelines.
export LIVEOIBENCH_DATA_DIR="/data2/kai/IOI-Bench-Restructured"
export LIVEOIBENCH_EVAL_RESOURCE_DIR="/data2/kai/LiveOIBench"
export LIVEOIBENCH_PREDICTIONS_DIR="/data2/kai/IOI-Evaluation/ioi-predictions/predictions"
export LIVEOIBENCH_RESULTS_DIR="/data2/kai/LiveOIBench/results"
export LIVEOIBENCH_CACHE_DIR="/data2/kai/LiveOIBench/cache"
export TESTLIB_PATH="/data2/kai/IOI-Evaluation/evaluation/testlib.h"

# Print values safely (will error if unset because of -u, which is desired here)
printf "%s\n" "LiveOIBench environment configured:"
printf "  LIVEOIBENCH_DATA_DIR=%s\n"         "${LIVEOIBENCH_DATA_DIR}"
printf "  LIVEOIBENCH_EVAL_RESOURCE_DIR=%s\n" "${LIVEOIBENCH_EVAL_RESOURCE_DIR}"
printf "  LIVEOIBENCH_PREDICTIONS_DIR=%s\n"   "${LIVEOIBENCH_PREDICTIONS_DIR}"
printf "  LIVEOIBENCH_RESULTS_DIR=%s\n"       "${LIVEOIBENCH_RESULTS_DIR}"
printf "  LIVEOIBENCH_CACHE_DIR=%s\n"         "${LIVEOIBENCH_CACHE_DIR}"
printf "  TESTLIB_PATH=%s\n"                  "${TESTLIB_PATH}"