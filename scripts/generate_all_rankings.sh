#!/usr/bin/env bash
# Generate all rankings from submission results (all three stages)
#
# Usage:
#   ./scripts/generate_all_rankings.sh [--models model1 model2 ...] [--stage problem|contest|final|all]
#
# Environment variables (see scripts/set_liveoibench_env.sh):
#   LIVEOIBENCH_SUBMISSION_RESULTS_DIR - Submission results directory
#   LIVEOIBENCH_PROBLEM_RESULTS_DIR    - Problem results directory
#   LIVEOIBENCH_CONTEST_RESULTS_DIR    - Contest results directory
#   LIVEOIBENCH_FINAL_RESULTS          - Final results CSV path
#
# Examples:
#   # Generate all stages for all models
#   ./scripts/generate_all_rankings.sh
#
#   # Generate only problem results for specific models
#   ./scripts/generate_all_rankings.sh --models gpt-4o claude-3.5 --stage problem
#
#   # Generate only final results CSV
#   ./scripts/generate_all_rankings.sh --stage final

set -euo pipefail

# Default values
MODELS="all"
STAGE="all"
CONTESTANT_PARQUET="${CONTESTANT_PARQUET:-/data2/kai/huggingface/LiveOIBench_contestants/data/contest_results.parquet}"
PROBLEMS_PARQUET="${PROBLEMS_PARQUET:-/data2/kai/huggingface/LiveOIBench/data/liveoibench_v1.parquet}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            MODELS=""
            while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
                MODELS="$MODELS $1"
                shift
            done
            MODELS="${MODELS## }"  # Trim leading space
            ;;
        --stage)
            shift
            STAGE="$1"
            shift
            ;;
        --contestant-parquet)
            shift
            CONTESTANT_PARQUET="$1"
            shift
            ;;
        --problems-parquet)
            shift
            PROBLEMS_PARQUET="$1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--models model1 model2 ...] [--stage problem|contest|final|all]"
            echo ""
            echo "Options:"
            echo "  --models           Model names to process (default: all)"
            echo "  --stage            Which stage to run: problem, contest, final, or all (default: all)"
            echo "  --contestant-parquet  Path to contestant standings parquet file"
            echo "  --problems-parquet    Path to problems parquet file"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if environment variables are set
if [[ -z "${LIVEOIBENCH_SUBMISSION_RESULTS_DIR:-}" ]]; then
    echo "Error: LIVEOIBENCH_SUBMISSION_RESULTS_DIR not set"
    echo "Please source scripts/set_liveoibench_env.sh first"
    exit 1
fi

if [[ -z "${LIVEOIBENCH_PROBLEM_RESULTS_DIR:-}" ]]; then
    echo "Error: LIVEOIBENCH_PROBLEM_RESULTS_DIR not set"
    echo "Please source scripts/set_liveoibench_env.sh first"
    exit 1
fi

if [[ -z "${LIVEOIBENCH_CONTEST_RESULTS_DIR:-}" ]]; then
    echo "Error: LIVEOIBENCH_CONTEST_RESULTS_DIR not set"
    echo "Please source scripts/set_liveoibench_env.sh first"
    exit 1
fi

if [[ -z "${LIVEOIBENCH_FINAL_RESULTS:-}" ]]; then
    echo "Error: LIVEOIBENCH_FINAL_RESULTS not set"
    echo "Please source scripts/set_liveoibench_env.sh first"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "Generate Rankings Pipeline"
echo "================================================================================"
echo "Stage:                    $STAGE"
echo "Models:                   $MODELS"
echo "Submission Results Dir:   $LIVEOIBENCH_SUBMISSION_RESULTS_DIR"
echo "Problem Results Dir:      $LIVEOIBENCH_PROBLEM_RESULTS_DIR"
echo "Contest Results Dir:      $LIVEOIBENCH_CONTEST_RESULTS_DIR"
echo "Final Results CSV:        $LIVEOIBENCH_FINAL_RESULTS"
echo "Contestant Parquet:       $CONTESTANT_PARQUET"
echo "Problems Parquet:         $PROBLEMS_PARQUET"
echo "================================================================================"
echo ""

# Build command
CMD="python src/generate_rankings.py"
CMD="$CMD --submission-results-dir \"$LIVEOIBENCH_SUBMISSION_RESULTS_DIR\""
CMD="$CMD --problem-results-dir \"$LIVEOIBENCH_PROBLEM_RESULTS_DIR\""
CMD="$CMD --contest-results-dir \"$LIVEOIBENCH_CONTEST_RESULTS_DIR\""
CMD="$CMD --final-results-file \"$LIVEOIBENCH_FINAL_RESULTS\""
CMD="$CMD --contestant-parquet \"$CONTESTANT_PARQUET\""
CMD="$CMD --problems-parquet \"$PROBLEMS_PARQUET\""
CMD="$CMD --models $MODELS"
CMD="$CMD --stage \"$STAGE\""

# Execute
echo "Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "================================================================================"
echo "Done!"
echo "================================================================================"
