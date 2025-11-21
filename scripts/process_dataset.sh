#!/usr/bin/env bash
set -euo pipefail

# Wrapper that downloads LiveOIBench metadata + tests from HuggingFace artifacts
# and rebuilds the IOI-Bench style data directory. Extra CLI flags are
# forwarded to src/process_dataset.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_CACHE="${LIVEOIBENCH_DATA_CACHE:-/data2/kai/huggingface/LiveOIBench_tests}"
EVAL_RESOURCE_DIR="${IOI_EVAL_RESOURCE_DIR:-/data2/kai/LiveOIBench}"
OUTPUT_DIR="${LIVEOIBENCH_DATA_OUTPUT:-${EVAL_RESOURCE_DIR}/data}"
METADATA_PARQUET="${LIVEOIBENCH_METADATA_PARQUET:-/data2/kai/huggingface/LiveOIBench/data/liveoibench_v1.parquet}"

printf "%s\n" "Rehydrating tests into: ${OUTPUT_DIR}"

python "${ROOT_DIR}/src/process_dataset.py" \
  --download-dir "${DATA_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --metadata-parquet "${METADATA_PARQUET}" \
  --overwrite \
  "$@"
