#!/usr/bin/env bash
set -euo pipefail

# Wrapper to rebuild the IOI-Bench style dataset layout for LiveOIBench.
# All paths are derived from the LiveOIBench root directory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVEOIBENCH_ROOT="${LIVEOIBENCH_ROOT:-/data2/kai/LiveOIBench}"

python "${ROOT_DIR}/src/process_dataset.py" \
  --download-dir "${LIVEOIBENCH_ROOT}/cache" \
  --output-dir "${LIVEOIBENCH_ROOT}/data" \
  "$@"
