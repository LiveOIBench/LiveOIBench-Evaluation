export LIVEOIBENCH_ROOT="/data2/kai/LiveOIBench-Testing"

python src/process_dataset.py \
  --download-dir "${LIVEOIBENCH_ROOT}/parquet_files" \
  --output-dir "${LIVEOIBENCH_ROOT}/data"
