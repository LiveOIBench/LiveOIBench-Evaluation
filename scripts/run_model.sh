python src/generate_solutions.py \
  --model gpt-oss-20b-low \
  --competitions IOI BOI CEOI CCO COCI EGOI EJOI IATI OOI USACO RMI APIO JOI NOINordic \
  --years 2023 2024 2025 \
  --mode json \
  --vllm \
  --port 8080 \
  --seeds 8
