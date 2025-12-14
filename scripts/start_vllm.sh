#!/usr/bin/env bash
# Lightweight wrapper around vLLM server startup with neutral defaults.
# Configure environment variables before running to match your hardware setup.

set -euo pipefail

export HF_HOME=/data2/shared_model/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2

GPUS=1
MODEL_NAME=openai/gpt-oss-20b
MAX_MODEL_LEN=128000
VLLM_PORT=8080

vllm serve "${MODEL_NAME}" \
    --tensor-parallel-size="${GPUS}" \
    --max-model-len="${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --port="${VLLM_PORT}" \
    --disable-custom-all-reduce \
    --enable-prefix-caching
