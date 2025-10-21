#!/usr/bin/env bash
# Lightweight wrapper around vLLM server startup with neutral defaults.
# Configure environment variables before running to match your hardware setup.

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPUS="${GPUS:-1}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-45000}"
VLLM_PORT="${VLLM_PORT:-8080}"

vllm serve "${MODEL_NAME}" \
    --tensor-parallel-size="${GPUS}" \
    --max-model-len="${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --port="${VLLM_PORT}" \
    --disable-custom-all-reduce \
    --enable-prefix-caching
