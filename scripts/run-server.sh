#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/qwen3.8-sglang"
api_key_file="${state_dir}/api-key"
cuda_home="${repo_root}/.venv/lib/python3.12/site-packages/nvidia/cu13"

# The CUDA 13 wheels provide versioned runtime libraries under lib/, while
# SGLang and FlashInfer JIT builds expect unversioned names under lib64/.
mkdir -p "${cuda_home}/lib64"
for cuda_library in libcudart libcublas libcublasLt; do
  cuda_library_link="${cuda_home}/lib64/${cuda_library}.so"
  if [[ ! -e "${cuda_library_link}" && ! -L "${cuda_library_link}" ]]; then
    ln -s "../lib/${cuda_library}.so.13" "${cuda_library_link}"
  fi
done
unset cuda_library cuda_library_link

export CUDA_HOME="${cuda_home}"
export PATH="${cuda_home}/bin:${PATH}"
export LIBRARY_PATH="${cuda_home}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${cuda_home}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "${state_dir}"
chmod 700 "${state_dir}"

if [[ ! -s "${api_key_file}" ]]; then
  openssl rand -hex -out "${api_key_file}" 32
  chmod 600 "${api_key_file}"
fi

IFS= read -r api_key < "${api_key_file}"
export SGLANG_API_KEY="${api_key}"
unset api_key

exec "${repo_root}/.venv/bin/python" "${repo_root}/scripts/launch.py" \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b-nvfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 \
  --attention-backend flashinfer \
  --max-running-requests 1 \
  --cuda-graph-max-bs-decode 1 \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 5 \
  --mamba-ssm-dtype float32 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --log-level warning \
  --host 127.0.0.1 \
  --port 30000
