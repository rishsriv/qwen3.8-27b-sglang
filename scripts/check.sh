#!/usr/bin/env bash
set -euo pipefail

state_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/qwen3.8-sglang"
api_key_file="${state_dir}/api-key"
base_url="${SGLANG_BASE_URL:-http://127.0.0.1:30000/v1}"

if [[ ! -s "${api_key_file}" ]]; then
  echo "API key not found at ${api_key_file}; start the service first." >&2
  exit 1
fi

IFS= read -r api_key < "${api_key_file}"

curl --fail --silent --show-error \
  "${base_url}/models" \
  -H "Authorization: Bearer ${api_key}"
echo

curl --fail --silent --show-error \
  "${base_url}/responses" \
  -H "Authorization: Bearer ${api_key}" \
  -H "Content-Type: application/json" \
  --data '{"model":"qwen3.8-27b-nvfp4","input":"Reply with exactly: SGLang DSpark is ready.","max_output_tokens":128}'
echo
