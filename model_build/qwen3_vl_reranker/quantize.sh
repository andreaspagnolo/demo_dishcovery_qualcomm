#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_F16.gguf OUTPUT_Q8.gguf [LLAMA_QUANTIZE]" >&2
  exit 2
fi

input_model=$1
output_model=$2
quantize_bin=${3:-/home/ubuntu/llama.cpp/build/bin/llama-quantize}

"${quantize_bin}" \
  --token-embedding-type F16 \
  --tensor-type 'cls\.output\.weight=F16' \
  "${input_model}" \
  "${output_model}" \
  Q8_0 \
  8

