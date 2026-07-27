#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:-"${root_dir}/build"}
llama_cpp_source=${LLAMA_CPP_SOURCE:-/home/ubuntu/llama.cpp}
geniex_llama_lib_dir=${GENIEX_LLAMA_LIB_DIR:-/home/ubuntu/.local/share/geniex/llama_cpp}

cmake -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_SOURCE="${llama_cpp_source}" \
  -DGENIEX_LLAMA_LIB_DIR="${geniex_llama_lib_dir}"
cmake --build "${build_dir}" --parallel

echo "Built ${build_dir}/qwen3-vl-reranker-mtmd"

