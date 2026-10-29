#!/bin/bash
# Local open-weight LLM for the grading loop: llama.cpp built for the CPU and one Qwen2.5 Instruct model, both pinned.
#
#   bash tools/setup_llm.sh all 7b        build llama.cpp and download the 7B (about 4.7 GB)
#   bash tools/setup_llm.sh build         llama.cpp only (tag LLAMA_TAG, checked against its commit)
#   bash tools/setup_llm.sh download 1.5b one model only (about 1.1 GB); every file is checked against its SHA-256
#   bash tools/setup_llm.sh path 7b       the model file to serve (first part of a split model)
#   bash tools/setup_llm.sh alias 7b      the name the server reports (CIERRA_MODEL)
#
# Everything goes to llm/ (git-ignored). Cierra ships no weights. The models are Apache-2.0 (Qwen2.5-7B-Instruct and
# Qwen2.5-1.5B-Instruct model cards); llama.cpp is MIT. Needs git, cmake, a C++ compiler and curl. JOBS sets the build
# parallelism (default 4).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLM=$ROOT/llm
LLAMA_TAG=b11580
LLAMA_COMMIT=4413a192410fad1f054e9216114087a3e01527a0

# model -> "repo|alias|file sha256 [file sha256 ...]"
model_spec() {
  case "$1" in
    7b) echo "Qwen/Qwen2.5-7B-Instruct-GGUF|qwen2.5-7b-instruct-q4_k_m|\
qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db \
qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf 539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a" ;;
    1.5b) echo "Qwen/Qwen2.5-1.5B-Instruct-GGUF|qwen2.5-1.5b-instruct-q4_k_m|\
qwen2.5-1.5b-instruct-q4_k_m.gguf 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e" ;;
    *) echo "unknown model '$1' (7b or 1.5b)" >&2; exit 2 ;;
  esac
}

build() {
  if [ ! -d "$LLM/llama.cpp/.git" ]; then
    mkdir -p "$LLM"
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLM/llama.cpp"
  fi
  local head
  head=$(git -C "$LLM/llama.cpp" rev-parse HEAD)
  [ "$head" = "$LLAMA_COMMIT" ] || { echo "llama.cpp is at $head, expected $LLAMA_TAG ($LLAMA_COMMIT)" >&2; exit 1; }
  cmake -S "$LLM/llama.cpp" -B "$LLM/llama.cpp/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLM/llama.cpp/build" --target llama-server -j "${JOBS:-4}"
  echo "llama-server: $LLM/llama.cpp/build/bin/llama-server"
}

download() {
  local spec repo files
  spec=$(model_spec "$1")
  repo=${spec%%|*}
  files=${spec##*|}
  mkdir -p "$LLM/models"
  set -- $files
  while [ $# -ge 2 ]; do
    local f=$1 sum=$2
    shift 2
    if [ ! -f "$LLM/models/$f" ]; then
      curl -fL --retry 3 -o "$LLM/models/$f.part" "https://huggingface.co/$repo/resolve/main/$f"
      mv "$LLM/models/$f.part" "$LLM/models/$f"
    fi
    echo "$sum  $LLM/models/$f" | sha256sum -c - || { echo "hash mismatch: $f (delete it and retry)" >&2; exit 1; }
  done
}

case "${1:-}" in
  all) build; download "${2:-7b}" ;;
  build) build ;;
  download) download "${2:-7b}" ;;
  path) spec=$(model_spec "${2:-7b}"); f=${spec##*|}; echo "$LLM/models/${f%% *}" ;;
  alias) spec=$(model_spec "${2:-7b}"); s=${spec#*|}; echo "${s%%|*}" ;;
  *) echo "usage: $0 all|build|download|path|alias [7b|1.5b]" >&2; exit 2 ;;
esac
