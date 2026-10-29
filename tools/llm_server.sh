#!/bin/bash
# Local open-weight LLM for Cierra's grading loop, CPU only.
#
#   bash tools/llm_server.sh start | stop | status
#
# llama.cpp (MIT) b11580, built CPU-only (GGML_CUDA=OFF) in llm/llama.cpp; model Qwen2.5-7B-Instruct Q4_K_M
# (Apache-2.0, Qwen/Qwen2.5-7B-Instruct-GGUF @ bb5d59e) in llm/models. Serves an OpenAI-compatible API on
# 127.0.0.1:8090 (local only). CIERRA_LLM_THREADS sets the CPU threads (default 8).
# Another model: CIERRA_GGUF=llm/models/<file>.gguf CIERRA_GGUF_ALIAS=<name>. Then:
#   CIERRA_LLM_URL=http://127.0.0.1:8090/v1 CIERRA_MODEL=qwen2.5-7b-instruct-q4_k_m CIERRA_LLM_TIMEOUT=3600 \
#       python tools/live_run.py --driver openweight
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/llm/llama.cpp/build/bin/llama-server
MODEL=${CIERRA_GGUF:-$ROOT/llm/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}
ALIAS=${CIERRA_GGUF_ALIAS:-qwen2.5-7b-instruct-q4_k_m}
PIDF=$ROOT/llm/server.pid
LOG=$ROOT/llm/server.log
case "$1" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "running (pid $(cat "$PIDF"))"; exit 0; fi
    nohup "$BIN" -m "$MODEL" --host 127.0.0.1 --port 8090 -c 16384 -t ${CIERRA_LLM_THREADS:-8} -ngl 0 \
        --alias "$ALIAS" > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    for i in $(seq 1 300); do
      curl -sf http://127.0.0.1:8090/health > /dev/null && { echo "ready (pid $(cat "$PIDF"))"; exit 0; }
      sleep 2
    done
    echo "not ready after 600 s; see $LOG"; exit 1 ;;
  stop)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && rm -f "$PIDF" && echo stopped || echo "not running" ;;
  status)
    [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && echo "running (pid $(cat "$PIDF"))" || echo "not running" ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
