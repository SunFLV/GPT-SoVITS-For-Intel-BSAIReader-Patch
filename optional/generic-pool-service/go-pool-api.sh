#!/usr/bin/env bash
# GPT-SoVITS 多路并发 API 启动脚本（Linux / macOS）
# 用法：./go-pool-api.sh [--pool-size 4] [--port 9881] [--api-key auto] ...
set -euo pipefail

cd "$(dirname "$0")"

# 依次尝试 venv / .venv / 系统 python，可用 POOL_PYTHON 覆盖
PY="${POOL_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    PY="$(command -v python)"
  fi
fi

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export POOL_PYTHON="$PY"

echo "============================================================"
echo "  GPT-SoVITS Multi-route TTS API (process pool)"
echo "  python : $PY"
echo "  config : ${POOL_CONFIG_FILE:-config/pool_api.json (optional)}"
echo "------------------------------------------------------------"
echo "  default : 4 workers / port 9881 / IPv6 dual-stack / warmup"
echo "  public  : ./go-pool-api.sh --api-key auto"
echo "  examples: ./go-pool-api.sh --pool-size 3 --port 9882"
echo "            ./go-pool-api.sh --no-ipv6      (IPv4 only)"
echo "            ./go-pool-api.sh --default-ref refs/speaker.wav --default-prompt '参考文本'"
echo "============================================================"

exec "$PY" api_pool.py "$@"
