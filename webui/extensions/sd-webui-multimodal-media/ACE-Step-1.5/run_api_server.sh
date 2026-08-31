#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/root/data/repo/gongjunmin/miniconda3/bin/activate}"
CONDA_ENV_NAME="${ACESTEP_CONDA_ENV:-acestep_v15_train}"

HOST="${ACESTEP_API_HOST:-0.0.0.0}"
PORT="${ACESTEP_API_PORT:-8001}"
LOG_LEVEL="${ACESTEP_API_LOG_LEVEL:-debug}"

cd "$ROOT_DIR"

# 临时关闭 nounset 以避免 conda activate.d 脚本中的 unbound variable 错误
set +u
# shellcheck disable=SC1090
source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
set -u

# NOTE: api_server 使用内存队列/任务存储，要求 workers=1。
nohup python -m uvicorn acestep.api_server:app \
	--host "0.0.0.0" \
	--port "8001" \
	--workers 1 \
	--log-level "$LOG_LEVEL" > server.log 2>&1 &
echo "Server started in background with PID $!. Logs in server.log"