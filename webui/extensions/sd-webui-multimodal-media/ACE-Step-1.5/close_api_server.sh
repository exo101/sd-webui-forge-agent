#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
	./close_api_server.sh [--port PORT] [--pid PID] [--force]

Defaults:
	PORT: 8001

Behavior:
	- If --pid is provided, stops that PID.
	- Otherwise, finds the listening PID(s) on --port and stops them.
	- By default, only stops processes whose cmdline contains "uvicorn" or "acestep.api_server".
		Use --force to skip this safety check.
EOF
}

PORT="8002"
PID=""
FORCE="0"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--port)
			PORT="${2:-}"; shift 2 ;;
		--pid)
			PID="${2:-}"; shift 2 ;;
		--force)
			FORCE="1"; shift ;;
		-h|--help)
			usage; exit 0 ;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 2
			;;
	esac
done

if [[ -n "$PORT" ]] && ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
	echo "Invalid --port: $PORT" >&2
	exit 2
fi
if [[ -n "$PID" ]] && ! [[ "$PID" =~ ^[0-9]+$ ]]; then
	echo "Invalid --pid: $PID" >&2
	exit 2
fi

_cmdline() {
	local pid="$1"
	if [[ -r "/proc/${pid}/cmdline" ]]; then
		tr '\0' ' ' < "/proc/${pid}/cmdline" | sed 's/[[:space:]]\+/ /g' || true
	else
		echo ""
	fi
}

_is_target_process() {
	local pid="$1"
	local cmd
	cmd="$(_cmdline "$pid")"
	[[ "$cmd" == *"uvicorn"* || "$cmd" == *"acestep.api_server"* ]]
}

_find_pids_by_port() {
	local port="$1"
	local pids=""

	if command -v lsof >/dev/null 2>&1; then
		pids="$(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
	elif command -v ss >/dev/null 2>&1; then
		# 输出示例：LISTEN 0 4096 127.0.0.1:8001 ... users:("python",pid=12345,fd=3)
		pids="$(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ' || true)"
	elif command -v netstat >/dev/null 2>&1; then
		# 输出示例：tcp ... LISTEN 12345/python
		pids="$(netstat -lntp 2>/dev/null | awk -v p=":${port}" '$4 ~ p && $6=="LISTEN" {split($7,a,"/"); if (a[1] ~ /^[0-9]+$/) print a[1]}' | sort -u | tr '\n' ' ' || true)"
	elif command -v fuser >/dev/null 2>&1; then
		pids="$(fuser -n tcp "$port" 2>/dev/null | tr '\n' ' ' || true)"
	fi

	echo "$pids"
}

_stop_pid() {
	local pid="$1"

	if ! kill -0 "$pid" 2>/dev/null; then
		echo "PID $pid not running."
		return 0
	fi

	if [[ "$FORCE" != "1" ]] && ! _is_target_process "$pid"; then
		echo "Skip PID $pid (cmdline does not look like uvicorn/acestep.api_server). Use --force to stop anyway." >&2
		echo "cmdline: $(_cmdline "$pid")" >&2
		return 3
	fi

	echo "Stopping PID $pid..."
	kill -TERM "$pid" 2>/dev/null || true

	for _ in $(seq 1 30); do
		if ! kill -0 "$pid" 2>/dev/null; then
			echo "Stopped PID $pid."
			return 0
		fi
		sleep 0.2
	done

	echo "PID $pid did not exit; sending SIGKILL..." >&2
	kill -KILL "$pid" 2>/dev/null || true
	sleep 0.1
	if kill -0 "$pid" 2>/dev/null; then
		echo "Failed to kill PID $pid." >&2
		return 1
	fi
	echo "Killed PID $pid."
	return 0
}

if [[ -n "$PID" ]]; then
	_stop_pid "$PID"
	exit $?
fi

pids="$(_find_pids_by_port "$PORT")"
if [[ -z "${pids// }" ]]; then
	echo "No listening process found on port $PORT."
	exit 0
fi

rc=0
for pid in $pids; do
	if [[ -n "$pid" ]]; then
		_stop_pid "$pid" || rc=$?
	fi
done

exit "$rc"
