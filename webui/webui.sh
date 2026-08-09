#!/usr/bin/env bash

set -e

# Load optional settings
if [ -f "webui.settings.sh" ]; then
    source webui.settings.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 优先检测项目自带的便携版 Python（嵌入式模式）
if [ -x "$SCRIPT_DIR/python/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/python/bin/python3"
    echo "Using portable Python: $PYTHON"
elif [ -x "$SCRIPT_DIR/python/python3" ]; then
    PYTHON="$SCRIPT_DIR/python/python3"
    echo "Using portable Python: $PYTHON"
else
    # 如果没有便携 Python，使用系统 Python
    PYTHON="${PYTHON:-python3}"
    echo "Using system Python: $(which $PYTHON 2>/dev/null || echo $PYTHON)"
fi

VENV_DIR="${VENV_DIR:-$(cd "$(dirname "$0")" && pwd)/venv}"
SD_WEBUI_RESTART="tmp/restart"
ERROR_REPORTING="FALSE"

mkdir -p tmp

# Check python
if uv help python >tmp/stdout.txt 2>tmp/stderr.txt; then
    :
elif "$PYTHON" -c "" >tmp/stdout.txt 2>tmp/stderr.txt; then
    :
else
    echo "Couldn't launch python"
    goto_show_logs=1
fi

# Check pip
if [ -z "$goto_show_logs" ]; then
    if uv help pip >tmp/stdout.txt 2>tmp/stderr.txt; then
        :
    elif "$PYTHON" -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt; then
        :
    else
        echo "Couldn't launch pip"
        goto_show_logs=1
    fi
fi

# Venv handling - 跳过虚拟环境创建，直接使用当前 Python
if [ -z "$goto_show_logs" ]; then
    # 如果使用便携 Python，跳过虚拟环境创建
    if [ -x "$SCRIPT_DIR/python/bin/python3" ] || [ -x "$SCRIPT_DIR/python/python3" ]; then
        echo "Using portable Python, skipping venv creation"
    else
        # 云端模式：也跳过 venv 创建，直接使用系统 Python
        echo "Using system Python directly, skipping venv creation"
    fi
fi

# Launch
if [ -z "$goto_show_logs" ]; then
    "$PYTHON" launch.py "$@" --api --listen --cors-allow-origins=*

    if [ -f "$SD_WEBUI_RESTART" ]; then
        exec "$0" "$@"
    fi

    exit 0
fi

# Show logs
echo
echo "exit code: $?"

if [ -s tmp/stdout.txt ]; then
    echo
    echo "stdout:"
    cat tmp/stdout.txt
fi

if [ -s tmp/stderr.txt ]; then
    echo
    echo "stderr:"
    cat tmp/stderr.txt
fi

echo
echo "Launch Unsuccessful! Exiting..."
exit 1