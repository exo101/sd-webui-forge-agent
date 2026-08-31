#!/usr/bin/env bash

# Load optional settings
if [ -f "webui.settings.sh" ]; then
    source webui.settings.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect Python: prefer portable Python, fallback to system Python
if [ -z "$PYTHON" ]; then
    if [ -x "$SCRIPT_DIR/../system/python/python.exe" ]; then
        PYTHON="$SCRIPT_DIR/../system/python/python.exe"
        echo "Using portable Python: $PYTHON"
    else
        PYTHON="python3"
        echo "Using system Python: $(which $PYTHON 2>/dev/null || echo $PYTHON)"
    fi
fi

SD_WEBUI_RESTART="tmp/restart"
SKIP_VENV=1
PYTHONIOENCODING=utf-8
ERROR_REPORTING=FALSE

mkdir -p tmp

# Check python
if "$PYTHON" -c "" >tmp/stdout.txt 2>tmp/stderr.txt; then
    :
else
    echo "Couldn't launch python"
    echo "If you are using portable deployment, make sure Python is in system/python/"
    goto_show_logs=1
fi

# Check pip
if [ -z "$goto_show_logs" ]; then
    if "$PYTHON" -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt; then
        :
    else
        if uv help pip >tmp/stdout.txt 2>tmp/stderr.txt; then
            :
        else
            echo "Couldn't launch pip"
            goto_show_logs=1
        fi
    fi
fi

# Launch
if [ -z "$goto_show_logs" ]; then
    "$PYTHON" launch.py "$@"

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