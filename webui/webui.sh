#!/usr/bin/env bash

# Load optional settings
if [ -f "webui.settings.sh" ]; then
    source webui.settings.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Python 自动检测（优先便携版 → conda 3.13 → 系统 python3.13 → python3）──
find_python313() {
    # 1. 便携 Python（Windows 部署）
    if [ -x "$SCRIPT_DIR/../system/python/python.exe" ]; then
        echo "$SCRIPT_DIR/../system/python/python.exe"
        return
    fi

    # 2. 用户显式指定
    if [ -n "$PYTHON" ]; then
        echo "$PYTHON"
        return
    fi

    # 3. 查找 conda 环境中的 Python 3.13
    if command -v conda >/dev/null 2>&1; then
        # 优先 webui313 环境
        if [ -x "$(conda info --base 2>/dev/null)/envs/webui313/bin/python" ]; then
            echo "$(conda info --base)/envs/webui313/bin/python"
            return
        fi
        # 其次查找任意 3.13 环境
        for env in $(conda env list 2>/dev/null | awk '{print $1}'); do
            py="$(conda info --base 2>/dev/null)/envs/$env/bin/python"
            if [ -x "$py" ] && "$py" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,13) else 1)" 2>/dev/null; then
                echo "$py"
                return
            fi
        done
    fi

    # 4. 系统 python3.13
    if command -v python3.13 >/dev/null 2>&1; then
        echo "$(which python3.13)"
        return
    fi

    # 5. 回退：项目内 python 目录
    if [ -x "$SCRIPT_DIR/../python/bin/python3.13" ]; then
        echo "$SCRIPT_DIR/../python/bin/python3.13"
        return
    fi

    # 6. 最终回退 python3
    echo "python3"
}

PYTHON="$(find_python313)"
echo "Using Python: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo 'unknown'))"

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
    echo "Detected Python: $PYTHON"
    echo "To install Python 3.13 via conda, run:"
    echo "  conda create -n webui313 python=3.13 -y"
    echo "  conda activate webui313"
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
