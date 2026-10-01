#!/usr/bin/env bash

# Load optional settings
if [ -f "webui.settings.sh" ]; then
    source webui.settings.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 自动安装 Python 3.13 ──
install_python313() {
    echo ">>> Python 3.13 未找到，正在自动安装..."

    # 方案 1：conda 安装（优先）
    if command -v conda >/dev/null 2>&1; then
        echo ">>> 通过 conda 创建 webui313 环境（Python 3.13）..."
        conda create -n webui313 python=3.13 -y
        if [ $? -eq 0 ]; then
            CONDA_BASE="$(conda info --base)"
            if [ -x "$CONDA_BASE/envs/webui313/bin/python" ]; then
                echo ">>> conda 安装完成：$CONDA_BASE/envs/webui313/bin/python"
                echo "$CONDA_BASE/envs/webui313/bin/python"
                return 0
            fi
        fi
        echo ">>> conda 安装失败，尝试下载便携版 Python..."
    fi

    # 方案 2：下载独立便携版 Python 3.13 到项目目录
    PYTHON_INSTALL_DIR="$SCRIPT_DIR/../python"
    mkdir -p "$PYTHON_INSTALL_DIR"

    PY_VERSION="3.13.12"
    ARCH="$(uname -m)"
    if [ "$ARCH" = "x86_64" ]; then
        PY_TARBALL="Python-${PY_VERSION}.tgz"
        PY_URL="https://www.python.org/ftp/python/${PY_VERSION}/${PY_TARBALL}"
    else
        echo ">>> 不支持的架构：$ARCH，请手动安装 Python 3.13"
        return 1
    fi

    echo ">>> 下载 Python ${PY_VERSION} 源码..."
    cd /tmp
    if ! wget -q "$PY_URL" -O "$PY_TARBALL" 2>/dev/null && ! curl -sL "$PY_URL" -o "$PY_TARBALL" 2>/dev/null; then
        echo ">>> 下载失败，请检查网络或手动安装 Python 3.13"
        return 1
    fi

    echo ">>> 解压并编译 Python ${PY_VERSION}（这可能需要几分钟）..."
    tar -xzf "$PY_TARBALL"
    cd "Python-${PY_VERSION}"
    ./configure --prefix="$PYTHON_INSTALL_DIR" --enable-optimizations >/dev/null 2>&1
    make -j"$(nproc)" >/dev/null 2>&1
    make install >/dev/null 2>&1

    if [ -x "$PYTHON_INSTALL_DIR/bin/python3.13" ]; then
        echo ">>> Python 安装完成：$PYTHON_INSTALL_DIR/bin/python3.13"
        echo "$PYTHON_INSTALL_DIR/bin/python3.13"
        return 0
    fi

    echo ">>> Python 编译安装失败"
    return 1
}

# ── Python 自动检测（优先便携版 → conda 3.13 → 系统 python3.13 → 自动安装）──
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
        CONDA_BASE="$(conda info --base 2>/dev/null)"
        # 优先 webui313 环境
        if [ -x "$CONDA_BASE/envs/webui313/bin/python" ]; then
            echo "$CONDA_BASE/envs/webui313/bin/python"
            return
        fi
        # 其次查找任意 3.13 环境
        for env in $(conda env list 2>/dev/null | awk '{print $1}'); do
            py="$CONDA_BASE/envs/$env/bin/python"
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

    # 5. 项目内 python 目录
    if [ -x "$SCRIPT_DIR/../python/bin/python3.13" ]; then
        echo "$SCRIPT_DIR/../python/bin/python3.13"
        return
    fi

    # 6. 自动安装 Python 3.13
    echo ">>> 未找到 Python 3.13，开始自动安装..." >&2
    INSTALLED_PY="$(install_python313)"
    if [ -n "$INSTALLED_PY" ] && [ -x "$INSTALLED_PY" ]; then
        echo "$INSTALLED_PY"
        return
    fi

    # 7. 最终回退 python3（可能版本不满足，会有警告）
    echo ">>> 自动安装失败，回退到系统 python3（版本可能不兼容）" >&2
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
