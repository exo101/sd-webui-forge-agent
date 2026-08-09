"""
打包脚本 - 将 launcher 打包为单个 exe
需要先安装: pip install pyinstaller PyQt6 pyyaml
运行: python build.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(SCRIPT_DIR)  # 输出到项目根目录

EXCLUDES = [
    "torch", "torchvision", "torchaudio",
    "numpy", "matplotlib", "scipy",
    "PIL", "cv2", "sklearn", "pandas",
    "IPython", "jupyter", "notebook",
]
exclude_args = []
for m in EXCLUDES:
    exclude_args += ["--exclude-module", m]

ICON = os.path.join(SCRIPT_DIR, "forge_neo.ico")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",
    "--name", "forge_neo启动器",
    "--distpath", OUTPUT_DIR,
    "--workpath", os.path.join(SCRIPT_DIR, "build_tmp"),
    "--specpath", SCRIPT_DIR,
    *( ["--icon", ICON] if os.path.exists(ICON) else [] ),
    *exclude_args,
    os.path.join(SCRIPT_DIR, "main.py"),
]

out_path = os.path.join(OUTPUT_DIR, "forge_neo启动器.exe")

print("开始打包...")
print("模式: 单文件 (--onefile)")
print(" ".join(cmd))
result = subprocess.run(cmd, cwd=SCRIPT_DIR)
if result.returncode == 0:
    print(f"\n[OK] 打包成功！输出: {out_path}")
else:
    print(f"\n[FAIL] 打包失败，退出码: {result.returncode}")
