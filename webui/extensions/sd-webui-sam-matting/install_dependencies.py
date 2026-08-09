"""
SAM Matting 自动安装脚本
Copyright (C) 2024

此脚本用于在 WebUI 启动时自动安装 SAM Matting 所需的依赖包

所需依赖:
- rembg: 背景移除工具
- onnxruntime-gpu: GPU 加速的 ONNX 运行时
- litelama: 轻量级图像修复模型
- segment-anything: Meta 的分割一切模型

使用方法:
将本脚本放置在插件目录的 scripts 文件夹中，WebUI 启动时会自动执行
"""

import os
import sys
import subprocess
from pathlib import Path


def get_python_executable():
    """获取当前 Python 可执行文件路径"""
    python_exe = sys.executable
    if not python_exe:
        # 尝试常见路径
        venv_dir = Path(__file__).parent.parent.parent / "venv"
        if venv_dir.exists():
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = "python"
    return python_exe


def is_package_installed(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def install_package(package_name, display_name=None):
    """使用 pip 安装包"""
    if display_name is None:
        display_name = package_name
    
    print(f"\n{'='*60}")
    print(f"📦 正在安装 {display_name}...")
    print(f"{'='*60}")
    
    python_exe = get_python_executable()
    
    try:
        # 使用当前 Python 环境安装
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", package_name, "--upgrade"],
            check=True,
            capture_output=False,
            encoding="utf-8"
        )
        
        if result.returncode == 0:
            print(f"✅ {display_name} 安装成功！")
            return True
        else:
            print(f"❌ {display_name} 安装失败")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装 {display_name} 时出错：{e}")
        return False
    except Exception as e:
        print(f"❌ 安装 {display_name} 时发生错误：{e}")
        return False


def install_dependencies():
    """安装所有必需的依赖包"""
    print("\n" + "="*60)
    print("🔧 SAM Matting 依赖安装程序")
    print("="*60)
    
    # 定义需要安装的包及其显示名称
    packages = [
        ("rembg", "rembg (背景移除工具)"),
        ("onnxruntime-gpu", "onnxruntime-gpu (GPU 版 ONNX 运行时)"),
        ("litelama", "litelama (轻量级图像修复)"),
        ("segment-anything", "segment-anything (Meta 分割一切)"),
    ]
    
    installed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for package, display_name in packages:
        # 检查是否已安装
        if is_package_installed(package.split('[')[0]):  # 处理 extras
            print(f"\n✓ {display_name} 已安装，跳过")
            skipped_count += 1
            continue
        
        # 安装包
        if install_package(package, display_name):
            installed_count += 1
        else:
            failed_count += 1
            print(f"\n⚠️  {display_name} 安装失败，您可以手动安装:")
            print(f"   python -m pip install {package}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 安装汇总")
    print("="*60)
    print(f"✅ 新安装：{installed_count} 个")
    print(f"⏭️  已存在：{skipped_count} 个")
    print(f"❌ 失败：{failed_count} 个")
    
    if failed_count > 0:
        print(f"\n⚠️  有 {failed_count} 个包安装失败，请检查:")
        print("  1. 网络连接是否正常")
        print("  2. pip 源是否可访问（可配置国内镜像源）")
        print("  3. Python 版本是否兼容")
        print("\n手动安装命令:")
        for package, display_name in packages:
            if not is_package_installed(package.split('[')[0]):
                print(f"  python -m pip install {package}")
    else:
        print("\n🎉 所有依赖安装完成！")
        print("\n💡 提示：如果是首次安装，建议重启 WebUI 以确保所有模块正确加载")
    
    print("="*60 + "\n")
    
    return failed_count == 0


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SAM Matting 插件 - 自动安装脚本")
    print("="*60)
    print(f"Python 版本：{sys.version}")
    print(f"Python 路径：{sys.executable}")
    print(f"工作目录：{os.getcwd()}")
    print("="*60 + "\n")
    
    # 执行安装
    success = install_dependencies()
    
    if success:
        print("✨ 安装过程顺利完成，即将启动 WebUI...\n")
    else:
        print("⚠️  部分依赖安装失败，但将继续启动 WebUI...\n")
        print("您可以在 WebUI 启动后，打开终端手动安装失败的包\n")
