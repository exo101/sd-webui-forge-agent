"""
Multimodal Media 自动安装脚本
Copyright (C) 2024

此脚本用于在 WebUI 启动时自动安装 Multimodal Media 所需的依赖包

所需依赖:
- insightface: 人脸检测和分析
- onnxruntime-gpu: GPU 加速的 ONNX 运行时 (如果未安装)
- ffmpeg-python: FFmpeg Python 绑定
- torchaudio: PyTorch 音频处理

注意:
- LatentSync 是本地模块，位于 extensions/sd-webui-multimodal-media/LatentSync
- Qwen3-TTS 模型需要手动下载到 models/qwen3-tts 目录
- FFmpeg 需要单独安装并配置到系统 PATH

使用方法:
将本脚本放置在插件目录的 scripts 文件夹中，WebUI 启动时会自动执行
"""

import os
import sys
import subprocess
from pathlib import Path


def get_python_executable():
    """获取当前 Python 可执行文件路径，支持云端和本地环境"""
    python_exe = sys.executable
    if python_exe and os.path.exists(python_exe):
        return python_exe
    
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    extension_root = current_file.parent.parent  # sd-webui-multimodal-media
    project_root = extension_root.parent.parent  # sd-webui-forge-neo-v3
    
    # 定义可能的 Python 路径（云端和本地）
    python_paths = [
        # 云端环境：项目根目录下的 python
        project_root / "python" / "python.exe",
        project_root / "python" / "bin" / "python",  # Linux
        project_root / "python" / "bin" / "python3", # Linux
        
        # 本地环境：system/python
        project_root / "system" / "python" / "python.exe",
        project_root / "system" / "python" / "bin" / "python",
    ]
    
    for path in python_paths:
        if path.exists():
            return str(path)
    
    return "python"


def is_package_installed(package_name):
    """检查包是否已安装"""
    try:
        # 处理带额外说明的包名
        base_name = package_name.split('[')[0].split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
        
        # 特殊处理：某些包的导入名和包名不一致
        import_map = {
            "onnxruntime-gpu": "onnxruntime",
            "onnxruntime_cpu": "onnxruntime",
            "ffmpeg-python": "ffmpeg",
            "pillow": "PIL",
            "pytorch_lightning": "pytorch_lightning",
            "py3langid": "py3langid",
            "num2words": "num2words",
            "hangul-romanize": "hangul_romanize",
            "fugashi": "fugashi",
            "cutlet": "cutlet",
            "soundfile": "soundfile",
            "librosa": "librosa",
            "matplotlib": "matplotlib",
            "loguru": "loguru",
            "datasets": "datasets",
            "diffusers": "diffusers",
            "tqdm": "tqdm",
            "accelerate": "accelerate",
            "peft": "peft",
            "tensorboard": "tensorboard",
            "tensorboardx": "tensorboardX",
            "spacy": "spacy",
            "vector_quantize_pytorch": "vq",  # vector_quantize_pytorch 的导入名
        }
        
        # 先尝试使用映射表中的导入名
        import_name = import_map.get(base_name.lower(), base_name.replace("-", "_").replace(".", "_"))
        
        # 特殊处理：某些包需要尝试多个可能的导入名
        possible_import_names = [import_name]
        if base_name.lower() == "dashscope":
            possible_import_names = ["dashscope"]
        elif base_name.lower() == "qwen-tts":
            possible_import_names = ["qwen_tts", "qwen"]
        elif base_name.lower() == "pytorch_lightning":
            possible_import_names = ["pytorch_lightning", "lightning"]
        
        # 尝试所有可能的导入名
        for imp_name in possible_import_names:
            try:
                __import__(imp_name)
                return True
            except ImportError:
                continue
        
        return False
    except Exception:
        return False


def install_package(package_name, display_name=None):
    """使用 pip 安装包"""
    if display_name is None:
        display_name = package_name
    
    python_exe = get_python_executable()
    
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", package_name, "--upgrade"],
            check=True,
            capture_output=True,
            encoding="utf-8"
        )
        
        return result.returncode == 0
            
    except Exception as e:
        return False


def check_ffmpeg_installed():
    """检查 FFmpeg 是否已安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_dependencies():
    """安装所有必需的依赖包"""
    # 定义需要安装的 Python 包
    packages = [
        ("insightface", "insightface"),
        ("onnxruntime-gpu", "onnxruntime-gpu"),
        ("ffmpeg-python", "ffmpeg-python"),
        ("torchaudio", "torchaudio"),
        ("qwen-tts", "qwen-tts"),
        ("soundfile==0.13.1", "soundfile"),
        ("resampy", "resampy"),
        ("librosa==0.11.0", "librosa"),
        # Qwen Video 相关依赖
        ("dashscope", "dashscope"),  # 阿里云百炼 SDK
        ("pillow", "Pillow"),  # 图像处理（如果未安装）
        # IndexTTS-2 相关依赖
        ("omegaconf", "omegaconf"),  # 配置文件解析
        ("modelscope", "modelscope"),  # 魔搭社区 SDK
        ("munch", "munch"),  # 字典对象化工具 (IndexTTS-2 依赖)
        # ACE-Step 音乐生成相关依赖
        ("datasets==3.4.1", "datasets"),  # 数据集处理
        ("diffusers>=0.33.0", "diffusers"),  # 扩散模型库
        ("gradio", "gradio"),  # UI 框架
        ("loguru==0.7.3", "loguru"),  # 日志库
        ("matplotlib==3.10.1", "matplotlib"),  # 绘图库
        ("numpy", "numpy"),  # 数值计算
        ("pypinyin==0.53.0", "pypinyin"),  # 拼音转换
        ("pytorch_lightning==2.5.1", "pytorch_lightning"),  # PyTorch Lightning
        ("torch", "torch"),  # PyTorch
        ("torchvision", "torchvision"),  # 计算机视觉库
        ("tqdm", "tqdm"),  # 进度条
        ("transformers==4.50.0", "transformers"),  # 预训练模型库
        ("py3langid==0.3.0", "py3langid"),  # 语言检测
        ("hangul-romanize==0.1.0", "hangul-romanize"),  # 韩文罗马化
        ("num2words==0.5.14", "num2words"),  # 数字转单词
        ("spacy==3.8.4", "spacy"),  # NLP 库
        ("accelerate==1.6.0", "accelerate"),  # 分布式训练
        ("cutlet", "cutlet"),  # 日文罗马化
        ("fugashi[unidic-lite]", "fugashi"),  # 日文分词
        ("click", "click"),  # 命令行工具
        ("peft", "peft"),  # 参数高效微调
        ("tensorboard", "tensorboard"),  # 可视化工具
        ("tensorboardX", "tensorboardX"),  # TensorBoard 扩展
        # ACE-Step 1.5 新增依赖
        ("flash-attn", "flash-attn"),  # Flash Attention
        ("vector_quantize_pytorch", "vector_quantize_pytorch"),  # 向量量化（XL 模型依赖）
        ("sentencepiece", "sentencepiece"),  # 分词工具
        ("python-dotenv", "python-dotenv"),  # 环境变量加载
        ("aiofiles", "aiofiles"),  # 异步文件操作
        ("aiohttp", "aiohttp"),  # 异步 HTTP 请求
        ("fastapi", "fastapi"),  # API 框架
        ("uvicorn", "uvicorn"),  # ASGI 服务器
        ("pydantic", "pydantic"),  # 数据验证
        ("starlette", "starlette"),  # Web 框架
    ]
    
    installed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for package, display_name in packages:
        if is_package_installed(package.split('[')[0]):
            skipped_count += 1
            continue
        
        python_exe = get_python_executable()
        
        try:
            # 直接运行，不捕获输出，避免编码问题
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package, "--upgrade"],
                check=True
            )
            
            if result.returncode == 0:
                installed_count += 1
            else:
                failed_count += 1
                
        except subprocess.CalledProcessError:
            failed_count += 1
        except Exception:
            failed_count += 1
    
    # ⚠️ 重要：不再将 FFmpeg/SoX 作为硬性依赖
    # 这些工具应该在运行时动态检测，缺失时使用 Python 库替代或提示用户
    
    # 显示汇总
    if failed_count > 0 or installed_count > 0:
        print(f"\n{'='*60}")
        print("Multimodal Media dependency check complete")
        print(f"Newly installed: {installed_count}")
        print(f"Already exists: {skipped_count}")
        
        if failed_count > 0:
            print(f"Failed: {failed_count}")
            print("\nPlease manually install failed packages:")
            for package, display_name in packages:
                if not is_package_installed(package.split('[')[0]):
                    print(f"  python -m pip install {package}")
        elif installed_count > 0:
            print("Python dependencies installed successfully!")
        
        print(f"{'='*60}\n")
    
    # ✅ 只要 Python 依赖安装成功就返回成功
    return failed_count == 0


if __name__ == "__main__":
    print("\n" + "="*60)
    print("[INSTALL] Multimodal Media Plugin - Auto Install Script")
    print("="*60)
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    print("="*60 + "\n")
    
    # 执行安装
    success = install_dependencies()
    
    if success:
        print("[OK] All dependencies installed successfully, starting WebUI...\n")
    else:
        print("[WARN] Some dependencies are missing, but will continue to start WebUI...\n")
        print("Some features may not work. Please manually install missing dependencies in WebUI.\n")
