import gradio as gr
from modules import shared
from modules.paths_internal import default_output_dir
import os
import json
import time
from pathlib import Path
import warnings
import shutil
import datetime
import subprocess

# ============================================
# 关键环境变量设置（必须在导入任何音频处理库之前）
# ============================================
# 强制 transformers 使用系统 FFmpeg 而不是 torchcodec
os.environ["TRANSFORMERS_AUDIO_BACKEND"] = "ffmpeg"
# 强制 torchaudio 使用 ffmpeg 后端
os.environ["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"

# 忽略所有与音频处理相关的警告
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================
# FFmpeg 路径配置
# ============================================
def _configure_ffmpeg_path():
    """配置 FFmpeg 路径以确保系统可以找到 ffmpeg 和 ffprobe"""
    # 首先检查 ffmpeg 是否已经在 PATH 中
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, check=True)
        if result.returncode == 0:
            print(f"[FFmpeg配置] ✓ 系统 PATH 中已找到 FFmpeg")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # 优先检查 WebUI 环境中的 FFmpeg
    webui_root = Path(__file__).parent.parent.parent.parent
    webui_ffmpeg_paths = [
        webui_root / "ffmpeg" / "bin",
        webui_root / "venv" / "Scripts",
        webui_root / "_deps" / "ffmpeg" / "bin",
    ]
    
    # 常见的系统 FFmpeg 安装路径（Windows）
    windows_ffmpeg_paths = [
        "C:\\ffmpeg\\bin",
        "C:\\Program Files\\ffmpeg\\bin",
        "C:\\Program Files (x86)\\ffmpeg\\bin",
    ]
    
    # 常见的系统 FFmpeg 安装路径（Linux）
    linux_ffmpeg_paths = [
        "/usr/bin",
        "/usr/local/bin",
        "/opt/bin",
    ]
    
    # 根据操作系统选择路径
    if os.name == 'nt':
        system_ffmpeg_paths = windows_ffmpeg_paths
    else:
        system_ffmpeg_paths = linux_ffmpeg_paths
    
    # 合并所有路径，优先搜索 WebUI 环境
    all_ffmpeg_paths = webui_ffmpeg_paths + [Path(p) for p in system_ffmpeg_paths]
    
    # 尝试查找 FFmpeg
    system_path = os.environ.get("PATH", "")
    for ffmpeg_path in all_ffmpeg_paths:
        if isinstance(ffmpeg_path, Path):
            ffmpeg_path = str(ffmpeg_path)
        
        if os.path.exists(ffmpeg_path):
            # 确定 ffmpeg 和 ffprobe 的文件名
            if os.name == 'nt':
                ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")
                ffprobe_exe = os.path.join(ffmpeg_path, "ffprobe.exe")
            else:
                ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg")
                ffprobe_exe = os.path.join(ffmpeg_path, "ffprobe")
            
            # 检查 ffmpeg 和 ffprobe 是否存在
            has_ffmpeg = os.path.exists(ffmpeg_exe)
            has_ffprobe = os.path.exists(ffprobe_exe)
            
            if has_ffmpeg and has_ffprobe:
                # 将路径添加到系统 PATH 最前面
                if system_path:
                    os.environ["PATH"] = ffmpeg_path + os.pathsep + system_path
                else:
                    os.environ["PATH"] = ffmpeg_path
                print(f"[FFmpeg配置] ✓ 已找到 FFmpeg: {ffmpeg_path}")
                return True
    
    # 如果都没找到，尝试使用 ffmpeg-python
    try:
        import ffmpeg
        print(f"[FFmpeg配置] ✓ 使用 ffmpeg-python")
        return True
    except ImportError:
        pass
    
    print("[FFmpeg配置] ⚠️ 未找到 FFmpeg，音频处理功能可能受限")
    return False

# 执行 FFmpeg 配置
_configure_ffmpeg_path()

# 获取 Qwen3-TTS模型路径
qwen_tts_path = os.path.join(shared.models_path, "qwen3-tts")
model_dir = qwen_tts_path  # 直接使用 models/qwen3-tts 目录，不需要 checkpoints 子目录
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'qwen3_tts')

# 默认输出目录
default_qwen_tts_output = os.path.join(default_output_dir, "qwen3-tts")

# 确保目录存在
os.makedirs(qwen_tts_path, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

# 全局模型实例
qwen_tts_model = None

def send_audio_to_storyboard(audio_path, description=""):
    """
    将生成的音频发送到分镜助手
    
    Args:
        audio_path: 音频文件路径
        description: 分镜描述（可选）
    
    Returns:
        dict: {
            'success': bool,  # 成功标志
            'message': str,   # 消息
            'index': int,     # 分镜索引（从 0 开始）
            'total_count': int,  # 总分镜数
            'target_page': int   # 应该在第几页显示（最后一页）
        }
    """
    try:
        # 获取分镜数据目录 - 使用多种方法确保路径正确
        current_file = Path(__file__).resolve()
        
        # 方法 1: 通过 workspace 根目录构建
        workspace_root = current_file.parent.parent.parent
        data_dir = workspace_root / "sd-webui-MultiModal" / "scripts" / "storyboard_data"
        
        # 调试信息
        print(f"\n[调试] 当前文件：{current_file}")
        print(f"[调试] workspace_root: {workspace_root}")
        print(f"[调试] workspace_root 存在：{workspace_root.exists()}")
        print(f"[调试] 目标数据目录 (方法 1): {data_dir}")
        
        # 验证关键目录是否存在
        multimodal_dir = workspace_root / "sd-webui-MultiModal"
        scripts_dir = multimodal_dir / "scripts"
        
        print(f"[调试] MultiModal 目录：{multimodal_dir}, 存在：{multimodal_dir.exists()}")
        print(f"[调试] scripts 目录：{scripts_dir}, 存在：{scripts_dir.exists()}")
        
        if not multimodal_dir.exists():
            raise FileNotFoundError(f"❌ sd-webui-MultiModal 扩展目录不存在：{multimodal_dir}\n请确保已安装该扩展")
        
        if not scripts_dir.exists():
            raise FileNotFoundError(f"❌ scripts 目录不存在：{scripts_dir}")
        
        # 使用 as_posix() 确保路径分隔符统一，然后创建目录
        data_dir_str = str(data_dir)
        print(f"[调试] 目标目录字符串：{data_dir_str}")
        
        # 确保目录存在（使用 parents=True 递归创建所有父目录）
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[调试] ✅ 数据目录已准备就绪：{data_dir}\n")
        
        storyboard_file = data_dir / "storyboard.json"
        
        # 加载分镜数据
        def load_storyboard_data():
            if storyboard_file.exists():
                try:
                    with open(storyboard_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            return []
                        return json.loads(content)
                except Exception as e:
                    print(f"⚠️ 加载分镜数据失败：{e}")
                    return []
            return []
        
        # 保存分镜数据
        def save_storyboard_data(data):
            try:
                with open(storyboard_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"❌ 保存分镜数据失败：{e}")
                return False
        
        # 复制音频文件到分镜临时目录
        def process_audio_for_storyboard(audio_path):
            """将音频复制到分镜临时目录"""
            try:
                if audio_path is None or not os.path.exists(audio_path):
                    return None
                
                # 复制到临时目录
                output_dir = data_dir / "temp_audios"
                output_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"storyboard_audio_{timestamp}.wav"
                
                shutil.copy2(audio_path, output_path)
                return str(output_path)
            
            except Exception as e:
                print(f"⚠️ 处理音频失败：{e}")
                import traceback
                traceback.print_exc()
                return None
        
        # 处理音频
        processed_path = process_audio_for_storyboard(audio_path)
        if not processed_path:
            return "❌ 音频处理失败：无法复制音频文件"
        
        # 双重验证：确保 processed_path 是有效的文件路径
        if not isinstance(processed_path, str) or not os.path.isfile(processed_path):
            print(f"❌ 路径验证失败：processed_path={processed_path}, 类型={type(processed_path)}")
            return "❌ 音频处理失败：生成的路径无效"
        
        # 加载现有数据
        storyboard_data = load_storyboard_data()
        
        # 计算新的分镜索引
        new_index = len(storyboard_data)
        
        # 添加新的分镜记录（包含音频）
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_storyboard = {
            "id": new_index,
            "image_path": None,  # 没有图片
            "audio_path": processed_path,  # 新增音频路径
            "aspect_ratio": "16:9 (宽屏)",
            "description": description if description else "",
            "timestamp": timestamp
        }
        
        storyboard_data.append(new_storyboard)
        
        # 保存数据
        success = save_storyboard_data(storyboard_data)
        
        # 计算分页信息
        STORYBOARDS_PER_PAGE = 9  # 每页 9 个宫格
        total_count = len(storyboard_data)
        target_page = max(1, (total_count + STORYBOARDS_PER_PAGE - 1) // STORYBOARDS_PER_PAGE)
        
        if success:
            return f"✅ 音频已添加到分镜 #{new_index + 1}（第 {target_page} 页）"
        else:
            return "❌ 添加失败：无法保存分镜数据"
    
    except Exception as e:
        print(f"❌ 发送到分镜失败：{e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f"❌ 错误：{str(e)}",
            'index': -1,
            'total_count': 0,
            'target_page': 1
        }

def open_output_directory(output_dir_path=None):
    """
    打开输出目录
    注意：为了避免客户端缓存污染问题，始终使用动态计算的路径
    """
    try:
        import subprocess
        import sys
        
        # 关键修复：不使用传入的缓存路径，而是动态计算当前正确的输出目录
        # 这样可以避免浏览器localStorage中存储的旧路径（如v2版本的路径）
        from modules.paths_internal import default_output_dir
        current_output_dir = os.path.join(default_output_dir, "qwen3-tts")
        
        # 确保目录存在
        os.makedirs(current_output_dir, exist_ok=True)
        
        # 根据操作系统打开目录
        if sys.platform == "win32":
            # Windows
            subprocess.run(["explorer", current_output_dir], check=True)
        elif sys.platform == "darwin":
            # macOS
            subprocess.run(["open", current_output_dir], check=True)
        else:
            # Linux
            subprocess.run(["xdg-open", current_output_dir], check=True)
        
        return f"✓ 已打开输出目录：{current_output_dir}"
    except Exception as e:
        error_msg = f"打开目录失败：{str(e)}"
        print(error_msg)
        return error_msg

def initialize_qwen_tts_model(model_name):
    """
    初始化 Qwen3-TTS 模型
    """
    global qwen_tts_model
    try:
        import torch
        
        # 在导入 qwen_tts 之前抑制 sox（因为 qwen_tts 可能依赖 sox）
        import warnings
        import logging
        warnings.filterwarnings("ignore", category=UserWarning, module="sox")
        logging.getLogger("sox").setLevel(logging.CRITICAL)
        
        # 首先检查 qwen_tts 库的兼容性
        try:
            from qwen_tts import Qwen3TTSModel
        except TypeError as e:
            if "check_model_inputs" in str(e):
                error_msg = (
                    "❌ qwen_tts 库兼容性问题！\n\n"
                    f"错误详情：{str(e)}\n\n"
                    "解决方案：\n"
                    "1. 升级 qwen-tts 库到最新版本：\n"
                    "   pip install --upgrade qwen-tts\n\n"
                    "2. 如果已最新版，尝试重新安装：\n"
                    "   pip uninstall -y qwen-tts\n"
                    "   pip install qwen-tts\n\n"
                    "3. 检查 Python 版本（推荐 3.10-3.12）：\n"
                    f"   当前 Python 版本：{torch.__version__ if hasattr(torch, '__version__') else '未知'}\n\n"
                    "4. 重启 WebUI 后重试"
                )
                print(error_msg)
                return error_msg
            raise
        
        # 模型映射 - 支持多个版本
        model_map = {
            "Base": {
                "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            },
            "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        }
        
        if model_name not in model_map:
            return f"错误：未知的模型类型 {model_name}"
        
        # 处理 Base模型的多个版本
        if model_name == "Base":
            # 优先检查本地存在的版本
            base_versions = ["0.6B", "1.7B"]
            found_version = None
            
            for version in base_versions:
                test_path = os.path.join(model_dir, f"Qwen3-TTS-12Hz-{version}-{model_name}")
                if os.path.exists(test_path):
                    found_version = version
                    remote_model_path = model_map[model_name][version]
                    print(f"✓ 检测到 Base模型版本：{version}")
                    break
            
            if not found_version:
                # 默认使用 1.7B
                remote_model_path = model_map["Base"]["1.7B"]
                print("未找到本地 Base模型，将尝试加载 1.7B 版本")
        else:
            remote_model_path = model_map[model_name]
        
        # 检查本地是否有模型 - 增强的路径匹配逻辑
        local_model_path = os.path.join(model_dir, model_name)
        
        # 优先使用本地模型，支持多种目录结构
        possible_local_paths = []
        
        # 根据已确定的 remote_model_path 构建可能的本地路径
        if isinstance(model_map[model_name], dict):
            # Base模型有多个版本，需要检查所有版本
            for version in ["0.6B", "1.7B"]:
                # 完整 HuggingFace 格式（优先级最高）
                possible_local_paths.append(os.path.join(model_dir, f"Qwen3-TTS-12Hz-{version}-{model_name}"))
                # HuggingFace 路径格式转换
                hf_path = model_map[model_name][version]
                possible_local_paths.append(os.path.join(model_dir, hf_path.replace("/", "_")))
                possible_local_paths.append(os.path.join(model_dir, hf_path.replace("/", "--")))
        else:
            # 其他单一版本模型
            # 完整 HuggingFace 格式（优先级最高）
            possible_local_paths.append(os.path.join(model_dir, f"Qwen3-TTS-12Hz-1.7B-{model_name}"))
            # HuggingFace 路径格式转换
            possible_local_paths.append(os.path.join(model_dir, remote_model_path.replace("/", "_")))
            possible_local_paths.append(os.path.join(model_dir, remote_model_path.replace("/", "--")))
        
        # 添加简化格式路径
        possible_local_paths.extend([
            local_model_path,
            os.path.join(model_dir, model_name.lower()),
            os.path.join(model_dir, f"Qwen3-TTS-{model_name}"),
        ])
        
        found_local = False
        for path in possible_local_paths:
            if os.path.exists(path):
                local_model_path = path
                found_local = True
                print(f"✓ 找到本地模型路径：{local_model_path}")
                break
        
        if found_local:
            model_path = local_model_path
            print(f"将使用本地模型加载")
        else:
            model_path = remote_model_path
            print(f"警告：本地未找到模型，将尝试从远程加载")
            print(f"搜索的本地路径：{local_model_path}")
            print(f"远程模型ID: {remote_model_path}")
            print(f"\n提示：请确保模型文件在以下目录之一:")
            for path in possible_local_paths:
                print(f"  - {path}")
        
        print(f"\n正在加载 Qwen3-TTS-{model_name} 模型...")
        print(f"加载路径：{model_path}")
        
        # 准备加载参数 - 始终使用离线模式
        load_kwargs = {
            "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            "local_files_only": True,  # 优先使用本地文件
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        
        # 仅当 CUDA 可用时设置 dtype 和 attention 实现
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            # 检查是否支持 flash attention
            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("启用 Flash Attention 2 加速")
            except ImportError:
                load_kwargs["attn_implementation"] = "sdpa"
                print("未检测到 Flash Attention，使用 SDPA 注意力机制")
        else:
            load_kwargs["torch_dtype"] = torch.float32
        
        # 加载模型
        try:
            model = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
        except Exception as local_load_error:
            # 如果本地加载失败，且找到了本地路径，说明是格式问题
            if found_local:
                print(f"本地模型加载失败：{str(local_load_error)}")
                print(f"\n请检查模型目录结构是否正确:")
                print(f"模型目录应包含：config.json, model.safetensors, tokenizer_config.json 等文件")
                raise
            # 否则可能是网络问题，给出明确提示
            else:
                error_str = str(local_load_error)
                if "connect" in error_str.lower() or "timeout" in error_str.lower() or "network" in error_str.lower():
                    raise ConnectionError(
                        f"无法连接到 HuggingFace 服务器。\n"
                        f"请手动下载模型到本地目录"
                    )
                else:
                    raise
        
        qwen_tts_model = {
            "model": model,
            "name": model_name
        }
        
        print(f"\n✓ Qwen3-TTS-{model_name} 模型加载完成！")
        return f"成功加载 Qwen3-TTS-{model_name} 模型"
        
    except ConnectionError as e:
        import traceback
        error_msg = f"网络连接错误：{str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg
    except Exception as e:
        import traceback
        error_msg = f"模型加载失败：{str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def transcribe_audio(audio_path):
    """
    使用 Whisper 自动识别参考音频中的文本
    返回识别的文本内容
    """
    try:
        # 声明全局变量
        global whisper_model, whisper_processor
        
        # 设置环境变量，强制使用系统 FFmpeg 而不是 torchcodec
        os.environ["TRANSFORMERS_AUDIO_BACKEND"] = "ffmpeg"
        
        import torch
        import numpy as np
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        # 导入 librosa 用于音频重采样
        try:
            import librosa
        except ImportError:
            librosa = None
        
        # 手动加载音频文件，避免使用 torchcodec
        def load_audio_file(file_path):
            """手动加载音频文件为 numpy 数组"""
            try:
                # 尝试使用 soundfile
                import soundfile as sf
                audio_data, sample_rate = sf.read(file_path)
                return audio_data, sample_rate
            except ImportError:
                pass
            
            # 尝试使用 librosa
            try:
                import librosa
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                return audio_data, sample_rate
            except ImportError:
                pass
            
            # 尝试使用 subprocess + ffmpeg
            try:
                import subprocess
                import tempfile
                
                # 使用 ffmpeg 转换为 WAV 格式并读取
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_path = temp_wav.name
                
                cmd = [
                    "ffmpeg", "-i", file_path,
                    "-f", "wav", "-ar", "16000",
                    "-ac", "1", "-y", temp_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                
                # 使用 wave 模块读取 WAV
                import wave
                with wave.open(temp_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)
                
                # 转换为 numpy 数组
                import struct
                if sample_width == 1:
                    fmt = f"{n_frames * n_channels}B"
                    audio_data = np.array(struct.unpack(fmt, audio_bytes)) / 255.0
                elif sample_width == 2:
                    fmt = f"{n_frames * n_channels}h"
                    audio_data = np.array(struct.unpack(fmt, audio_bytes)) / 32767.0
                else:
                    fmt = f"{n_frames * n_channels}f"
                    audio_data = np.array(struct.unpack(fmt, audio_bytes))
                
                # 转换为单声道
                if n_channels > 1:
                    audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
                
                # 删除临时文件
                os.unlink(temp_path)
                
                return audio_data, sample_rate
            except Exception as e:
                print(f"音频加载失败: {e}")
                raise
        
        # 检查是否已有 Whisper 模型实例
        if 'whisper_model' not in globals() or whisper_model is None:
            print("正在加载 Whisper 模型进行语音识别...")
            
            model_id = "openai/whisper-tiny"  # 使用轻量级模型
            
            # 检查本地是否有 Whisper 模型 - 支持多个可能的位置
            possible_paths = [
                os.path.join(shared.models_path, "whisper-tiny"),  # models/whisper-tiny
                os.path.join(shared.models_path, "whisper", "whisper-tiny"),  # models/whisper/whisper-tiny
                os.path.join(shared.models_path, "ASR", "whisper-tiny"),  # models/ASR/whisper-tiny (通用 ASR 模型目录)
            ]
            
            # 优先使用本地模型
            use_local = False
            local_model_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    local_model_path = path
                    use_local = True
                    print(f"✓ 找到本地 Whisper 模型：{local_model_path}")
                    break
            
            if not use_local:
                print(f"警告：本地未找到 Whisper 模型，将尝试从远程下载")
                print(f"远程模型 ID: {model_id}")
                print(f"建议手动下载模型到以下位置之一:")
                for path in possible_paths:
                    print(f"  - {path}")
                print(f"\n下载地址：https://huggingface.co/openai/whisper-tiny")
                local_model_path = model_id
            else:
                print(f"将使用本地模型加载")
            
            # 加载处理器和模型
            print(f"\n正在加载 Whisper 处理器...")
            processor = AutoProcessor.from_pretrained(
                local_model_path,
                local_files_only=use_local,  # 本地模式
            )
            
            print(f"正在加载 Whisper 模型权重...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=use_local,  # 本地模式
            )
            
            if torch.cuda.is_available():
                model.to("cuda")
            
            # 不使用 pipeline，直接保存模型和处理器
            whisper_model = model
            whisper_processor = processor
            print("Whisper 模型加载完成！\n")
        
        # 手动加载音频文件，避免使用 torchcodec
        print(f"正在加载音频文件: {audio_path}")
        audio_array, sample_rate = load_audio_file(audio_path)
        
        # 确保音频是一维数组（Whisper 要求）
        if audio_array.ndim > 1:
            print(f"转换为单声道: {audio_array.shape} -> 一维")
            audio_array = audio_array.mean(axis=1)
        
        # 确保采样率正确（Whisper 需要 16kHz）
        if sample_rate != 16000:
            print(f"重新采样音频: {sample_rate} -> 16000")
            if librosa is not None:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            else:
                # 使用 numpy 手动重采样
                import math
                duration = len(audio_array) / sample_rate
                new_length = int(duration * 16000)
                old_indices = np.arange(len(audio_array))
                new_indices = np.linspace(0, len(audio_array) - 1, new_length)
                audio_array = np.interp(new_indices, old_indices, audio_array)
            sample_rate = 16000
        
        # 使用处理器处理音频
        print("正在处理音频...")
        input_features = whisper_processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_features
        
        # 确保输入数据类型与模型一致（模型使用 float16）
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        input_features = input_features.to(dtype)
        
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        
        # 执行语音识别
        print("正在执行语音识别...")
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        
        # 解码结果
        recognized_text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        
        print(f"语音识别结果：{recognized_text}")
        return recognized_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"语音识别失败：{str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查模型文件是否完整")
        print("2. 确保模型目录包含：config.json, model.safetensors, preprocessor_config.json 等文件")
        print("3. 重启 WebUI 后重试")
        return ""

def generate_speech_base(text, language, ref_audio_path, ref_text, output_dir, use_batch_mode=False, auto_transcribe=False):
    """
    Base 模型 - 语音克隆功能
    支持单次和批量推理，支持自动语音识别
    """
    global qwen_tts_model
    
    if qwen_tts_model is None or qwen_tts_model["name"] != "Base":
        msg = initialize_qwen_tts_model("Base")
        if not msg.startswith("成功"):
            return None, msg
    
    try:
        import torch
        import soundfile as sf
        
        # 如果启用自动识别且未提供参考文本，则自动识别
        actual_ref_text = ref_text
        if auto_transcribe and not ref_text.strip():
            print("正在自动识别参考音频文本...")
            actual_ref_text = transcribe_audio(ref_audio_path)
            
            if not actual_ref_text:
                return None, "语音识别失败，请手动输入参考音频文本"
            
            print(f"✓ 自动识别文本：{actual_ref_text}")
        
        model = qwen_tts_model["model"]
        
        # 打印调试信息
        print(f"\n=== Base模型生成参数 ===")
        print(f"文本：{text[:50]}...")
        print(f"语言：{language}")
        print(f"参考音频：{ref_audio_path}")
        print(f"参考文本：{actual_ref_text[:50] if actual_ref_text else 'None'}...")
        print(f"========================\n")
        
        # 生成语音克隆
        with torch.no_grad():
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=actual_ref_text,
            )
            
            # 保存音频文件
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            
            if use_batch_mode and len(wavs) > 1:
                # 批量模式：保存多个文件
                output_files = []
                for i, wav in enumerate(wavs):
                    output_filename = os.path.join(output_dir, f"speech_base_clone_{timestamp}_{i}.wav")
                    sf.write(output_filename, wav, sr)
                    output_files.append(output_filename)
                return output_files[0], f"批量语音克隆成功！已保存 {len(wavs)} 个文件到：{output_dir}"
            else:
                # 单次模式：保存一个文件
                output_filename = os.path.join(output_dir, f"speech_base_clone_{timestamp}.wav")
                sf.write(output_filename, wavs[0], sr)
                return output_filename, f"语音克隆成功！已保存到：{output_filename}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"语音克隆失败：{str(e)}"

def generate_speech_customvoice(text, language, speaker, instruct, output_dir, use_batch_mode=False):
    """
    CustomVoice 模型 - 自定义音色功能
    支持 9 种预设说话人和批量推理
    """
    global qwen_tts_model
    
    if qwen_tts_model is None or qwen_tts_model["name"] != "CustomVoice":
        msg = initialize_qwen_tts_model("CustomVoice")
        if not msg.startswith("成功"):
            return None, msg
    
    try:
        import torch
        import soundfile as sf
        
        model = qwen_tts_model["model"]
        
        # 打印调试信息
        print(f"\n=== CustomVoice 生成参数 ===")
        print(f"文本：{text[:50]}...")
        print(f"语言：{language}")
        print(f"说话人：{speaker}")
        print(f"语气指令：{instruct}")
        print(f"========================\n")
        
        # 生成自定义音色
        with torch.no_grad():
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
            
            # 保存音频文件
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            
            if use_batch_mode and len(wavs) > 1:
                # 批量模式：保存多个文件
                output_files = []
                for i, wav in enumerate(wavs):
                    output_filename = os.path.join(output_dir, f"speech_custom_{timestamp}_{i}.wav")
                    sf.write(output_filename, wav, sr)
                    output_files.append(output_filename)
                return output_files[0], f"批量自定义音色成功！已保存 {len(wavs)} 个文件到：{output_dir}"
            else:
                # 单次模式：保存一个文件
                output_filename = os.path.join(output_dir, f"speech_custom_{timestamp}.wav")
                sf.write(output_filename, wavs[0], sr)
                return output_filename, f"自定义音色生成成功！已保存到：{output_filename}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"自定义音色生成失败：{str(e)}"

def generate_speech_voicedesign(text, language, instruct, output_dir, use_batch_mode=False):
    """
    VoiceDesign 模型 - 声音设计功能
    支持基于描述的精细控制和批量推理
    """
    global qwen_tts_model
    
    if qwen_tts_model is None or qwen_tts_model["name"] != "VoiceDesign":
        msg = initialize_qwen_tts_model("VoiceDesign")
        if not msg.startswith("成功"):
            return None, msg
    
    try:
        import torch
        import soundfile as sf
        
        model = qwen_tts_model["model"]
        
        # 打印调试信息
        print(f"\n=== VoiceDesign生成参数 ===")
        print(f"文本：{text[:50]}...")
        print(f"语言：{language}")
        print(f"音色描述：{instruct[:100] if instruct else 'None'}...")
        print(f"===========================\n")
        
        # 生成声音设计
        with torch.no_grad():
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            
            # 保存音频文件
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            
            if use_batch_mode and len(wavs) > 1:
                # 批量模式：保存多个文件
                output_files = []
                for i, wav in enumerate(wavs):
                    output_filename = os.path.join(output_dir, f"speech_design_{timestamp}_{i}.wav")
                    sf.write(output_filename, wav, sr)
                    output_files.append(output_filename)
                return output_files[0], f"批量声音设计成功！已保存 {len(wavs)} 个文件到：{output_dir}"
            else:
                # 单次模式：保存一个文件
                output_filename = os.path.join(output_dir, f"speech_design_{timestamp}.wav")
                sf.write(output_filename, wavs[0], sr)
                return output_filename, f"声音设计生成成功！已保存到：{output_filename}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"声音设计生成失败：{str(e)}"

def generate_speech(text, language, voice_style, model_choice, output_dir, use_batch_mode=False):
    """
    生成语音 - 统一入口函数
    根据选择的模型类型调用相应的生成函数
    """
    if model_choice == "Base":
        return generate_speech_base(text, language, None, voice_style, output_dir, use_batch_mode)
    elif model_choice == "CustomVoice":
        # 对于 CustomVoice，voice_style 作为 speaker 名称
        return generate_speech_customvoice(text, language, voice_style, "", output_dir, use_batch_mode)
    elif model_choice == "VoiceDesign":
        # 对于 VoiceDesign，voice_style 作为 instruct 描述
        return generate_speech_voicedesign(text, language, voice_style, output_dir, use_batch_mode)
    else:
        return None, f"不支持的模型类型：{model_choice}"

def save_voice_preset(preset_name, voice_style, language, model_choice):
    """
    保存音色预设
    """
    try:
        preset_file = os.path.join(config_dir, f"{preset_name}.json")
        preset_data = {
            "name": preset_name,
            "voice_style": voice_style,
            "language": language,
            "model": model_choice,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, ensure_ascii=False, indent=2)
        
        return f"音色预设 '{preset_name}' 保存成功！"
    except Exception as e:
        return f"保存失败：{str(e)}"

def load_voice_presets():
    """
    加载已保存的音色预设列表
    """
    try:
        presets = []
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.json'):
                    preset_name = file[:-5]  # 去掉 .json 后缀
                    presets.append(preset_name)
        return presets
    except Exception as e:
        return []

def load_preset_data(preset_name):
    """
    加载指定音色预设的数据
    """
    try:
        preset_file = os.path.join(config_dir, f"{preset_name}.json")
        if os.path.exists(preset_file):
            with open(preset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "speaker": data.get("speaker", "Vivian"),
                "instruct": data.get("instruct", ""),
                "language": data.get("language", "Chinese"),
                "model": data.get("model", "CustomVoice")
            }
        return {"speaker": "Vivian", "instruct": "", "language": "Chinese", "model": "CustomVoice"}
    except Exception as e:
        return {"speaker": "Vivian", "instruct": "", "language": "Chinese", "model": "CustomVoice"}

def create_qwen3_tts_ui():
    """
    创建 Qwen3-TTS 语音合成 UI
    """
    import time
    
    with gr.Blocks(analytics_enabled=False) as ui:
        # 模型选择
        with gr.Row():
            model_choice = gr.Dropdown(
                label="选择模型",
                choices=[
                    ("Base - 基础模型（支持音频克隆）", "Base"),
                    ("CustomVoice - 自定义音色（9 种预设）", "CustomVoice"),
                    ("VoiceDesign - 声音设计（精细控制）", "VoiceDesign")
                ],
                value="Base",
                info="Base: 基础通用 | CustomVoice: 9 种预设音色 | VoiceDesign: 描述生成"
            )
        
        # Base 模型专用组件
        with gr.Group(visible=True) as base_group:
            gr.Markdown("### 📢 语音克隆模式（Base）")
            ref_audio_input = gr.Audio(
                label="参考音频（3 秒左右）", 
                type="filepath"
            )
            
            with gr.Row():
                auto_transcribe_checkbox = gr.Checkbox(
                    label="🎤 自动识别音频文本（推荐）",
                    value=True,
                    info="启用后将使用 AI 自动识别参考音频中的文字，无需手动输入"
                )
            
            ref_text_input = gr.Textbox(
                label="参考音频文本（可选）",
                placeholder="如果不启用自动识别，请手动输入参考音频对应的文本内容...",
                lines=2,
                info="参考音频中说的内容，用于帮助模型学习音色。启用自动识别后可留空"
            )
        
        # CustomVoice 模型专用组件
        with gr.Group(visible=False) as customvoice_group:
            gr.Markdown("### 🎤 自定义音色模式（CustomVoice）")
            speaker_dropdown = gr.Dropdown(
                label="选择说话人",
                choices=[
                    ("Vivian - 明亮、略带锐气的年轻女声（中文）", "Vivian"),
                    ("Serena - 温暖柔和的年轻女声（中文）", "Serena"),
                    ("Uncle_Fu - 音色低沉醇厚的成熟男声（中文）", "Uncle_Fu"),
                    ("Dylan - 清晰自然的北京青年男声（中文·北京方言）", "Dylan"),
                    ("Eric - 活泼、略带沙哑明亮感的成都男声（中文·四川方言）", "Eric"),
                    ("Ryan - 富有节奏感的动态男声（英语）", "Ryan"),
                    ("Aiden - 清晰中频的阳光美式男声（英语）", "Aiden"),
                    ("Ono_Anna - 轻快灵活的俏皮日语女声（日语）", "Ono_Anna"),
                    ("Sohee - 富含情感的温暖韩语女声（韩语）", "Sohee"),
                ],
                value="Vivian",
                info="建议选择对应说话人的母语以获得最佳音质"
            )
            custom_instruct = gr.Textbox(
                label="语气指令（可选）",
                placeholder="例如：用特别愤怒的语气说、非常开心地说、低声细语...",
                lines=2,
                info="控制说话的语气和情感，留空则使用自然语气"
            )
        
        # VoiceDesign 模型专用组件
        with gr.Group(visible=False) as voicedesign_group:
            gr.Markdown("### 🎨 声音设计模式（VoiceDesign）")
            design_instruct = gr.Textbox(
                label="音色描述",
                placeholder="例如：体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
                lines=3,
                info="详细描述你想要的音色特征，包括年龄、性别、情感、语调等"
            )
        
        # 通用组件
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="请输入要合成的文本（支持多语言，最多 2000 字符）",
                    lines=4,
                    max_lines=8
                )
                
                language = gr.Dropdown(
                    label="语言",
                    choices=[
                        ("中文", "Chinese"),
                        ("英文", "English"),
                        ("日文", "Japanese"),
                        ("韩文", "Korean"),
                        ("德文", "German"),
                        ("法文", "French"),
                        ("俄文", "Russian"),
                        ("葡萄牙文", "Portuguese"),
                        ("西班牙文", "Spanish"),
                        ("意大利文", "Italian"),
                    ],
                    value="Chinese",
                    info="Auto 可自动检测，但建议明确指定以获得最佳效果"
                )
            
            with gr.Column(scale=1):
                # 输出目录打开按钮（不显示路径文本框，避免浏览器缓存问题）
                with gr.Group():
                    open_dir_btn = gr.Button(
                        "📁 打开输出目录",
                        variant="secondary",
                        size="sm",
                        info="生成的音频文件将保存在 output/qwen3-tts 目录"
                    )
                
                batch_mode = gr.Checkbox(
                    label="批量模式（实验性）",
                    value=False,
                    info="启用后可一次性生成多个音频文件"
                )
        
        # 生成按钮
        generate_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
        
        # 结果展示
        with gr.Row():
            audio_output = gr.Audio(label="生成的音频", type="filepath")
            with gr.Column():
                status_info = gr.Textbox(
                    label="操作状态",
                    lines=2,
                    interactive=False
                )
                
                # 发送到分镜按钮
                send_to_storyboard_btn = gr.Button(
                    "📤 发送到分镜",
                    variant="secondary",
                    size="md",
                    visible=True
                )
        
        # 音色预设
        with gr.Accordion("💾 音色预设管理", open=False):
            preset_name_input = gr.Textbox(
                label="预设名称",
                placeholder="输入预设名称，如：温柔女声 - 愤怒",
                lines=1
            )
            
            with gr.Row():
                save_preset_btn = gr.Button("保存当前配置为预设", variant="secondary")
            
            preset_list = gr.Dropdown(
                label="加载已有预设",
                choices=[],
                value=None,
                info="选择预设将自动填充下方配置"
            )
        
        # 模型下载说明
        with gr.Accordion("📁 模型下载说明", open=False):
            gr.Markdown("""
            ### 模型存储位置
            - **Qwen3-TTS 模型**：`WebUI根目录/models/qwen3-tts`
            - **Whisper 模型**：`WebUI根目录/models/whisper-tiny` 或 `WebUI根目录/models/whisper/whisper-tiny`
            
            ### 模型下载
            - **首次使用**：首次生成时会自动下载所需模型
            - **下载来源**：所有模型均已包含在整合包中，如需更新请从群主网盘中获取
            - **模型大小**：约 3-4GB，请确保磁盘空间充足
            
            ### 手动下载
            如需手动下载模型，请将模型文件放入对应目录：
            - Qwen3-TTS 模型：从 Hugging Face 下载 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 等模型
            - Whisper 模型：从 Hugging Face 下载 `openai/whisper-tiny` 模型
            """)
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### Base 模型 - 语音克隆
            1. 上传 3 秒左右的参考音频
            2. ✅ **推荐**：启用"自动识别音频文本"，AI 会自动识别音频中的内容
            3. 如果不启用自动识别，需要手动输入参考音频的文本内容
            4. 输入要生成的新文本
            5. 点击生成即可克隆音色
            
            **自动识别优势**：
            - 🎯 无需手动输入，节省时间
            - 🔍 AI 精准识别，准确率高
            - 🌍 支持多语言自动检测
            - ⚡ 使用轻量级 Whisper-tiny 模型，速度快
            
            ### CustomVoice 模型 - 自定义音色
            1. 从 9 种预设说话人中选择
            2. 可选：输入语气指令控制情感
            3. 输入要生成的文本
            4. 点击生成即可获得定制化音色
            
            ### VoiceDesign 模型 - 声音设计
            1. 详细描述想要的音色特征
            2. 描述越详细，效果越精准
            3. 示例："体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"
            4. 输入要生成的文本
            5. 点击生成即可创造独特音色
            
            ### 批量推理
            - 在文本框中输入多行文本，每行作为一句
            - 启用"批量模式"复选框
            - 生成的多个音频文件会分别保存
            
            **提示**：首次生成需要下载模型（约 3-4GB），请耐心等待。
            """)
        
        # 切换模型时显示/隐藏对应组件
        def on_model_change(model_type):
            base_visible = model_type == "Base"
            custom_visible = model_type == "CustomVoice"
            design_visible = model_type == "VoiceDesign"
            
            return (
                gr.update(visible=base_visible),
                gr.update(visible=custom_visible),
                gr.update(visible=design_visible)
            )
        
        model_choice.change(
            fn=on_model_change,
            inputs=[model_choice],
            outputs=[base_group, customvoice_group, voicedesign_group]
        )
        
        # 生成逻辑
        def on_generate(text, language, model_type, ref_audio, ref_text, 
                       speaker, custom_instruct, design_instruct, 
                       batch_mode, auto_transcribe):
            if not text.strip():
                return None, "错误：请输入要合成的文本"
            
            # 关键修复：不使用UI组件传入的可能被缓存的output_dir路径
            # 而是动态计算当前正确的输出目录，避免客户端缓存污染问题
            from modules.paths_internal import default_output_dir
            current_output_dir = os.path.join(default_output_dir, "qwen3-tts")
            
            if model_type == "Base":
                if not ref_audio:
                    return None, "错误：Base 模型需要上传参考音频"
                
                # 如果未启用自动识别且没有手动输入文本，则报错
                if not auto_transcribe and not ref_text.strip():
                    return None, "错误：请启用自动识别或手动输入参考音频文本"
                
                return generate_speech_base(
                    text=text,
                    language=language,
                    ref_audio_path=ref_audio,
                    ref_text=ref_text if not auto_transcribe else "",
                    output_dir=current_output_dir,  # 使用动态计算的正确路径
                    use_batch_mode=batch_mode,
                    auto_transcribe=auto_transcribe
                )
            elif model_type == "CustomVoice":
                instruct_text = custom_instruct.strip() if custom_instruct else ""
                return generate_speech_customvoice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct_text,
                    output_dir=current_output_dir,  # 使用动态计算的正确路径
                    use_batch_mode=batch_mode
                )
            elif model_type == "VoiceDesign":
                if not design_instruct.strip():
                    return None, "错误：VoiceDesign 模型需要输入音色描述"
                return generate_speech_voicedesign(
                    text=text,
                    language=language,
                    instruct=design_instruct.strip(),
                    output_dir=current_output_dir,  # 使用动态计算的正确路径
                    use_batch_mode=batch_mode
                )
            else:
                return None, f"错误：不支持的模型类型 {model_type}"
        
        generate_btn.click(
            fn=on_generate,
            inputs=[
                text_input, language, model_choice, 
                ref_audio_input, ref_text_input,
                speaker_dropdown, custom_instruct, design_instruct,
                batch_mode, auto_transcribe_checkbox
            ],
            outputs=[audio_output, status_info]
        )
        
        # 打开输出目录按钮事件
        open_dir_btn.click(
            fn=open_output_directory,
            inputs=[],
            outputs=[status_info]
        )
        
        # 发送到分镜按钮事件
        send_to_storyboard_btn.click(
            fn=lambda audio_path: send_audio_to_storyboard(
                audio_path, 
                description="Qwen3-TTS 生成的音频"
            ),
            inputs=[audio_output],
            outputs=[status_info]
        )
        
        # 预设功能
        def update_preset_list():
            presets = load_voice_presets()
            # 添加"无"选项，用于清空当前选择
            presets_with_none = ["【无】清空选择"] + presets
            return gr.update(choices=presets_with_none, value=None)
        
        def on_preset_selected(preset_name):
            # 如果选择了"无"或空值，清空所有配置
            if not preset_name or preset_name == "【无】清空选择":
                return "Vivian", "", "Chinese", "✅ 已清空预设选择"
            
            # 否则加载预设数据
            data = load_preset_data(preset_name)
            return (
                data.get("speaker", "Vivian"),
                data.get("instruct", ""),
                data.get("language", "Chinese"),
                f"已加载预设：{preset_name}"
            )
        
        ui.load(fn=update_preset_list, outputs=[preset_list])
        
        save_preset_btn.click(
            fn=lambda name, speaker, instruct, lang, model: save_voice_preset(
                name, speaker, lang, model
            ),
            inputs=[preset_name_input, speaker_dropdown, custom_instruct, language, model_choice],
            outputs=[status_info]
        ).then(
            fn=update_preset_list,
            outputs=[preset_list]
        )
        
        preset_list.change(
            fn=on_preset_selected,
            inputs=[preset_list],
            outputs=[speaker_dropdown, custom_instruct, language, status_info]
        )
    
    return ui