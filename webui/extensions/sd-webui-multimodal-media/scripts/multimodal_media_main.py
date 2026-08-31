import gradio as gr
from modules import script_callbacks
from pathlib import Path
import sys
import os

# 添加当前插件目录到 Python 路径
plugin_dir = Path(__file__).parent
if str(plugin_dir) not in sys.path: 
    sys.path.insert(0, str(plugin_dir))

# 自动安装依赖
def auto_install_dependencies():
    """自动安装 Multimodal Media 所需的依赖包"""
    # 定义需要检查的包 - 格式：{导入名：pip 包名}
    required_packages = {
        "insightface": "insightface",           # 人脸检测
        "onnxruntime": "onnxruntime-gpu",       # GPU 加速 ONNX 运行时（注意：导入名是 onnxruntime）
        "ffmpeg": "ffmpeg-python",              # FFmpeg Python 绑定（注意：导入名是 ffmpeg）
        "torchaudio": "torchaudio",             # PyTorch 音频处理
        "qwen_tts": "qwen-tts",                 # Qwen3-TTS 语音合成
        "soundfile": "soundfile",               # 音频文件读写
        "resampy": "resampy",                   # 高质量音频重采样
        "librosa": "librosa",                   # 音频分析和处理
        # Qwen Video 相关依赖
        "dashscope": "dashscope",               # 阿里云百炼 SDK
        "PIL": "Pillow",                        # 图像处理（注意：导入名是 PIL）
    }
    
    missing_packages = []
    
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nMultimodal Media: Found {len(missing_packages)} missing dependencies, installing automatically...")
        
        # 执行安装脚本
        install_script = plugin_dir / "install_dependencies.py"
        if install_script.exists():
            import subprocess
            python_exe = sys.executable
            
            try:
                # 不捕获输出，让 pip 的输出直接显示到控制台
                result = subprocess.run(
                    [python_exe, str(install_script)],
                    check=True,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                # 检查是否成功
                if result.returncode == 0:
                    pass  # 成功时不显示额外信息
                else:
                    print(f"\nMultimodal Media dependency installation failed, exit code: {result.returncode}")
                    print("Please manually install:")
                    for pkg in missing_packages:
                        print(f"  python -m pip install {pkg}")
                    
            except subprocess.CalledProcessError as e:
                print(f"\nMultimodal Media dependency installation failed: {e}")
                print("Please manually install:")
                for pkg in missing_packages:
                    print(f"  python -m pip install {pkg}")
            except Exception as e:
                print(f"\nMultimodal Media dependency installation error: {e}")
                print("Please manually install:")
                for pkg in missing_packages:
                    print(f"  python -m pip install {pkg}")
        else:
            print(f"\nMultimodal Media: Install script not found")
            print("Please manually install missing dependencies:")
            for pkg in missing_packages:
                print(f"  python -m pip install {pkg}")
    
    # ⚠️ 重要：不再检查系统级工具（FFmpeg、SoX），改为在运行时动态检测
    # 这样即使没有这些工具，插件也能正常加载和显示
    print("✅ Multimodal Media Python dependencies loaded (system tools like FFmpeg/SoX are optional)")


def multimodal_media_tab():
    """创建多模态媒体功能标签页"""
    with gr.Blocks(analytics_enabled=False, elem_id="multimodal_media_container") as ui:
        gr.Markdown("""
        ## 🎬 Multimodal Media - 多媒体处理工具
        提供语音合成、视频生成和视频分析功能
        """)
        
        with gr.Tabs(elem_id="multimodal_media_tabs"):
            # Qwen3-TTS 语音合成标签页
            with gr.TabItem("1. Qwen3-TTS 语音合成"):
                try:
                    from scripts.qwen3_tts_ui import create_qwen3_tts_ui
                    # 创建并添加 Qwen3-TTS 功能
                    qwen3_tts_ui = create_qwen3_tts_ui()
                except Exception as e:
                    gr.Markdown(f"❌ Qwen3-TTS 模块初始化错误：{e}")
                    import traceback
                    traceback.print_exc()

            # 视频关键帧提取标签页
            with gr.TabItem("2. 视频关键帧提取"):
                try:
                    from scripts.video_frame_extractor import create_video_frame_extractor
                    # 创建并添加视频分帧组件
                    video_frame_components = create_video_frame_extractor()
                    
                    # 将视频分帧组件解包
                    video_input = video_frame_components["video_input"]
                    frame_output = video_frame_components["frame_output"]
                    frame_quality = video_frame_components["frame_quality"]
                    frame_mode = video_frame_components["frame_mode"]
                    frame_preview = video_frame_components["frame_preview"]
                    extract_video_frames = video_frame_components["extract_video_frames"]
                    
                    # 绑定按钮点击事件
                    extract_button = gr.Button("🎬 提取关键帧")
                    extract_button.click(
                        fn=extract_video_frames,
                        inputs=[video_input, frame_output, frame_quality, frame_mode],
                        outputs=[gr.File(label="提取的帧文件"), frame_preview]
                    )
                except Exception as e:
                    gr.Markdown(f"❌ 视频分帧模块初始化错误：{e}")
                    import traceback
                    traceback.print_exc()

            # Kling 可灵视频生成标签页
            with gr.TabItem("3. Kling 可灵视频生成"):
                try:
                    from scripts.kling_video.main_ui import create_kling_video_gen_ui
                    # 创建 Kling 可灵视频生成 UI 组件
                    kling_video_gen_ui = create_kling_video_gen_ui()
                except Exception as e:
                    gr.Markdown(f"❌ Kling 可灵视频模块初始化错误：{e}")
                    import traceback
                    traceback.print_exc()

            # ACE-Step 音乐生成标签页
            with gr.TabItem("5. ACE-Step 音乐生成"):
                try:
                    from scripts.ace_step_ui import create_ace_step_ui
                    # 创建并添加 ACE-Step 音乐生成功能
                    ace_step_ui = create_ace_step_ui()
                except Exception as e:
                    gr.Markdown(f"❌ ACE-Step 模块初始化错误：{e}")
                    import traceback
                    traceback.print_exc()
    
    # ✅ 正确的返回值格式：返回元组列表
    return [(ui, "多媒体处理", "multimodal_media_tab")]


def on_app_started(demo=None, app=None):
    """在应用启动时执行依赖安装"""
    # 先执行依赖安装（静默模式）
    auto_install_dependencies()
    
    print("✅ Multimodal Media 插件已准备就绪")


# 注册 UI 标签页
script_callbacks.on_ui_tabs(multimodal_media_tab)