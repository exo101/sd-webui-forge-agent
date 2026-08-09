import gradio as gr
import os
import sys
import json
from pathlib import Path
from modules import shared

# 定义插件目录
plugin_dir = Path(__file__).parent.parent

# ACE-Step-1.5 代码路径（相对于插件目录）
ace_step_path = plugin_dir / "ACE-Step-1.5"
if str(ace_step_path) not in sys.path:
    sys.path.insert(0, str(ace_step_path))

# ACE-Step-1.5 模型路径 - 用户把所有模型集中放在 models/ace-step/ 目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent  # sd-webui-forge-neo-v3/webui
MODELS_DIR = WEBUI_ROOT / "models" / "ace-step"

# 设置 ACESTEP_CHECKPOINTS_DIR 环境变量，告诉原项目模型在哪里
os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(MODELS_DIR)

# 打印调试信息
print(f"[ACE-Step-1.5] Checkpoints 目录: {MODELS_DIR}")

# 模型版本配置
# display_name: UI 显示名称
# internal_name: ACE-Step 1.5 内部使用的模型名称
MODEL_VERSIONS = {
    "ACE-Step-v15-xl-turbo": {
        "repo_id": "ACE-Step/ACE-Step-v15-xl-turbo",
        "internal_name": "acestep-v15-xl-turbo",  # ACE-Step 1.5 内部使用的实际模型名称
        "quantized": False,
        "description": "ACE-Step 1.5 Turbo 版本，速度更快，质量更高"
    }
}

# 全局模型实例
ace_step_handler = None
current_model_version = None

def load_ace_step_model(model_version="ACE-Step-v15-xl-turbo"):
    """加载 ACE-Step-1.5 模型"""
    global ace_step_handler, current_model_version
    
    # 获取模型配置的内部名称
    model_config = MODEL_VERSIONS.get(model_version, {})
    internal_name = model_config.get("internal_name", model_version)
    
    # 如果模型已加载且版本相同，使用缓存
    if ace_step_handler is not None and current_model_version == internal_name:
        print(f"[ACE-Step-1.5] 模型已加载 (版本: {internal_name})")
        return ace_step_handler
    
    try:
        
        # 导入 ACE-Step-1.5 模块
        from acestep.handler import AceStepHandler
        
        # 创建 handler
        handler = AceStepHandler()
        
        # 初始化服务 - 使用原项目的自动查找机制（ACESTEP_CHECKPOINTS_DIR 已设置）
        print(f"[ACE-Step-1.5] 正在初始化模型...")
        print(f"[ACE-Step-1.5] 模型: {internal_name}")
        print(f"[ACE-Step-1.5] Checkpoints 目录: {MODELS_DIR}")
        
        # 不强制设置 offload 参数，让原项目根据显存大小自动判断
        status_msg, ok = handler.initialize_service(
            project_root=str(ace_step_path),
            config_path=internal_name,
            device="auto",
            use_flash_attention=True,
            compile_model=False,
        )
        
        if ok:
            ace_step_handler = handler
            current_model_version = internal_name
            print(f"✅ ACE-Step-1.5 模型加载成功 (版本: {internal_name})")
            return handler
        else:
            raise RuntimeError(f"模型初始化失败: {status_msg}")
    
    except Exception as e:
        print(f"❌ ACE-Step-1.5 模型加载失败: {e}")
        import traceback
        print(traceback.format_exc())
        raise

# 全局 LLM Handler（用于音频分析）
_llm_handler_instance = None
_model_version_for_llm = None  # 保存模型版本以供 LLM 初始化使用

def get_llm_handler(model_version=None):
    """获取或初始化 LLM Handler"""
    global _llm_handler_instance, _model_version_for_llm
    
    # 如果提供了新的模型版本，更新它
    if model_version:
        _model_version_for_llm = model_version
    
    if _llm_handler_instance is None:
        try:
            from acestep.llm_inference import LLMHandler
            _llm_handler_instance = LLMHandler()
            print("[ACE-Step-1.5] LLM Handler 已创建（未初始化）")
        except Exception as e:
            print(f"[ACE-Step-1.5] 创建 LLM Handler 失败: {e}")
            return None
    
    # 如果 LLM 未初始化，尝试初始化
    if not _llm_handler_instance.llm_initialized and _model_version_for_llm:
        try:
            print("[ACE-Step-1.5] 正在尝试初始化 LLM...")
            
            # 获取模型目录
            from acestep.model_downloader import get_checkpoints_dir
            import os
            
            model_dir = str(get_checkpoints_dir())
            lm_model_path = None
            
            # 查找可用的 LLM 模型
            possible_lm_names = [
                "acestep-5Hz-lm-1.7B",
                "acestep-5Hz-lm-0.6B",
                "acestep-5Hz-lm-1.7B-v4-fix",
            ]
            
            for lm_name in possible_lm_names:
                lm_path = os.path.join(model_dir, lm_name)
                if os.path.exists(lm_path):
                    lm_model_path = lm_path
                    print(f"[ACE-Step-1.5] 找到 LLM 模型: {lm_model_path}")
                    break
            
            if lm_model_path:
                # 初始化 LLM（不强制设置 offload 参数）
                status, success = _llm_handler_instance.initialize(
                    checkpoint_dir=model_dir,
                    lm_model_path=lm_model_path,
                    backend="pt",  # 使用 PyTorch 后端（兼容性最好）
                    device="auto",
                    dtype=None,
                )
                
                if success:
                    print(f"[ACE-Step-1.5] LLM 初始化成功！")
                else:
                    print(f"[ACE-Step-1.5] LLM 初始化失败: {status}")
            else:
                print("[ACE-Step-1.5] 未找到 LLM 模型，跳过初始化")
                
        except Exception as e:
            print(f"[ACE-Step-1.5] 初始化 LLM 时出错: {e}")
            import traceback
            print(traceback.format_exc())
    
    return _llm_handler_instance

def analyze_src_audio_wrapper(src_audio, model_version):
    """分析源音频，提取歌词、曲风、BPM、时长、调式、语言、拍号等信息"""
    global _llm_handler_instance
    try:
        handler = load_ace_step_model(model_version)
        
        if not src_audio:
            return "请先上传源音频", "", "", None, None, "", "", ""
        
        print(f"[ACE-Step-1.5] 开始分析源音频: {src_audio}")
        
        # 第一步：转换为 codes
        try:
            codes_string = handler.convert_src_audio_to_codes(src_audio)
            print(f"[ACE-Step-1.5] 音频代码转换成功，长度: {len(codes_string) if codes_string else 0}")
        except Exception as e:
            print(f"[ACE-Step-1.5] 音频转换失败: {e}")
            return f"音频转换失败: {str(e)}", "", "", None, None, "", "", ""
        
        if not codes_string:
            return "未能从音频中提取代码", "", "", None, None, "", "", ""
        
        # 第二步：尝试理解音乐（如果有 LLM 模型）
        caption = ""
        lyrics = ""
        bpm = None
        duration = None
        keyscale = ""
        language = ""
        timesignature = ""
        
        llm_handler = get_llm_handler(model_version)
        
        if llm_handler and llm_handler.llm_initialized:
            try:
                from acestep.inference import understand_music

                result = understand_music(
                    llm_handler=llm_handler,
                    audio_codes=codes_string,
                    temperature=0.3,  # 使用更低的温度以获得更准确的结果
                    use_constrained_decoding=True,
                    constrained_decoding_debug=False,
                )
                
                if result.success:
                    caption = result.caption or ""
                    lyrics = result.lyrics or ""
                    bpm = result.bpm
                    duration = result.duration
                    keyscale = result.keyscale or ""
                    language = result.language or ""
                    timesignature = result.timesignature or ""
                    status_msg = "✅ 音频分析完成！已提取曲风、歌词、BPM、时长等信息"
                    print(f"[ACE-Step-1.5] LLM 分析成功: BPM={bpm}, Key={keyscale}, Language={language}")
                else:
                    status_msg = "⚠️ 音频代码已生成，但 LLM 分析失败"
                    print(f"[ACE-Step-1.5] LLM 分析失败: {result.status_message}")
                    
            except Exception as e:
                status_msg = "⚠️ 音频代码已生成，但 LLM 分析出错"
                print(f"[ACE-Step-1.5] 理解音乐时出错: {e}")
        else:
            status_msg = "⚠️ 音频代码已生成，但 LLM 未初始化（需要加载 LLM 模型才能分析歌词和曲风）"
            print("[ACE-Step-1.5] LLM 未初始化，无法分析歌词和曲风")
        
        # 分析完成后卸载 LLM 模型释放显存
        if _llm_handler_instance is not None:
            print(f"[ACE-Step-1.5] 卸载 LLM 模型释放显存...")
            _llm_handler_instance.unload()
            _llm_handler_instance = None
            print(f"[ACE-Step-1.5] LLM 模型已卸载")
        
        # 清理缓存
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 打印显存状态
        if torch.cuda.is_available():
            try:
                free = torch.cuda.memory_reserved(0) / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"[ACE-Step-1.5] 显存使用分析后: {(total - free):.1f} GB / {total:.1f} GB")
            except Exception:
                pass
        
        return status_msg, caption, lyrics, bpm, duration, keyscale, language, timesignature
        
    except Exception as e:
        import traceback
        error_msg = f"分析音频失败: {str(e)}"
        print(f"[ACE-Step-1.5] {error_msg}")
        print(traceback.format_exc())
        # 即使出错也要尝试清理
        if _llm_handler_instance is not None:
            try:
                _llm_handler_instance.unload()
                _llm_handler_instance = None
            except Exception:
                pass
        return error_msg, "", "", None, None, "", "", ""

def generate_music(prompt, lyrics, duration, infer_steps, guidance_scale, model_version, bpm, key_scale, time_signature, vocal_language):
    """生成音乐（仅文本生成音乐）"""
    try:
        # 清理显存
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 打印显存状态
        if torch.cuda.is_available():
            try:
                free = torch.cuda.memory_reserved(0) / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"[ACE-Step-1.5] 显存使用生成前: {(total - free):.1f} GB / {total:.1f} GB")
            except Exception:
                pass
        
        handler = load_ace_step_model(model_version)
        
        print(f"[ACE-Step-1.5] 开始生成音乐...")
        print(f"[ACE-Step-1.5] 提示词: {prompt[:50]}..." if len(prompt) > 50 else f"[ACE-Step-1.5] 提示词: {prompt}")
        print(f"[ACE-Step-1.5] 歌词: {lyrics[:50]}..." if len(lyrics) > 50 else f"[ACE-Step-1.5] 歌词: {lyrics}")
        print(f"[ACE-Step-1.5] 推理步数: {infer_steps}, 引导强度: {guidance_scale}, 时长: {duration}秒")
        print(f"[ACE-Step-1.5] BPM: {bpm}, 调式: {key_scale}, 拍号: {time_signature}, 语言: {vocal_language}")
        
        # 准备参数
        kwargs = {
            "captions": prompt,           # 曲风/风格提示
            "lyrics": lyrics if lyrics.strip() else "",  # 歌词
            "inference_steps": infer_steps,  # 推理步数
            "guidance_scale": guidance_scale,  # 引导强度
            "use_random_seed": True,      # 使用随机种子
            "seed": -1,
            "task_type": "text2music",
            "bpm": bpm if bpm else None,
            "key_scale": key_scale if key_scale else "",
            "time_signature": time_signature if time_signature else "",
            "vocal_language": vocal_language if vocal_language else "en",
            "batch_size": 1,  # 只生成一个音频
        }
        
        # 设置时长
        if duration is not None and duration > 0:
            kwargs["audio_duration"] = duration
        else:
            kwargs["audio_duration"] = 30  # 默认 30 秒
        
        # 生成音乐（使用命名参数）
        result = handler.generate_music(**kwargs)
        
        # 打印返回结果结构用于调试
        print(f"[ACE-Step-1.5] 返回类型: {type(result)}")
        if isinstance(result, dict):
            print(f"[ACE-Step-1.5] 字典键: {result.keys()}")
            for key, value in result.items():
                print(f"[ACE-Step-1.5]   {key}: {type(value)} = {value if not hasattr(value, 'shape') else f'shape={value.shape}'}")
        
        # 处理输出 - 支持多种返回格式
        audio_tensor = None
        output_path = None
        sample_rate = 48000  # 默认采样率
        
        if isinstance(result, dict):
            # 字典格式 - ACE-Step 1.5 返回格式
            if 'audios' in result and result['audios']:
                # audios 是一个列表，取第一个
                audio_data = result['audios'][0]
                if isinstance(audio_data, dict):
                    audio_tensor = audio_data.get('tensor')
                    sample_rate = audio_data.get('sample_rate', 48000)
                elif hasattr(audio_data, 'tensor'):
                    audio_tensor = audio_data.tensor
                    sample_rate = getattr(audio_data, 'sample_rate', 48000)
            elif 'audio' in result and result['audio'] is not None:
                audio_tensor = result['audio']
            elif 'wav' in result and result['wav'] is not None:
                audio_tensor = result['wav']
            # 检查是否有输出路径
            if 'output_path' in result:
                output_path = result['output_path']
            elif 'output_paths' in result and result['output_paths']:
                output_path = result['output_paths'][0] if isinstance(result['output_paths'], list) else result['output_paths']
        elif hasattr(result, 'audio') and result.audio is not None:
            audio_tensor = result.audio
        elif hasattr(result, 'output_paths') and result.output_paths:
            output_path = result.output_paths[0] if isinstance(result.output_paths, list) else result.output_paths
        
        # 如果有输出路径但没有音频张量，读取文件
        if audio_tensor is None and output_path:
            if os.path.exists(output_path):
                import torchaudio
                audio_tensor, _ = torchaudio.load(output_path)
                print(f"[ACE-Step-1.5] 从文件读取音频: {output_path}")
            else:
                raise ValueError(f"音频文件不存在: {output_path}")
        
        if audio_tensor is None:
            raise ValueError(f"无法从结果中提取音频")
        
        # 保存音频
        output_path = os.path.join(shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples, "ace-step-output.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用标准 wave 模块保存（避免 torchcodec 依赖）
        import wave
        import numpy as np
        
        # 转换为 numpy 数组
        if hasattr(audio_tensor, 'cpu'):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = np.array(audio_tensor)
        
        # 检查形状，处理可能的通道维度
        print(f"[ACE-Step-1.5] 音频张量形状: {audio_np.shape}")
        
        # 如果是双通道，转换为单通道
        if len(audio_np.shape) > 1:
            if audio_np.shape[0] > 1:
                audio_np = audio_np.mean(axis=0)
        
        # 确保是一维数组
        audio_np = np.squeeze(audio_np)
        print(f"[ACE-Step-1.5] 处理后音频形状: {audio_np.shape}")
        
        # 归一化到 [-1, 1]
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val
            print(f"[ACE-Step-1.5] 归一化因子: {max_val}")
        
        # 转换为 int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # 保存为 WAV 文件（使用模型原生采样率）
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)  # 使用模型原生采样率
            wf.writeframes(audio_int16.tobytes())
        
        print(f"✅ 音乐生成成功！文件已保存到: {output_path} (采样率: {sample_rate} Hz)")
        return output_path, None
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 音乐生成失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def create_ace_step_ui():
    """创建 ACE-Step-1.5 音乐生成 UI"""
    
    # 示例歌词和提示词
    EXAMPLE_LYRICS = """[第一节]
一剑霜寒照九州
半生风雨踏清秋
马蹄踏碎红尘路
恩怨未休
情字难收
[第二节]
青山隐隐水悠悠
红袖添香为谁留
江湖纵有千般险
一念温柔
便胜所有
[副歌]
刀光剑影
藏不住眼底温柔
策马天涯
忘不了你回眸
[结尾]
一剑 一酒 一知己
一生 一世 一双人"""

    EXAMPLE_PROMPT = """E minor (E小调)
演唱：成熟女声
曲风定位：古风武侠 / 江湖抒情
曲风：大气悲壮 + 温柔婉转，侠气与柔情交织
节奏：中速偏缓，4/4 拍，起承转合分明，副歌深情有力
乐器：古筝、竹笛、二胡、琵琶、弦乐铺底、轻微鼓点、古风打击乐（木鱼、铜铃），间奏加入箫声"""

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("""
        ## 🎵 ACE-Step 1.5 音乐生成
        使用 ACE-Step 1.5 模型生成音乐，支持歌词控制和音频分析参考
        
        **模型要求：**
        - 将模型放入 `models/acestep-v15-xl-turbo/` 目录
        - 模型需包含以下文件：
          - `config.json` - 模型配置
          - `pytorch_model.bin` - 模型权重
          
        **如果本地没有模型，首次运行会自动从 Hugging Face 下载。**
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="🎵 曲风/风格提示",
                    placeholder="例如：硬摇滚，节奏紧凑，充满能量...",
                    value=EXAMPLE_PROMPT,
                    lines=3
                )
                
                lyrics = gr.Textbox(
                    label="📝 歌词（可选）",
                    placeholder="输入歌词内容，支持段落式格式...",
                    value=EXAMPLE_LYRICS,
                    lines=8
                )
                
                # 音乐参数行
                with gr.Row():
                    bpm = gr.Number(
                        label="🎵 BPM",
                        value=120,
                        minimum=30,
                        maximum=300,
                        step=1
                    )
                    
                    key_scale = gr.Dropdown(
                        label="🎼 调式",
                        choices=[
                            "C major", "C minor", "C# major", "C# minor",
                            "D major", "D minor", "D# major", "D# minor",
                            "E major", "E minor",
                            "F major", "F minor", "F# major", "F# minor",
                            "G major", "G minor", "G# major", "G# minor",
                            "A major", "A minor", "A# major", "A# minor",
                            "B major", "B minor",
                        ],
                        value="E minor"
                    )
                    
                    time_signature = gr.Dropdown(
                        label="⏱️ 拍号",
                        choices=["2/4", "3/4", "4/4", "6/8"],
                        value="4/4"
                    )
                    
                    vocal_language = gr.Dropdown(
                        label="🗣️ 演唱语言",
                        choices=["en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "unknown"],
                        value="zh"
                    )
                
                # 示例按钮
                gr.Examples(
                    examples=[[EXAMPLE_PROMPT, EXAMPLE_LYRICS]],
                    label="示例：古风武侠",
                    inputs=[prompt, lyrics],
                )
                
                # 音频分析部分（用于参考）
                gr.Markdown("### 🎧 音频分析参考")
                gr.Markdown("可以上传音频，自动提取曲风、歌词等信息作为参考")
                
                with gr.Row():
                    src_audio = gr.Audio(
                        label="上传参考音频",
                        type="filepath"
                    )
                    analyze_button = gr.Button(
                        "🔍 分析音频",
                        variant="secondary",
                        size="lg"
                    )
                
                # 音频分析状态
                analysis_status = gr.Markdown(label="分析状态")
                
                with gr.Row():
                    duration = gr.Number(
                        label="⏱️ 时长（秒）",
                        value=30,
                        minimum=5,
                        maximum=300,
                        step=1
                    )
                    
                    infer_steps = gr.Slider(
                        label="🔄 推理步数",
                        minimum=4,
                        maximum=50,
                        value=8,
                        step=1
                    )
                
                guidance_scale = gr.Slider(
                    label="🎚️ 引导强度",
                    minimum=1.0,
                    maximum=20.0,
                    value=1.0,
                    step=0.5
                )
                
                model_version = gr.Dropdown(
                    label="🏷️ 模型版本",
                    choices=list(MODEL_VERSIONS.keys()),
                    value=list(MODEL_VERSIONS.keys())[0]
                )
                
                generate_button = gr.Button("🎵 生成音乐", variant="primary")
            
            with gr.Column(scale=2):
                audio_output = gr.Audio(label="🎧 生成的音乐")
                status_message = gr.Markdown(label="状态")
        
        # 分析按钮点击事件
        analyze_button.click(
            fn=analyze_src_audio_wrapper,
            inputs=[src_audio, model_version],
            outputs=[
                analysis_status,    # 分析状态
                prompt,              # 曲风描述
                lyrics,             # 歌词
                bpm,                # BPM
                duration,           # 时长
                key_scale,          # 调式
                vocal_language,     # 演唱语言
                time_signature,     # 拍号
            ]
        )
        
        # 生成按钮点击事件
        generate_button.click(
            fn=generate_music,
            inputs=[
                prompt,              # 曲风提示词
                lyrics,              # 歌词
                duration,            # 时长
                infer_steps,         # 推理步数
                guidance_scale,      # 引导强度
                model_version,       # 模型版本
                bpm,                # BPM
                key_scale,          # 调式
                time_signature,     # 拍号
                vocal_language,     # 演唱语言
            ],
            outputs=[audio_output, status_message]
        )
    
    return ui

# Note: UI will be created by multimodal_media_main.py
# Do not auto-create UI here to avoid duplicate rendering
