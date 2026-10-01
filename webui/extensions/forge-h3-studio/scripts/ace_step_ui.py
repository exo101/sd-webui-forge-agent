import gradio as gr
import os
import sys
import json
from pathlib import Path
from modules import shared

# 定义插件目录
plugin_dir = Path(__file__).resolve().parent.parent

# ACE-Step-1.5 代码路径（相对于插件目录）
ace_step_path = (plugin_dir / "ACE-Step-1.5").resolve()
ace_step_path_str = str(ace_step_path)
if not (ace_step_path / "acestep" / "__init__.py").is_file():
    raise ImportError(f"ACE-Step source package not found: {ace_step_path / 'acestep'}")
if ace_step_path_str in sys.path:
    sys.path.remove(ace_step_path_str)
sys.path.insert(0, ace_step_path_str)

# 与 ACE-Step 1.5 官方模型配置保持一致。可通过环境变量覆盖。
os.environ.setdefault("ACESTEP_DTYPE", "bfloat16")

# ACE-Step-1.5 模型路径 - 用户把所有模型集中放在 models/ace-step/ 目录
WEBUI_ROOT = Path(__file__).parent.parent.parent.parent  # sd-webui-forge-neo-v3/webui
MODELS_DIR = WEBUI_ROOT / "models" / "ace-step"

# 设置 ACESTEP_CHECKPOINTS_DIR 环境变量，告诉原项目模型在哪里
os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(MODELS_DIR)

# 打印调试信息
# print(f"[ACE-Step-1.5] Checkpoints 目录: {MODELS_DIR}")

# 模型版本配置
# display_name: UI 显示名称
# internal_name: ACE-Step 1.5 内部使用的模型名称
MODEL_VERSIONS = {
    "ACE-Step-v15-xl-turbo": {
        "repo_id": "ACE-Step/ACE-Step-v15-xl-turbo",
        "internal_name": "acestep-v15-xl-turbo",
        "quantized": False,
        "description": "ACE-Step 1.5 XL Turbo (4B)，质量更高，推荐使用"
    },
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
    model_dir = MODELS_DIR / internal_name
    config_file = model_dir / "config.json"
    if config_file.is_file():
        try:
            with config_file.open("r", encoding="utf-8") as f:
                model_config_json = json.load(f)
            # XL (4B) 模型有独立的 encoder_hidden_size=2048（与 hidden_size=2560 不同）
            # 1.3B turbo 模型 encoder_hidden_size 与 hidden_size 相同（=2048），config 中无此字段
            if internal_name == "acestep-v15-xl-turbo":
                if model_config_json.get("encoder_hidden_size") != 2048:
                    raise RuntimeError(
                        f"模型配置不匹配: {config_file} 缺少 encoder_hidden_size=2048。"
                        "请重新下载与 ACE-Step-1.5 源码匹配的模型。"
                    )
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"模型配置损坏: {config_file}: {exc}") from exc
    
    # 如果模型已加载且版本相同，使用缓存
    # 但如果 dtype 与当前环境变量不一致（如改了精度设置），强制重载
    if ace_step_handler is not None and current_model_version == internal_name:
        import torch
        cached_dtype = getattr(ace_step_handler, "dtype", None)
        env_dtype_str = os.environ.get("ACESTEP_DTYPE", "").strip().lower()
        env_dtype = {
            "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }.get(env_dtype_str)
        if env_dtype is not None and cached_dtype is not None and cached_dtype != env_dtype:
            print(f"[ACE-Step-1.5] 缓存模型 dtype={cached_dtype} 与 ACESTEP_DTYPE={env_dtype_str} 不一致，强制重载...")
            ace_step_handler = None
        else:
            print(f"[ACE-Step-1.5] 模型已加载 (版本: {internal_name}, dtype={cached_dtype})")
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
        
        # Keep the official initialization contract. CPU offload is needed for
        # the XL model on 16 GB cards; the official loader selects the safe
        # attention implementation for the detected GPU.
        status_msg, ok = handler.initialize_service(
            project_root=str(ace_step_path),
            config_path=internal_name,
            device="auto",
            offload_to_cpu=True,
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

def generate_music(prompt, lyrics, duration, infer_steps, guidance_scale, model_version, bpm, key_scale, time_signature, vocal_language,
                   task_mode="文本生成音乐", cover_src_audio=None, cover_strength=1.0, cover_noise=0.0):
    """生成音乐：文本生成音乐 或 翻唱（基于源音频）"""
    print(
        f"[ACE-Step-1.5] callback module={__name__}, file={__file__}, "
        f"worker_env={os.environ.get('H3_ACE_STEP_WORKER', '<unset>')}",
        flush=True,
    )
    # Run ACE-Step in the same isolated interpreter used by Breeze-TTS.  The
    # Forge process cannot switch Python interpreters after startup, so the
    # worker is deliberately a subprocess.
    if "--h3-worker" not in sys.argv:
        import subprocess
        worker_python = plugin_dir / "breeze-tts" / "venv" / "Scripts" / "python.exe"
        worker_script = Path(__file__).with_name("ace_step_worker.py")
        if not worker_python.is_file():
            return None, f"❌ ACE-Step 隔离环境不存在: {worker_python}"
        request = {
            "prompt": prompt, "lyrics": lyrics, "duration": duration,
            "infer_steps": infer_steps, "guidance_scale": guidance_scale,
            "model_version": model_version, "bpm": bpm, "key_scale": key_scale,
            "time_signature": time_signature, "vocal_language": vocal_language,
            "task_mode": task_mode, "cover_src_audio": cover_src_audio,
            "cover_strength": cover_strength, "cover_noise": cover_noise,
        }
        env = os.environ.copy()
        env["H3_ACE_STEP_WORKER"] = "1"
        try:
            print(f"[ACE-Step-1.5] 启动隔离 worker: {worker_python}")
            proc = subprocess.run(
                [str(worker_python), str(worker_script), "--h3-worker", json.dumps(request, ensure_ascii=False)],
                env=env, capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=str(ace_step_path), check=False,
            )
            if proc.stdout:
                print(proc.stdout)
            if proc.returncode != 0:
                return None, f"❌ ACE-Step 隔离进程失败:\n{proc.stderr or proc.stdout}"
            for line in reversed((proc.stdout or "").splitlines()):
                if line.startswith("H3_ACE_RESULT="):
                    result = json.loads(line.split("=", 1)[1])
                    return result.get("path"), result.get("error")
            return None, f"❌ ACE-Step 隔离进程没有返回结果:\n{proc.stderr or proc.stdout}"
        except Exception as exc:
            return None, f"❌ 无法启动 ACE-Step 隔离环境: {exc}"

    is_cover = str(task_mode or "").strip() == "翻唱"
    if is_cover and not cover_src_audio:
        return None, "❌ 翻唱模式请先上传翻唱源音频（原曲）"
    try:
        # Forge may keep the currently selected SD model resident on the GPU.
        # torch.cuda.empty_cache() cannot release those live model tensors, and
        # ACE-Step XL then falls back to an extremely memory-starved tiled VAE
        # decode (often producing near-zero/noisy audio).  Use Forge's official
        # unload hook for the duration of this generation.
        forge_model_was_loaded = False
        if "--h3-worker" not in sys.argv:
            try:
                from modules import sd_models
                forge_model_was_loaded = getattr(shared, "sd_model", None) is not None
                if forge_model_was_loaded:
                    print("[ACE-Step-1.5] 卸载 Forge SD 模型，释放显存...")
                    sd_models.unload_model_weights()
            except Exception as unload_error:
                print(f"[ACE-Step-1.5] Forge SD 模型卸载失败（继续尝试）: {unload_error}")

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
        # 初始化 LLM（5Hz 语言模型），用于 Phase 1 生成 audio codes
        llm_handler = get_llm_handler(model_version)
        if llm_handler is None or not llm_handler.llm_initialized:
            raise RuntimeError("LLM（5Hz 语言模型）未初始化，无法生成 audio codes。请确认已下载 acestep-5Hz-lm-1.7B 模型。")

        print(f"[ACE-Step-1.5] 开始生成音乐... (模式: {task_mode})")
        print(f"[ACE-Step-1.5] 提示词: {prompt[:50]}..." if len(prompt) > 50 else f"[ACE-Step-1.5] 提示词: {prompt}")
        print(f"[ACE-Step-1.5] 歌词: {lyrics[:50]}..." if len(lyrics) > 50 else f"[ACE-Step-1.5] 歌词: {lyrics}")
        print(f"[ACE-Step-1.5] 推理步数: {infer_steps}, 引导强度: {guidance_scale}, 时长: {duration}秒")
        print(f"[ACE-Step-1.5] BPM: {bpm}, 调式: {key_scale}, 拍号: {time_signature}, 语言: {vocal_language}")

        # 使用官方两阶段管线：Phase 1 LM 生成 audio codes → Phase 2 DiT 生成音乐
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            task_type="cover" if is_cover else "text2music",
            caption=prompt,
            lyrics=lyrics if lyrics.strip() else "",
            bpm=bpm if bpm else None,
            keyscale=key_scale if key_scale else "",
            timesignature=time_signature if time_signature else "",
            vocal_language=vocal_language if vocal_language else "en",
            duration=duration if (duration and duration > 0 and not is_cover) else -1.0,
            inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            seed=-1,
            # 翻唱模式官方强制关闭 LM thinking（无需 CoT，直接基于原曲 latent 加噪去噪）
            thinking=not is_cover,
            use_cot_caption=not is_cover,
            use_cot_metas=not is_cover,
            use_cot_language=not is_cover,
        )

        if is_cover:
            params.src_audio = str(cover_src_audio)
            # audio_cover_strength=混音强度（控制去噪步数/CFG混合），官方默认 0.0
            # cover_noise_strength=翻唱强度（控制旋律还原：0=纯噪声，1=最接近原曲），官方默认 0.2
            params.audio_cover_strength = float(cover_strength if cover_strength is not None else 0.0)
            params.cover_noise_strength = float(cover_noise if cover_noise is not None else 0.2)
            print(f"[ACE-Step-1.5] 翻唱源音频: {cover_src_audio}, 混音强度: {params.audio_cover_strength}, 翻唱强度: {params.cover_noise_strength}")

        config = GenerationConfig(
            batch_size=1,
            use_random_seed=True,
            audio_format="wav",
        )

        # 输出目录：webui 采样输出目录
        save_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples

        # 调用官方两阶段生成管线
        result = generate_music(
            dit_handler=handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=save_dir,
        )

        print(f"[ACE-Step-1.5] 生成结果 success={result.success}, audios={len(result.audios)}")
        if not result.success:
            raise RuntimeError(result.error or result.status_message or "模型生成失败")

        if not result.audios:
            raise ValueError("生成结果中没有音频")

        # 取第一个音频的路径
        output_path = result.audios[0].get("path")
        if not output_path or not os.path.exists(output_path):
            raise ValueError(f"音频文件不存在: {output_path}")

        print(f"✅ 音乐生成成功（{task_mode}）！文件: {output_path}")
        return output_path, None
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 音乐生成失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg
    finally:
        # The next Forge image generation will reload lazily; do not keep the
        # large ACE-Step XL model and Forge model resident together.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def create_ace_step_ui(asset_bridge_id="h3studio-asset-bridge"):
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
        使用 ACE-Step 1.5 模型生成音乐，支持**文本生成音乐**和**翻唱（基于原曲音频）**
        
        **翻唱说明：**
        - 切换到「翻唱」模式，上传**一段原曲音频**即可（无需分轨/人声分离）
        - 翻唱强度控制旋律还原程度（推荐 0.1~0.25），混音强度控制风格转换程度
        - 曲风/歌词留空则尽量保留原曲，填写则会向新风格/新歌词方向翻唱
        - 时长自动跟随原曲，无需手动设置
        
        **模型要求：**
        - 模型存放目录：`models/ace-step/`
        - 主模型仓库：`ACE-Step/Ace-Step1.5`（DiT、LM、文本编码器、VAE）
        - XL DiT 仓库：`ACE-Step/acestep-v15-xl-turbo`
          
        **如果本地没有模型，首次运行会自动从魔搭（ModelScope）下载，无需手动操作。**
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # 生成模式：文本生成音乐 / 翻唱
                task_mode = gr.Radio(
                    label="🎼 生成模式",
                    choices=["文本生成音乐", "翻唱"],
                    value="文本生成音乐"
                )
                
                # 翻唱模式参数（仅在「翻唱」模式下显示）
                cover_group = gr.Group(visible=False)
                with cover_group:
                    cover_src_audio = gr.Audio(
                        label="🎤 上传翻唱源音频（原曲，仅需一段）",
                        type="filepath"
                    )
                    with gr.Row():
                        cover_noise = gr.Slider(
                            label="翻唱强度（旋律还原：0=纯噪声，1=最接近原曲，推荐 0.1~0.25）",
                            minimum=0.0, maximum=1.0, value=0.2, step=0.05
                        )
                        cover_strength = gr.Slider(
                            label="混音强度（风格转换：0=完全去噪，1=保留最多原曲信息）",
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05
                        )
                
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
                send_btn = gr.Button("📥 发送到 H3 工作台素材区")
                send_status = gr.Markdown()
                asset_bridge = gr.Textbox(elem_id=asset_bridge_id, visible=False)

        def send_music(path):
            if not path:
                return "⚠️ 请先生成音乐。", ""
            from h3studio.asset_import import import_local_asset, queue_external_asset
            import json
            item = import_local_asset(path)
            queue_external_asset(item)
            return "✅ 音乐已发送到 H3 工作台素材区。", json.dumps([item], ensure_ascii=False)
        
        # 生成模式切换：翻唱模式显示翻唱参数
        def _toggle_cover_group(mode):
            return gr.update(visible=str(mode or "").strip() == "翻唱")
        task_mode.change(fn=_toggle_cover_group, inputs=task_mode, outputs=cover_group)
        
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
                task_mode,          # 生成模式（文本/翻唱）
                cover_src_audio,    # 翻唱源音频
                cover_strength,     # 翻唱强度
                cover_noise,        # 翻唱噪声强度
            ],
            outputs=[audio_output, status_message]
        )
        send_btn.click(send_music, inputs=[audio_output], outputs=[send_status, asset_bridge])
    
    return ui

# Note: UI will be created by multimodal_media_main.py
# Do not auto-create UI here to avoid duplicate rendering
