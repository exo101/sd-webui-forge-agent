"""Breeze-TTS-2 语音合成 UI（sd-webui-multimodal-media 插件）

- 推理代码：插件内置 breeze-tts/（github.com/breezeblue-ai/breeze-tts，Apache-2.0）
- 模型权重：webui/models/Breeze-TTS-2（从魔搭社区 BreezeBlue/Breeze-TTS-2 下载，
  可在 WebUI 的「开源社区模型下载器」标签页一键下载）
- 推理代码：通过 subprocess 调用 breeze-tts/infer.py，使用独立环境避免与 WebUI 依赖冲突。
"""
import os
import sys
import time
import shutil
import datetime
import subprocess
from pathlib import Path

import gradio as gr

from modules import shared
from modules.paths import models_path
from modules.paths_internal import default_output_dir

EXT_ROOT = Path(__file__).resolve().parent.parent
BREEZE_DIR = EXT_ROOT / "breeze-tts"
VENV_DIR = BREEZE_DIR / "venv"
VENV_PYTHON = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
MODEL_DIR = os.path.join(models_path, "Breeze-TTS-2")
OUTPUT_DIR = os.path.join(default_output_dir, "breeze-tts")

# infer.py 运行所需的关键模型文件（整仓下载后应全部存在）
REQUIRED_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "audio_tokenizer/model.safetensors",
]


def check_environment() -> bool:
    """Check that the isolated interpreter has the runtime packages."""
    if not VENV_PYTHON.is_file():
        return False
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import torch, transformers, qwen_tts"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


# ════════════════════════════════════════════════════════════════
# Whisper 语音识别（用于自动生成 Breeze 参考音频文字稿）
# ════════════════════════════════════════════════════════════════
whisper_model = None
whisper_processor = None


def transcribe_audio(audio_path: str) -> str:
    """使用 Whisper 自动识别参考音频中的文本，返回识别结果。"""
    global whisper_model, whisper_processor
    os.environ.setdefault("TRANSFORMERS_AUDIO_BACKEND", "ffmpeg")
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        try:
            import librosa
        except ImportError:
            librosa = None

        def load_audio_file(file_path):
            import soundfile as sf
            return sf.read(file_path)

        if whisper_model is None:
            print("[Breeze-TTS-2] 正在加载 Whisper 模型进行语音识别...")
            model_id = "openai/whisper-tiny"
            possible_paths = [
                os.path.join(shared.models_path, "whisper-tiny"),
                os.path.join(shared.models_path, "whisper", "whisper-tiny"),
                os.path.join(shared.models_path, "ASR", "whisper-tiny"),
            ]
            use_local = False
            local_model_path = model_id
            for path in possible_paths:
                if os.path.exists(path):
                    local_model_path = path
                    use_local = True
                    break
            whisper_processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=use_local)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=use_local,
            )
            if torch.cuda.is_available():
                model.to("cuda")
            whisper_model = model
            print("[Breeze-TTS-2] Whisper 模型加载完成。")

        audio_array, sample_rate = load_audio_file(audio_path)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            if librosa is not None:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            else:
                duration = len(audio_array) / sample_rate
                new_length = int(duration * 16000)
                old_indices = np.arange(len(audio_array))
                new_indices = np.linspace(0, len(audio_array) - 1, new_length)
                audio_array = np.interp(new_indices, old_indices, audio_array)
            sample_rate = 16000

        input_features = whisper_processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_features
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        input_features = input_features.to(dtype)
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        recognized_text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        print(f"[Breeze-TTS-2] 语音识别结果：{recognized_text}")
        return recognized_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Breeze-TTS-2] 语音识别失败：{e}")
        return ""


# ════════════════════════════════════════════════════════════════
# 状态检查
# ════════════════════════════════════════════════════════════════
def check_model() -> tuple:
    """检查模型文件是否齐全，返回 (是否齐全, 提示文本)。"""
    if not os.path.isdir(MODEL_DIR):
        return False, (
            f"❌ 未找到模型目录 `{MODEL_DIR}`。\n\n"
            "请到 WebUI 顶部「开源社区模型下载器」标签页 → 插件模型组合下载区，"
            "点击 **Breeze-TTS-2** 的一键下载按钮（约 7.7 GB，从魔搭社区下载）。"
        )
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(MODEL_DIR, f))]
    if missing:
        return False, f"❌ 模型文件不完整，缺少：{', '.join(missing)}。请重新在「开源社区模型下载器」中下载。"
    return True, f"✅ 模型已就绪：`{MODEL_DIR}`"


def _patch_transformers_version_check() -> None:
    """禁用 transformers 4.57.3 对 huggingface_hub<1.0 的硬性版本检查。
    transformers 在 import 时就执行 dependency_versions_check，运行时 monkey-patch 来不及，
    因此直接修改 venv 中的源文件。"""
    for sp in list(VENV_DIR.glob("Lib/site-packages")) + list(VENV_DIR.glob("lib/python*/site-packages")):
        dep_check = sp / "transformers" / "dependency_versions_check.py"
        if dep_check.is_file():
            src = dep_check.read_text(encoding="utf-8")
            if "_patched_no_ver_check" in src:
                continue
            # 在文件开头标记，并用 no-op 替换 require_version_core 调用
            marker = "# _patched_no_ver_check: disable huggingface_hub version gate\n"
            if "require_version_core" in src:
                import re
                new_src = marker + re.sub(
                    r"require_version_core\(deps\[pkg\]\)",
                    "pass  # _patched_no_ver_check",
                    src,
                )
                dep_check.write_text(new_src, encoding="utf-8")
            break


def _patch_qwen_tts_causal_mask() -> str:
    """根据 venv 中实际安装的 transformers 版本，修正 qwen-tts 0.1.1 源码里
    create_causal_mask 的参数名。transformers 4.57.x 的签名是
    create_causal_mask(config, input_embeds, attention_mask, cache_position, ...)，
    而更新版本改为 inputs_embeds 且移除了 cache_position。qwen_tts 0.1.1 源码
    固定写死 input_embeds + cache_position，因此需要按目标 transformers 签名对齐。"""
    import inspect

    target = None
    # 用 venv 的 Python 定位 qwen_tts 包路径（比硬编码 glob 更可靠）
    try:
        probe = subprocess.run(
            [str(VENV_PYTHON), "-c",
             "import qwen_tts, os; print(os.path.dirname(qwen_tts.__file__))"],
            capture_output=True, text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            pkg_dir = Path(probe.stdout.strip())
            for p in pkg_dir.glob("**/modeling_qwen3_tts_tokenizer_v2.py"):
                target = p
                break
    except Exception:
        pass

    # 回退：手动 glob
    if target is None:
        for sp in list(VENV_DIR.glob("Lib/site-packages")) + list(VENV_DIR.glob("lib/python*/site-packages")):
            for p in (sp / "qwen_tts").glob("**/modeling_qwen3_tts_tokenizer_v2.py"):
                target = p
                break
            if target is not None:
                break
    if target is None:
        return "⚠️ 未在独立环境中找到 qwen_tts 的 modeling 文件，跳过修补。"

    # 用 venv 的 Python 探测 transformers 签名（WebUI 自己的 transformers 版本可能不同）
    # 先禁用 transformers 对 huggingface_hub 的版本检查（我们用 1.33.0 兼容 diffusers）
    try:
        probe = subprocess.run(
            [str(VENV_PYTHON), "-c",
             "import transformers.utils.versions as v; v.require_version=lambda *a,**k:None; "
             "import inspect; from transformers.masking_utils import create_causal_mask; "
             "print(','.join(inspect.signature(create_causal_mask).parameters.keys()))"],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            return f"⚠️ 无法探测 venv transformers 签名：{probe.stderr.strip()}，跳过修补。"
        params = set(probe.stdout.strip().split(","))
    except Exception as e:
        return f"⚠️ 无法探测 venv transformers 签名：{e}，跳过修补。"

    embeds_key = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
    has_cache_position = "cache_position" in params

    src = target.read_text(encoding="utf-8")
    new_src = src
    # 全局替换：把文件中所有 inputs_embeds / input_embeds 统一对齐到目标签名
    # 包括函数签名参数名、变量名、字典键、关键字参数，确保函数体内外一致
    if embeds_key == "inputs_embeds":
        new_src = new_src.replace("input_embeds", "inputs_embeds")
    else:
        new_src = new_src.replace("inputs_embeds", "input_embeds")
    if not has_cache_position:
        new_src = new_src.replace('                "cache_position": cache_position,\n', "")
    else:
        if '"cache_position": cache_position,' not in new_src:
            new_src = new_src.replace(
                f'"{embeds_key}": {embeds_key},\n                "attention_mask": attention_mask,',
                f'"{embeds_key}": {embeds_key},\n                "attention_mask": attention_mask,\n                "cache_position": cache_position,',
            )

    if new_src != src:
        target.write_text(new_src, encoding="utf-8")
        modeling_patched = True
    else:
        modeling_patched = False

    # 同时修补 breeze-tts 项目自身的调用方（如 lane.py），把 inputs_embeds= 对齐到目标签名
    patched_calls = 0
    for py_file in BREEZE_DIR.rglob("*.py"):
        try:
            fsrc = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if embeds_key == "inputs_embeds":
            fnew = fsrc.replace("input_embeds=", "inputs_embeds=")
        else:
            fnew = fsrc.replace("inputs_embeds=", "input_embeds=")
        if fnew != fsrc:
            py_file.write_text(fnew, encoding="utf-8")
            patched_calls += 1

    return f"✅ 已按 transformers 签名对齐（embeds={embeds_key}, cache_position={has_cache_position}, modeling={'已修补' if modeling_patched else '已匹配'}, 调用方修补={patched_calls}）。"


def prepare_environment():
    """Create the isolated Breeze environment and install its pinned dependencies."""
    if not BREEZE_DIR.is_dir():
        return f"❌ 未找到推理代码目录：{BREEZE_DIR}"
    req = BREEZE_DIR / "requirements.txt"
    if not req.is_file():
        return f"❌ 未找到依赖清单：{req}"
    logs = []
    if not VENV_PYTHON.is_file():
        # 用 --system-site-packages 继承 WebUI 已安装的 torch，避免重新下载几个 GB
        code, tail = _run_stream([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)], logs)
        if code != 0 or not VENV_PYTHON.is_file():
            return "❌ 创建独立环境失败：\n" + "\n".join(tail)
    code, tail = _run_stream([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], logs)
    if code != 0:
        return "❌ pip 升级失败：\n" + "\n".join(tail)

    # 检查 venv 是否能访问 WebUI 的 torch（--system-site-packages 继承）
    check_torch = subprocess.run(
        [str(VENV_PYTHON), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True,
    )
    if check_torch.returncode == 0:
        logs.append(f"复用 WebUI torch: {check_torch.stdout.strip()}")
    else:
        # 没有继承到 torch，手动安装 cu128 + 2.10
        code, tail = _run_stream([
            str(VENV_PYTHON), "-m", "pip", "install",
            "torch==2.10.0+cu128",
            "torchaudio==2.10.0+cu128",
            "--index-url", "https://download.pytorch.org/whl/cu128",
        ], logs)
        if code != 0:
            return "❌ torch 安装失败（cu128 + 2.10）：\n" + "\n".join(tail)

    code, tail = _run_stream([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(req),
                              "--extra-index-url", "https://pypi.org/simple"], logs)
    if code != 0:
        return "❌ Breeze-TTS 依赖安装失败：\n" + "\n".join(tail)
    # 强制安装 huggingface_hub==1.33.0（与 WebUI 一致，兼容 diffusers），跳过依赖检查
    # transformers 4.57.3 声明需要 huggingface_hub<1.0，但实际运行兼容 1.33.0
    code, tail = _run_stream([str(VENV_PYTHON), "-m", "pip", "install",
                              "huggingface_hub==1.33.0", "--no-deps",
                              "--extra-index-url", "https://pypi.org/simple"], logs)
    if code != 0:
        logs.append("⚠️ huggingface_hub 升级失败，ACE-Step 可能无法使用 diffusers")
    # 禁用 transformers 4.57.3 对 huggingface_hub<1.0 的硬性版本检查
    # 直接修改 venv 中的 dependency_versions_check.py，因为运行时 monkey-patch 来不及
    _patch_transformers_version_check()
    patch_msg = _patch_qwen_tts_causal_mask()
    return f"✅ Breeze-TTS 独立推理环境准备完成。\n{patch_msg}"


# ════════════════════════════════════════════════════════════════
# 子进程执行（流式收集输出）
# ════════════════════════════════════════════════════════════════
def _run_stream(cmd, log_lines, max_tail=400):
    """运行命令并流式收集输出，返回 (返回码, 尾部日志)。

    每行输出同时 print 到 WebUI 后台，避免用户以为卡住没进度。
    """
    log_lines.clear()
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[Breeze-TTS-2] 执行: {cmd_str}", flush=True)

    def _append(line):
        log_lines.append(line)
        # 实时输出到后台，让用户看到进度
        print(f"[Breeze-TTS-2] {line}", flush=True)
        if len(log_lines) > max_tail:
            del log_lines[: len(log_lines) - max_tail]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BREEZE_DIR),
        )
    except Exception as e:
        _append(f"启动进程失败: {e}")
        return 1, list(log_lines)

    assert proc.stdout is not None
    for line in proc.stdout:
        _append(line.rstrip())
    code = proc.wait()
    print(f"[Breeze-TTS-2] 命令结束，返回码: {code}", flush=True)
    return code, list(log_lines)


# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# 推理
# ════════════════════════════════════════════════════════════════
def _ensure_wav(audio_path: str) -> str:
    """参考音频统一转成 WAV（infer.py 按 wav 处理最稳妥）。已是常见格式则直接用。"""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        return audio_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"参考音频格式为 {ext or '未知'}，需要 FFmpeg 转换，请上传 wav 文件。")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    converted = os.path.join(
        OUTPUT_DIR,
        f"ref_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{os.getpid() % 1000}.wav",
    )
    code, tail = _run_stream(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path, "-ar", "24000", "-ac", "1", converted],
        [],
    )
    if code != 0 or not os.path.isfile(converted):
        raise RuntimeError(f"参考音频转换失败：{' '.join(tail[-5:])}")
    return converted


def _transcribe_reference_audio(audio_path: str) -> str:
    """使用 Whisper 识别参考音频，自动生成 Breeze 的 ref-text。"""
    try:
        return (transcribe_audio(audio_path) or "").strip()
    except Exception as e:
        print(f"[Breeze-TTS-2] 参考音频识别失败：{e}")
        return ""


def generate_breeze_speech(mode, text, instruction, ref_audio, ref_text, cfg_scale, seed, fast_all):
    """调用 breeze-tts/infer.py 生成一条语音，返回 (音频路径, 日志)。"""
    ok_model, model_msg = check_model()
    if not ok_model:
        return None, model_msg
    if not check_environment():
        return None, "❌ Breeze-TTS 独立环境未准备完成，请先点击「准备推理环境」。"
    # 每次生成前都确保 qwen_tts 的 causal_mask 补丁已应用（防止重复点击准备环境后源码被还原）
    _patch_qwen_tts_causal_mask()

    text = (text or "").strip()
    instruction = (instruction or "").strip()
    ref_text = (ref_text or "").strip()
    if not text:
        return None, "❌ 请输入要合成的文本。"

    if mode == "voice-clone" or mode == "voice-direction":
        if not ref_audio:
            return None, "❌ 该模式需要上传参考音频。"
    if mode == "voice-direction" and not instruction:
        return None, "❌ 声音引导模式需要填写引导指令（语气/情感/语速描述）。"

    try:
        ref_wav = _ensure_wav(ref_audio) if ref_audio else None
    except Exception as e:
        return None, f"❌ {e}"

    # 未手动填写时，自动使用插件已有的 Whisper 识别参考音频。
    if mode in ("voice-clone", "voice-direction") and not ref_text:
        ref_text = _transcribe_reference_audio(ref_wav)
        if not ref_text:
            return None, "❌ 无法自动识别参考音频文字，请检查 Whisper 模型或手动填写文字稿。"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = os.path.join(OUTPUT_DIR, f"breeze_{mode}_{ts}.wav")

    cmd = [
        str(VENV_PYTHON),
        str(BREEZE_DIR / "infer.py"),
        MODEL_DIR,
        "--text", text,
        "--output", out_wav,
        "--seed", str(int(seed)),
        "--cfg-scale", f"{float(cfg_scale):.4f}",
    ]
    if instruction:
        cmd += ["--instruction", instruction]
    if ref_wav:
        cmd += ["--ref-audio", ref_wav, "--ref-text", ref_text]
    if fast_all:
        cmd += ["--fast-all"]

    print(f"[Breeze-TTS-2] 生成：{' '.join(cmd)}")
    log_lines = []
    code, tail = _run_stream(cmd, log_lines)

    if code != 0 or not os.path.isfile(out_wav):
        return None, "❌ 生成失败（退出码 {}）：\n{}".format(code, "\n".join(tail))

    # 清理转换出的临时参考音频
    if ref_wav and ref_wav != ref_audio and os.path.isfile(ref_wav):
        try:
            os.remove(ref_wav)
        except OSError:
            pass
    return out_wav, "\n".join(tail)


def open_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["explorer", OUTPUT_DIR], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", OUTPUT_DIR], check=False)
    else:
        subprocess.run(["xdg-open", OUTPUT_DIR], check=False)
    return f"✓ 已打开输出目录：{OUTPUT_DIR}"


# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════
def create_breeze_tts_ui(asset_bridge_id="h3studio-asset-bridge"):
    gr.Markdown(
        "## 🎙️ Breeze-TTS-2 语音合成\n"
        "开源权重双语 TTS（中文/英文），支持**声音克隆**、**声音设计**、**声音引导**三种模式，"
        "文本中可插入语音事件：英文用括号如 `(laugh)` `(sigh)`，中文用方括号如 `[笑]` `[叹气]`。\n"
        "- 模型来源：魔搭社区 [BreezeBlue/Breeze-TTS-2](https://www.modelscope.cn/models/BreezeBlue/Breeze-TTS-2)"
        "（在 WebUI「开源社区模型下载器」一键下载，约 7.7 GB）\n"
        "- 显存要求：eager 推理约 7.7 GiB（建议 ≥12 GB 显卡）；`fast-all` 快速路径约 14.4 GiB（建议 24 GB）\n"
        "- 许可证：模型权重仅限研究与非商业用途"
    )

    # 状态行：模型与独立推理环境
    with gr.Row():
        model_status_box = gr.Markdown(check_model()[1])
    with gr.Row():
        prepare_btn = gr.Button("⚙️ 准备推理环境", variant="secondary", scale=0)
        prepare_status = gr.Markdown("✅ 推理环境已就绪" if check_environment() else "⚠️ 请先点击「准备推理环境」")
    with gr.Accordion("环境准备日志", open=False):
        prepare_log = gr.Textbox(label="日志", interactive=False, lines=6)
    prepare_btn.click(prepare_environment, outputs=prepare_log)

    with gr.Row():
        with gr.Column(scale=3):
            mode = gr.Radio(
                choices=[
                    ("声音克隆（参考音频，自动识别文字）", "voice-clone"),
                    ("声音设计（自然语言描述音色，无需参考音频）", "voice-design"),
                    ("声音引导（克隆音色 + 语气/情感引导）", "voice-direction"),
                ],
                value="voice-clone",
                label="合成模式",
            )
            text = gr.Textbox(
                label="要合成的文本（必填）",
                placeholder="例如：[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。",
                lines=4,
            )
            instruction = gr.Textbox(
                label="音色/引导指令（可选；声音设计与声音引导模式建议填写，指令语言请与文本一致）",
                placeholder="例如：一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。",
                lines=2,
            )
            with gr.Row():
                cfg_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, step=0.1, value=1.0,
                    label="CFG 强度（声音设计/引导建议 4.0）",
                )
                seed = gr.Number(value=42, precision=0, label="随机种子")
                fast_all = gr.Checkbox(value=False, label="fast-all 快速路径（需 24 GB 显存，冷启动较慢）")
        with gr.Column(scale=2):
            ref_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="参考音频（克隆/引导模式必填，建议 3-10 秒清晰人声）",
            )
            ref_text = gr.Textbox(
                label="参考音频文字稿（可选，留空将自动识别）",
                placeholder="留空即可，系统会自动识别参考音频中的文字",
                lines=3,
            )

    generate_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
    audio_out = gr.Audio(label="生成结果", type="filepath", autoplay=False)
    gen_log = gr.Textbox(label="推理日志", interactive=False, lines=8)
    open_dir_btn = gr.Button("📁 打开输出目录", scale=0)
    send_btn = gr.Button("📥 发送到 H3 工作台素材区", scale=0)
    send_status = gr.Markdown()
    asset_bridge = gr.Textbox(elem_id=asset_bridge_id, visible=False)

    def send_voice(path):
        if not path:
            return "⚠️ 请先生成语音。", ""
        from h3studio.asset_import import import_local_asset, queue_external_asset
        import json
        item = import_local_asset(path)
        queue_external_asset(item)
        return "✅ 语音已发送到 H3 工作台素材区。", json.dumps([item], ensure_ascii=False)

    generate_btn.click(
        fn=generate_breeze_speech,
        inputs=[mode, text, instruction, ref_audio, ref_text, cfg_scale, seed, fast_all],
        outputs=[audio_out, gen_log],
    )
    open_dir_btn.click(open_output_directory, outputs=gen_log)
    send_btn.click(send_voice, inputs=[audio_out], outputs=[send_status, asset_bridge])
