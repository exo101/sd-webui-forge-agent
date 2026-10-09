"""SCAIL-2 角色动画/替换 UI。

基于 DiffSynth-Studio 的 SCAIL2Pipeline，提供：
- 参考图上传
- 驱动视频上传
- 动画模式 / 替换模式切换
- **DiT 量化格式选择**（支持 int8_convrot / fp8_scaled / mxfp8 / fp16）
- SAM3 目标跟踪（替换模式）
- 长视频分段生成
"""
import gradio as gr
import gc
import contextlib
import logging
import os
import subprocess
import sys
import threading
import uuid
import numpy as np
from pathlib import Path
from PIL import Image

# 插件目录与 DiffSynth-Studio 路径
plugin_dir = Path(__file__).resolve().parent.parent
diffsynth_path = (plugin_dir / "DiffSynth-Studio").resolve()
if str(diffsynth_path) not in sys.path:
    sys.path.insert(0, str(diffsynth_path))

WEBUI_ROOT = Path(__file__).parent.parent.parent.parent  # sd-webui-forge/webui
MODELS_DIR = WEBUI_ROOT / "models"
COMFY_ROOT = Path(os.environ.get("SCAIL2_COMFY_ROOT", r"D:\ai\ComfyUI-aki-v2.2\ComfyUI"))
COMFY_PYTHON = COMFY_ROOT.parent / "python" / "python.exe"


class _Scail2QuietStream:
    """Suppress generation-thread console chatter without muting other WebUI threads."""

    _scail2_quiet_proxy = True

    def __init__(self, stream, state):
        self._stream = stream
        self._state = state

    def write(self, value):
        if getattr(self._state, "quiet", False):
            # Keep concise sampling progress visible in the WebUI backend log,
            # while suppressing verbose DiffSynth/model-loading output.
            if "[SCAIL2][进度]" in value:
                return self._stream.write(value)
            return len(value)
        return self._stream.write(value)

    def flush(self):
        if not getattr(self._state, "quiet", False):
            return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_existing_quiet_stream = sys.stdout if getattr(sys.stdout, "_scail2_quiet_proxy", False) else None
_console_state = (
    _existing_quiet_stream._state if _existing_quiet_stream is not None
    else threading.local()
)
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if not getattr(_stream, "_scail2_quiet_proxy", False):
        setattr(sys, _stream_name, _Scail2QuietStream(_stream, _console_state))


@contextlib.contextmanager
def _quiet_generation_console():
    previous = getattr(_console_state, "quiet", False)
    _console_state.quiet = True
    try:
        yield
    finally:
        _console_state.quiet = previous

# --- DiT checkpoint 注册 ---------------------------------------------------
# label → 文件名 / 大小 / 备注，用户在 UI 下拉里直接看 label。
# 顺序按 GPU 架构自动重排，下面 _auto_default_label() 决定初始选中。
_DIT_FORMATS = [
    # (label, 文件名, 大小 GB, 备注)
    ("INT8 ConvRot (通用, ~16.65GB)",   "wan2.1_14B_SCAIL_2_int8_convrot.safetensors",     16.65,  "Ampere/Hopper/Blackwell 全兼容"),
    ("MXFP8 (Hopper+/Blackwell, ~17GB)","wan2.1_14B_SCAIL_2_mxfp8.safetensors",            17.15,  "RTX 40/50 系原生加速"),
    ("FP8 E4M3 (Ampere+, ~17.7GB)",     "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",       17.69,  "RTX 30/40/50 系可用"),
    ("FP16 (完整版, ~32.8GB)",          "wan2.1_14B_SCAIL_2_fp16.safetensors",             32.79,  "需要 ≥33GB VRAM"),
]


def _auto_default_label():
    """按 GPU compute capability + VRAM 选 UI 默认选中的格式。"""
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        cc = props.major * 10 + props.minor
        vram_gb = props.total_memory / (1024 ** 3)
    except Exception:
        # 无 CUDA 信息 → 选最小 int8 作为默认
        return next((lbl for lbl, _, sz, _ in _DIT_FORMATS if lbl.startswith("INT8")), None)

    # 逐个检查候选的 VRAM 可行性（大小 + 留 ~3GB 给激活值）
    fits = lambda gb: gb + 3.0 <= vram_gb

    # Match the official ComfyUI SCAIL-2 Character Replacement workflow.
    # Prefer its INT8 ConvRot checkpoint when installed, including on Blackwell.
    int8_path = MODELS_DIR / "diffusion_models" / "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"
    if int8_path.exists():
        return "INT8 ConvRot (通用, ~16.65GB)"

    if cc >= 90 and fits(17.15):
        return "MXFP8 (Hopper+/Blackwell, ~17GB)"
    if fits(16.65):
        return "INT8 ConvRot (通用, ~16.65GB)"
    if fits(17.69):
        return "FP8 E4M3 (Ampere+, ~17.7GB)"
    if fits(32.79):
        return "FP16 (完整版, ~32.8GB)"
    # 都装不下 → 选 int8，让 pipeline 的 segment_len 自动降低
    return "INT8 ConvRot (通用, ~16.65GB)"


def _list_available_dit_files():
    """扫描 diffusion_models 目录，返回 (label, path) 列表 — 实际存在的格式。"""
    dit_dir = MODELS_DIR / "diffusion_models"
    available = []
    for label, filename, size_gb, note in _DIT_FORMATS:
        p = dit_dir / filename
        if p.exists():
            available.append((label, p))
    return available


def _label_to_path(label):
    """把 UI 下拉的 label 回传成 checkpoint 路径。"""
    for lbl, p in _list_available_dit_files():
        if lbl == label:
            return p
    # 回退：label → 文件名再找一次
    for lbl, filename, _, _ in _DIT_FORMATS:
        if lbl == label:
            candidate = MODELS_DIR / "diffusion_models" / filename
            if candidate.exists():
                return candidate
            return candidate  # 不存在也返回（会在 pipeline init 时报错提示）
    return None


# --- 其他静态模型路径 -------------------------------------------------------
SCAIL2_STATIC_PATHS = {
    "text_encoder": MODELS_DIR / "text_encoder" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "clip_vision": MODELS_DIR / "clip_vision" / "clip_vision_h.safetensors",
    "vae": MODELS_DIR / "vae" / "wan_2.1_vae.safetensors",
    "lightx2v_lora": MODELS_DIR / "Lora" / "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    "dpo_lora": MODELS_DIR / "Lora" / "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
    "sam3": MODELS_DIR / "Stable-diffusion" / "sam3.1_multiplex_fp16.safetensors",
}

# pipeline cache: { dit_label: SCAIL2Pipeline } — 选不同格式自动换，同格式复用
_pipeline_cache = {}


def get_pipeline(dit_label):
    """按 DiT 格式懒加载 pipeline。dit_label 变了就释放老的、构建新的。"""
    global _pipeline_cache

    # 1) 命中缓存
    if dit_label in _pipeline_cache and _pipeline_cache[dit_label] is not None:
        return _pipeline_cache[dit_label]

    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.scail2 import SCAIL2Pipeline

    dit_path = _label_to_path(dit_label)
    if dit_path is None or not dit_path.exists():
        avail = _list_available_dit_files()
        avail_str = "\n  ".join(lbl for lbl, _ in avail) if avail else "(none)"
        raise FileNotFoundError(
            f"DiT checkpoint 不存在：{dit_path}\n\n"
            f"当前 models/diffusion_models/ 里可用的：\n  {avail_str}\n\n"
            f"请下载受支持的 DiT 版本到 models/diffusion_models/ 目录。"
        )

    # 2) 必要的静态模型检查
    missing = []
    for key in ("text_encoder", "clip_vision", "vae"):
        p = SCAIL2_STATIC_PATHS[key]
        if not p.exists():
            missing.append(f"{key}: {p}")
    if missing:
        raise FileNotFoundError(
            "SCAIL-2 必需模型缺失：\n" + "\n".join(missing) +
            "\n\n请在「模型下载器」里下载 SCAIL-2 所需模型。"
        )

    print(f"[SCAIL2] 加载 DiT = {dit_label}")

    # 3) 清理所有缓存里的旧 pipeline（它们占显存）
    for lbl, pipe in list(_pipeline_cache.items()):
        if pipe is not None:
            for attr in ("dit", "vae", "text_encoder", "image_encoder", "sam3_tracker"):
                obj = getattr(pipe, attr, None)
                if obj is not None:
                    try:
                        obj.to("cpu")
                        del obj
                    except Exception:
                        pass
            _pipeline_cache[lbl] = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 4) 构建。ComfyUI keeps a quantized DiT resident on the GPU when it fits.
    # CPU-offloaded NVFP4 weights are especially slow here: long-sequence
    # chunked attention invokes the same Q/K/V layers repeatedly, and the
    # current wrapper would copy their packed weights to CUDA on every chunk.
    try:
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024 ** 3)
        free_vram = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
    except Exception:
        props = None
        total_vram = 16.0
        free_vram = 0.0

    vram = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cpu",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    if dit_label.startswith("NVFP4+MXFP8"):
        checkpoint_gb = dit_path.stat().st_size / (1024 ** 3)
        # The checkpoint is staged to CPU during conditioning, so initial
        # pipeline construction needs only modest workspace. Before sampling,
        # `_restore_dit_to_gpu` performs the stricter live-memory check.
        safety_reserve_gb = 3.0
        if props is None or props.major < 12:
            raise RuntimeError("NVFP4+MXFP8 SCAIL-2 checkpoint requires an NVIDIA Blackwell GPU (SM 12.0+).")
        if free_vram < checkpoint_gb + safety_reserve_gb:
            raise RuntimeError(
                "NVFP4+MXFP8 SCAIL-2 would fall back to very slow CPU weight transfers. "
                f"Free VRAM: {free_vram:.1f} GiB; need about {checkpoint_gb + safety_reserve_gb:.1f} GiB "
                "to keep the DiT on GPU with activation headroom. Unload other GPU models/apps and retry."
            )

        # Keep the packed weights on CPU while conditioning models run, then
        # transfer them to CUDA once before sampling. Load auxiliary encoders
        # on CPU so they never compete with the DiT for VRAM.
        vram.update({
            "offload_device": "cpu",
            "onload_device": "cuda",
            "preparing_device": "cuda",
        })
        print(
            f"[SCAIL2] GPU-resident NVFP4 DiT: checkpoint={checkpoint_gb:.1f} GiB, "
            f"free={free_vram:.1f} GiB, activation reserve≥{safety_reserve_gb:.1f} GiB",
            flush=True,
        )
    elif dit_label.startswith("INT8 ConvRot"):
        # Match ComfyUI's low-VRAM patcher: keep quantized layers on CPU until
        # the per-layer GPU budget permits caching them on CUDA. This avoids
        # copying every Linear weight for every attention sub-call.
        vram.update({
            "offload_device": "cpu",
            "onload_device": "cpu",
            "preparing_device": "cuda",
        })
        print(
            f"[SCAIL2] Comfy-style INT8 layer residency: free={free_vram:.1f} GiB, "
            f"GPU cache limit={max(1.0, total_vram - 2.0):.1f} GiB",
            flush=True,
        )

    model_configs = [
        # SCAIL2Pipeline inspects actual safetensors markers/metadata to build
        # single- or mixed-format configs (NVFP4+MXFP8).
        ModelConfig(path=str(dit_path), **vram),
        ModelConfig(path=str(SCAIL2_STATIC_PATHS["text_encoder"]), computation_device="cpu"),
        ModelConfig(path=str(SCAIL2_STATIC_PATHS["vae"]), computation_device="cpu"),
        ModelConfig(path=str(SCAIL2_STATIC_PATHS["clip_vision"]), computation_device="cpu"),
    ]
    tokenizer_config = ModelConfig(
        model_id="google/umt5-xxl",
        origin_file_pattern="spiece.model",
    )

    sam3_path = str(SCAIL2_STATIC_PATHS["sam3"]) if SCAIL2_STATIC_PATHS["sam3"].exists() else None
    lightx2v_path = str(SCAIL2_STATIC_PATHS["lightx2v_lora"]) if SCAIL2_STATIC_PATHS["lightx2v_lora"].exists() else None
    dpo_path = str(SCAIL2_STATIC_PATHS["dpo_lora"]) if SCAIL2_STATIC_PATHS["dpo_lora"].exists() else None

    pipe = SCAIL2Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        sam3_model_path=sam3_path,
        sam3_comfy_root=str(COMFY_ROOT),
        sam3_comfy_python=str(COMFY_PYTHON),
        sam3_worker_path=str(plugin_dir / "scripts" / "scail2_comfy_sam3_worker.py"),
        lightx2v_lora_path=lightx2v_path,
        dpo_lora_path=dpo_path,
        vram_limit=max(1.0, total_vram - 2.0),
    )
    pipe.dit_gpu_resident = dit_label.startswith("NVFP4+MXFP8")
    if pipe.dit_gpu_resident:
        pipe.dit_checkpoint_gb = dit_path.stat().st_size / (1024 ** 3)
    _pipeline_cache[dit_label] = pipe
    return pipe


def generate_scail2(
    reference_image: Image.Image,
    pose_video,
    prompt: str,
    negative_prompt: str,
    mode: str,
    dit_format: str,
    sampling_steps: int,
    guide_scale: float,
    shift: float,
    pose_strength: float,
    pose_start: float,
    pose_end: float,
    seed: int,
    output_width: int,
    output_height: int,
    use_sam3: bool,
    sam3_point_x: float,
    sam3_point_y: float,
    progress=gr.Progress(),
):
    """运行 SCAIL-2 生成。"""
    if reference_image is None:
        raise gr.Error("请上传参考图")
    if pose_video is None:
        raise gr.Error("请上传驱动视频")

    # 尺寸对齐：SCAIL-2 latent 要求可被 16 整除
    output_width = max(256, (int(output_width) // 16) * 16)
    output_height = max(256, (int(output_height) // 16) * 16)

    if isinstance(pose_video, (tuple, list)):
        video_path = pose_video[0]
    else:
        video_path = pose_video

    import imageio.v3 as iio
    import numpy as np
    try:
        source_fps = float(iio.immeta(video_path).get("fps", 30.0))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
    except Exception:
        source_fps = 30.0
    pose_frames_np = iio.imread(video_path)
    if pose_frames_np.ndim == 3:
        pose_frames_np = pose_frames_np[np.newaxis, ...]

    # VAE stride=4 → T ≡ 1 mod 4
    total_frames = pose_frames_np.shape[0]
    valid_T = ((total_frames - 1) // 4) * 4 + 1
    if total_frames != valid_T:
        pose_frames_np = pose_frames_np[:valid_T]

    from PIL import Image
    target_size = (output_width, output_height)
    if reference_image.size != target_size:
        reference_image = reference_image.resize(target_size, Image.Resampling.LANCZOS)
    resized_frames = []
    for frame in pose_frames_np:
        fi = Image.fromarray(frame)
        if fi.size != target_size:
            fi = fi.resize(target_size, Image.Resampling.BILINEAR)
        resized_frames.append(np.asarray(fi))
    pose_frames_np = np.stack(resized_frames, axis=0)

    if valid_T <= 81:
        segment_len = valid_T
        segment_overlap = 0
    else:
        # Match the official ComfyUI SCAIL-2 workflow: 81-frame window,
        # 5-frame overlap, and 76-frame pose offset per continuation.
        segment_len = 81
        segment_overlap = 5

    pose_video_frames = [Image.fromarray(f) for f in pose_frames_np]
    replace_flag = (mode == "替换模式")

    import torch
    pose_tensors = []
    for frame in pose_video_frames:
        if isinstance(frame, Image.Image):
            arr = np.array(frame).astype(np.float32) / 127.5 - 1.0
            pose_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    pose_video_tensor = torch.stack(pose_tensors) if pose_tensors else None
    if pose_video_tensor is None:
        raise gr.Error("无法从视频中提取帧")

    sam3_points = None
    if use_sam3 and replace_flag:
        w, h = reference_image.size
        sam3_points = [[sam3_point_x * w, sam3_point_y * h]]

    progress(0, desc=f"加载 SCAIL-2 {dit_format} ...")
    logging.getLogger("modelscope_hub.download").setLevel(logging.WARNING)
    with _quiet_generation_console():
        pipe = get_pipeline(dit_format)

    progress(0.1, desc="生成中...")
    with _quiet_generation_console():
        frames = pipe(
            prompt=prompt or "",
            negative_prompt=negative_prompt or "",
            reference_image=reference_image,
            pose_video_frames=pose_video_tensor,
            replace_flag=replace_flag,
            segment_len=int(segment_len),
            segment_overlap=int(segment_overlap),
            shift=float(shift),
            sampling_steps=int(sampling_steps),
            guide_scale=float(guide_scale),
            seed=int(seed) if seed >= 0 else -1,
            pose_strength=float(pose_strength),
            pose_start=float(pose_start),
            pose_end=float(pose_end),
            use_sam3=use_sam3,
            sam3_points=sam3_points,
            progress_callback=progress,
        )

    if not frames:
        raise gr.Error("SCAIL-2 已完成采样，但没有返回可保存的视频帧")

    output_dir = WEBUI_ROOT / "outputs" / "scail2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scail2_{uuid.uuid4().hex[:12]}.mp4"
    muxed_path = output_path.with_name(f"{output_path.stem}_with_audio.mp4")
    progress(0.98, desc="编码并保存视频...")
    try:
        with _quiet_generation_console():
            iio.imwrite(
                output_path,
                np.stack([np.asarray(frame.convert("RGB")) for frame in frames]),
                fps=source_fps,
                codec="libx264",
                quality=8,
            )
        # The generated frames replace only the video stream. Keep the
        # original driving video's audio, clipped to the generated duration.
        import imageio_ffmpeg
        duration = len(frames) / source_fps
        mux_result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(output_path),
                "-i", str(video_path),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{duration:.6f}",
                "-movflags", "+faststart",
                str(muxed_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if mux_result.returncode != 0:
            raise RuntimeError(mux_result.stderr.strip() or "FFmpeg 音视频合并失败")
        os.replace(muxed_path, output_path)
    except Exception as exc:
        try:
            muxed_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise gr.Error(f"采样已完成，但 MP4 编码/音频合并失败：{type(exc).__name__}: {exc}") from exc

    progress(1.0, desc="完成")
    return str(output_path)


def create_scail2_ui():
    """创建 SCAIL-2 Gradio UI。"""
    avail = _list_available_dit_files()
    if not avail:
        default_label = _auto_default_label()
        dit_choices = [default_label]
    else:
        dit_choices = [lbl for lbl, _ in avail]
        default_label = _auto_default_label()
        if default_label not in dit_choices:
            default_label = dit_choices[0]

    with gr.Column():
        gr.Markdown("## SCAIL-2 角色动画 / 替换")
        gr.Markdown(
            "基于 Wan2.1 I2V 的端到端角色动画模型。\n"
            "- **动画模式**：参考角色执行驱动视频的动作\n"
            "- **替换模式**：将驱动视频中的人物替换为参考角色（需 SAM3）"
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_image = gr.Image(label="参考图", type="pil", height=300)
                pose_video = gr.Video(label="驱动视频")

                dit_format = gr.Dropdown(
                    choices=dit_choices,
                    value=default_label,
                    label="SCAIL-2 DiT 量化格式",
                    info=f"已检测到 {len(dit_choices)} 个 checkpoint；重启 WebUI 或下载新文件后刷新此列表",
                )

                mode = gr.Radio(
                    ["动画模式", "替换模式"],
                    value="替换模式",
                    label="模式",
                )

                with gr.Accordion("高级参数", open=False):
                    prompt = gr.Textbox(label="提示词", value="")
                    negative_prompt = gr.Textbox(label="负面提示词", value="")
                    sampling_steps = gr.Slider(1, 50, value=6, step=1, label="SCAIL-2 采样步数")
                    guide_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="SCAIL-2 CFG 比例")
                    shift = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Shift")
                    pose_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pose 强度")
                    pose_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Pose 起始")
                    pose_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Pose 结束")
                    seed = gr.Number(label="随机种子", value=-1, precision=0)
                    with gr.Row():
                        output_width = gr.Slider(256, 1536, value=832, step=16, label="SCAIL-2 输出宽度")
                        output_height = gr.Slider(256, 1536, value=480, step=16, label="SCAIL-2 输出高度")

                with gr.Accordion("SAM3 跟踪（替换模式）", open=False):
                    use_sam3 = gr.Checkbox(label="启用 SAM3 目标跟踪", value=True)
                    sam3_point_x = gr.Slider(0.0, 1.0, value=0.5, label="目标点 X（归一化）")
                    sam3_point_y = gr.Slider(0.0, 1.0, value=0.5, label="目标点 Y（归一化）")

                generate_btn = gr.Button("生成", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="输出视频")
                status = gr.Textbox(label="状态", value="")

        generate_btn.click(
            fn=generate_scail2,
            inputs=[
                ref_image, pose_video, prompt, negative_prompt, mode,
                dit_format,
                sampling_steps, guide_scale,
                shift, pose_strength, pose_start, pose_end, seed,
                output_width, output_height,
                use_sam3, sam3_point_x, sam3_point_y,
            ],
            outputs=[output_video],
        )
