from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from modules import scripts_postprocessing

_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from te_dlss5 import frame_guidance, native_backend


def _find_ffmpeg() -> str | None:
    """在 PATH 和常见位置查找 ffmpeg.exe。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for candidate in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _run_ffmpeg(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return proc.returncode, proc.stdout, proc.stderr


class TEDLSS5PostprocessingScript(scripts_postprocessing.ScriptPostprocessing):
    name = "TE DLSS5"
    order = 150

    def __init__(self):
        self._enhancer = None
        self._guidance = None
        self._size = None
        self._config = None

    def ui(self):
        with gr.Accordion("TE DLSS5 神经画质处理", open=False):
            enabled = gr.Checkbox(label="启用 TE DLSS5", value=False)
            mode = gr.Dropdown(
                label="处理模式",
                choices=[
                    ("单图 / 独立帧", "still"),
                    ("视频 / 连续帧", "video"),
                ],
                value="still",
                info="处理视频时选择“视频 / 连续帧”，以保留运动与时序历史。",
            )
            style = gr.Dropdown(
                label="画面风格",
                choices=["default", "natural", "cinematic"],
                value="default",
            )
            guidance = gr.Dropdown(
                label="引导方式",
                choices=[
                    ("无引导，兼容性最好", "zero"),
                    ("NVOF 光流", "nvof"),
                    ("NVOF + Depth Anything V2", "nvof_depth"),
                ],
                value="zero",
                info="视频建议使用 nvof；nvof_depth 需要深度模型与额外显存。",
            )
            with gr.Row():
                intensity = gr.Slider(label="整体强度", minimum=-1, maximum=2, step=0.05, value=1)
                local_tone = gr.Slider(label="局部色调", minimum=-1, maximum=2, step=0.05, value=1)
                local_structure = gr.Slider(label="局部结构", minimum=-1, maximum=2, step=0.05, value=1)
            depth_interval = gr.Slider(
                label="深度推理间隔",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
                info="仅 nvof_depth 生效；数值越大速度越快。",
            )
            gr.Markdown(
                "后端读取当前 WebUI 插件目录中的 `te_dlss5_native.dll` 和 DLSSNR 运行时。"
            )

            # ===== 视频文件批量处理 =====
            with gr.Accordion("🎬 视频文件处理（上传视频逐帧增强）", open=False):
                gr.Markdown(
                    "上传视频文件，逐帧应用 TE DLSS5 神经画质增强后合成新视频（保留音频）。"
                    "处理模式请在上方选择「视频 / 连续帧」以获得最佳时序效果。"
                )
                video_input = gr.Video(label="输入视频")
                with gr.Row():
                    video_fps = gr.Number(
                        label="输出帧率（留空=沿用原视频）", value=0, precision=0, minimum=0, maximum=120
                    )
                    video_crf = gr.Slider(
                        label="视频质量 CRF（越小越清晰）", minimum=0, maximum=51, step=1, value=18
                    )
                video_process_btn = gr.Button("🚀 开始处理视频", variant="primary")
                video_output = gr.Video(label="输出视频")
                video_status = gr.Textbox(label="处理状态", interactive=False)

                video_process_btn.click(
                    fn=self._process_video_file,
                    inputs=[
                        video_input, mode, style, guidance,
                        intensity, local_tone, local_structure,
                        depth_interval, video_fps, video_crf,
                    ],
                    outputs=[video_output, video_status],
                )

        return {
            "enabled": enabled,
            "mode": mode,
            "style": style,
            "guidance": guidance,
            "intensity": intensity,
            "local_tone": local_tone,
            "local_structure": local_structure,
            "depth_interval": depth_interval,
        }

    @staticmethod
    def _settings(style, guidance, intensity, local_tone, local_structure):
        return {
            "style": str(style),
            "intensity": max(-1.0, min(2.0, float(intensity))),
            "localToneStrength": max(-1.0, min(2.0, float(local_tone))),
            "localStructureStrength": max(-1.0, min(2.0, float(local_structure))),
            "guidance": str(guidance),
            "runtimeDir": str(native_backend.resolve_runtime_dir()),
        }

    def _close_backend(self):
        if self._guidance is not None:
            self._guidance.close()
            self._guidance = None
        if self._enhancer is not None:
            self._enhancer.close()
            self._enhancer = None
        self._size = None
        self._config = None

    def _process_frame(self, image: Image.Image, mode: str, style: str, guidance: str,
                       intensity: float, local_tone: float, local_structure: float,
                       depth_interval: int) -> Image.Image:
        source = image.convert("RGBA")
        frame = np.ascontiguousarray(np.asarray(source, dtype=np.uint8))
        height, width = frame.shape[:2]
        config = self._settings(style, guidance, intensity, local_tone, local_structure)

        # A still frame gets a fresh temporal context. Video mode keeps one
        # enhancer/guidance pair alive across successive WebUI video frames.
        if mode == "still":
            self._close_backend()
            backend = native_backend.resolve_backend()
            if not backend:
                raise RuntimeError("TE DLSS5 native backend not found")
            with native_backend.NativeEnhancer(backend, width, height, config) as enhancer:
                if guidance == "zero":
                    result = enhancer.process(frame.tobytes(order="C"))
                else:
                    with frame_guidance.FrameGuidance(
                        width, height, guidance, config["runtimeDir"], depth_interval=1
                    ) as guide:
                        motion, depth = guide.next(frame)
                        if guide.consume_reset():
                            enhancer.reset()
                        result = enhancer.process(frame.tobytes(order="C"), motion, depth)
        else:
            if self._size != (width, height) or self._config != config:
                self._close_backend()
                backend = native_backend.resolve_backend()
                if not backend:
                    raise RuntimeError("TE DLSS5 native backend not found")
                self._enhancer = native_backend.NativeEnhancer(backend, width, height, config)
                self._guidance = (
                    None if guidance == "zero" else frame_guidance.FrameGuidance(
                        width, height, guidance, config["runtimeDir"], depth_interval=int(depth_interval)
                    )
                )
                self._size = (width, height)
                self._config = config
            motion = depth = None
            if self._guidance is not None:
                motion, depth = self._guidance.next(frame)
                if self._guidance.consume_reset():
                    self._enhancer.reset()
            result = self._enhancer.process(frame.tobytes(order="C"), motion, depth)

        output = np.frombuffer(result, dtype=np.uint8).reshape((height, width, 4))
        return Image.fromarray(output, mode="RGBA").convert(image.mode if image.mode in ("RGB", "RGBA") else "RGB")

    def process(self, pp, **kwargs):
        if not kwargs.get("enabled", False):
            return
        pp.image = self._process_frame(
            pp.image,
            kwargs.get("mode", "still"),
            kwargs.get("style", "default"),
            kwargs.get("guidance", "zero"),
            kwargs.get("intensity", 1.0),
            kwargs.get("local_tone", 1.0),
            kwargs.get("local_structure", 1.0),
            int(kwargs.get("depth_interval", 4)),
        )
        pp.nametags.append("dlss5")
        pp.info["TE DLSS5"] = (
            f"mode={kwargs.get('mode', 'still')}; guidance={kwargs.get('guidance', 'zero')}; "
            f"style={kwargs.get('style', 'default')}; intensity={float(kwargs.get('intensity', 1.0)):.2f}"
        )

    def image_changed(self):
        self._close_backend()

    # ===== 视频文件处理 =====

    def _process_video_file(self, video_path, mode, style, guidance,
                            intensity, local_tone, local_structure,
                            depth_interval, out_fps, crf):
        """上传视频文件：拆帧 → 逐帧 DLSS5 → 合成视频（保留音频）。"""
        if not video_path:
            yield None, "❌ 请先上传视频文件"
            return

        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            yield None, "❌ 未找到 ffmpeg，请将 ffmpeg.exe 加入 PATH 或放到 C:\\ffmpeg\\bin\\"
            return

        backend = native_backend.resolve_backend()
        if not backend:
            yield None, "❌ TE DLSS5 原生后端未找到（te_dlss5_native.dll）"
            return

        src = Path(video_path)
        if not src.is_file():
            yield None, f"❌ 视频文件不存在：{video_path}"
            return

        yield None, f"🔍 正在探测视频信息：{src.name} ..."

        # 探测帧率和分辨率
        probe_cmd = [
            ffmpeg, "-i", str(src),
            "-hide_banner", "-loglevel", "error",
        ]
        _rc, _out, err = _run_ffmpeg(probe_cmd)
        # ffmpeg -i 无输出文件时返回非 0，从 stderr 解析
        fps_val = 0.0
        for line in err.splitlines():
            line = line.strip()
            if "Stream #" in line and "Video:" in line:
                # 形如:  Stream #0:0: Video: ..., 30 fps, ...
                import re
                m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
                if m:
                    fps_val = float(m.group(1))
                break
        if out_fps and float(out_fps) > 0:
            fps_val = float(out_fps)
        if fps_val <= 0:
            fps_val = 30.0

        work_dir = Path(tempfile.mkdtemp(prefix="dlss5_video_"))
        frames_in = work_dir / "in"
        frames_out = work_dir / "out"
        frames_in.mkdir()
        frames_out.mkdir()

        try:
            yield None, f"🎬 正在拆帧（{fps_val:.2f} fps）..."
            rc, out, err = _run_ffmpeg([
                ffmpeg, "-i", str(src),
                "-vsync", "0",
                str(frames_in / "frame_%06d.png"),
            ])
            if rc != 0:
                yield None, f"❌ 拆帧失败：{err[-500:]}"
                return

            frame_files = sorted(frames_in.glob("frame_*.png"))
            total = len(frame_files)
            if total == 0:
                yield None, "❌ 未提取到任何帧"
                return

            yield None, f"⚙️  初始化 DLSS5 后端（共 {total} 帧）..."

            # 用第一帧确定尺寸并初始化后端（video 模式保持时序）
            first_img = Image.open(frame_files[0])
            width, height = first_img.size
            config = self._settings(style, guidance, intensity, local_tone, local_structure)

            enhancer = native_backend.NativeEnhancer(backend, width, height, config)
            guide = (
                None if guidance == "zero" else frame_guidance.FrameGuidance(
                    width, height, guidance, config["runtimeDir"], depth_interval=int(depth_interval)
                )
            )

            try:
                for idx, fp in enumerate(frame_files, 1):
                    img = Image.open(fp).convert("RGBA")
                    frame = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
                    motion = depth = None
                    if guide is not None:
                        motion, depth = guide.next(frame)
                        if guide.consume_reset():
                            enhancer.reset()
                    result = enhancer.process(frame.tobytes(order="C"), motion, depth)
                    out_arr = np.frombuffer(result, dtype=np.uint8).reshape((height, width, 4))
                    Image.fromarray(out_arr, mode="RGBA").convert("RGB").save(
                        frames_out / fp.name
                    )
                    if idx % 10 == 0 or idx == total:
                        yield None, f"🔧 处理中：{idx}/{total} 帧"
            finally:
                if guide is not None:
                    guide.close()
                enhancer.close()

            yield None, "🎞️  正在合成视频（保留音频）..."

            out_video = work_dir / f"dlss5_{src.stem}.mp4"
            # 合成：处理后的帧 + 原视频音频
            rc, out, err = _run_ffmpeg([
                ffmpeg, "-y",
                "-framerate", str(fps_val),
                "-i", str(frames_out / "frame_%06d.png"),
                "-i", str(src),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-crf", str(int(crf)),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(out_video),
            ])
            if rc != 0:
                # 无音频时回退：仅视频
                rc, out, err = _run_ffmpeg([
                    ffmpeg, "-y",
                    "-framerate", str(fps_val),
                    "-i", str(frames_out / "frame_%06d.png"),
                    "-c:v", "libx264", "-crf", str(int(crf)),
                    "-pix_fmt", "yuv420p",
                    str(out_video),
                ])
                if rc != 0:
                    yield None, f"❌ 合成视频失败：{err[-500:]}"
                    return

            if not out_video.is_file():
                yield None, "❌ 合成视频失败：未生成输出文件"
                return

            yield str(out_video), f"✅ 处理完成：{total} 帧，输出 {out_video.name}"

        finally:
            # 清理临时帧目录（保留输出视频供 Gradio 读取，由调用方/系统后续清理）
            try:
                shutil.rmtree(frames_in, ignore_errors=True)
                shutil.rmtree(frames_out, ignore_errors=True)
            except Exception:
                pass
