from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from modules import scripts_postprocessing

_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from te_dlss5 import frame_guidance, native_backend


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
