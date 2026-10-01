"""Per-frame motion and depth guides for the TE DLSSNR bridge.

This module deliberately has no dependency on Magpie's implementation.  It
produces the two packed planes required by our private bridge ABI:

* motion: ``H x W x 2`` float16, measured in source pixels
* depth: ``H x W`` float32, normalized inverse-relative depth

The NVIDIA Optical Flow SDK is not part of ComfyUI, so OpenCV Farneback is a
portable fallback. Depth Anything V2 is optional: when the ControlNet Aux
package and its checkpoint are available, its PyTorch model is used on
ComfyUI's CUDA device. The bundled ONNX Runtime model remains a fallback for
installations without a usable PyTorch CUDA device.
"""

from __future__ import annotations

import os
import ctypes
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")


class GuidanceError(RuntimeError):
    """Raised when a requested guide cannot be produced."""


class _CudaNvof:
    """Adapter for the official CUDA Optical Flow SDK bridge."""

    def __init__(self, width: int, height: int):
        if os.name != "nt":
            raise GuidanceError("CUDA NVOF guidance requires Windows")
        root = Path(__file__).resolve().parent
        candidates = []
        configured = os.environ.get("TE_NVOF_DLL", "").strip()
        if configured:
            candidates.append(Path(configured))
        candidates.extend((root / "te_nvof_cuda.dll", root / "native" / "te_nvof_cuda.dll"))
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            raise GuidanceError(
                "CUDA NVOF bridge is not built; run native\\build_nvof.bat "
                "from a Visual Studio Developer Command Prompt"
            )
        try:
            self._dll_dirs = []
            if hasattr(os, "add_dll_directory"):
                cuda_roots = [os.environ.get("CUDA_PATH", "").strip(),
                              os.environ.get("CUDA_PATH_V13_1", "").strip(),
                              r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"]
                for cuda_root in cuda_roots:
                    if not cuda_root:
                        continue
                    cuda_bin = Path(cuda_root) / "bin"
                    if cuda_bin.is_dir():
                        try:
                            self._dll_dirs.append(os.add_dll_directory(str(cuda_bin)))
                        except OSError:
                            pass
                        break
            self.dll = ctypes.WinDLL(str(path))
            self.create = self._bind("te_nvof_create", ctypes.c_void_p,
                                     [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p])
            self.process = self._bind("te_nvof_process_rgba", ctypes.c_int,
                                      [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32])
            self.reset_fn = self._bind("te_nvof_reset", None, [ctypes.c_void_p])
            self.destroy = self._bind("te_nvof_destroy", None, [ctypes.c_void_p])
            self.info_fn = getattr(self.dll, "te_nvof_info", None)
            if self.info_fn is not None:
                self.info_fn.restype = ctypes.c_char_p
                self.info_fn.argtypes = [ctypes.c_void_p]
            self.last_error = getattr(self.dll, "te_nvof_last_error", None)
            if self.last_error is not None:
                self.last_error.restype = ctypes.c_char_p
                self.last_error.argtypes = []
            self.handle = self.create(width, height, b"{}")
        except (AttributeError, OSError) as exc:
            raise GuidanceError(f"Could not load CUDA NVOF bridge {path}: {exc}") from exc
        if not self.handle:
            detail = ""
            if self.last_error is not None:
                detail = (self.last_error() or b"").decode("utf-8", "replace")
            for handle in self._dll_dirs:
                handle.close()
            self._dll_dirs = []
            raise GuidanceError(f"CUDA NVOF initialization failed{': ' + detail if detail else ''}")
        self.path = path
        details = ""
        if self.info_fn is not None:
            try:
                details = (self.info_fn(self.handle) or b"").decode("utf-8", "replace")
            except Exception:
                details = ""
        LOGGER.info("[TE DLSS5] NVOF ready: dll=%s, size=%sx%s%s", path, width, height,
                    ", " + details if details else "")
        self.frame_bytes = width * height * 4
        self.guide_bytes = width * height * 4
        self.confidence_bytes = width * height

    def _bind(self, name, restype, argtypes):
        fn = getattr(self.dll, name)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def next(self, frame: bytes):
        src = ctypes.create_string_buffer(frame)
        motion = ctypes.create_string_buffer(self.guide_bytes)
        confidence = ctypes.create_string_buffer(self.confidence_bytes)
        status = self.process(self.handle, src, motion, confidence, self.frame_bytes)
        if status != 0:
            detail = ""
            if self.last_error is not None:
                detail = (self.last_error() or b"").decode("utf-8", "replace")
            raise GuidanceError(f"CUDA NVOF frame processing failed ({status}){': ' + detail if detail else ''}")
        return motion.raw, confidence.raw

    def reset(self):
        if self.handle:
            self.reset_fn(self.handle)

    def close(self):
        if getattr(self, "handle", None):
            self.destroy(self.handle)
            self.handle = None
        for handle in getattr(self, "_dll_dirs", []):
            handle.close()
        self._dll_dirs = []


def _load_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise GuidanceError("NumPy is required for DLSSNR frame guidance") from exc
    return np


def _load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise GuidanceError(
            "Motion guidance requires opencv-python-headless in the ComfyUI environment"
        ) from exc
    return cv2


class _DepthModel:
    def __init__(self, runtime_root: Path):
        np = _load_numpy()
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise GuidanceError(
                "Depth guidance requires onnxruntime; install onnxruntime-gpu or "
                "onnxruntime-directml in ComfyUI's Python environment"
            ) from exc

        model = runtime_root.parent / "frame_guidance" / "DepthAnythingV2" / "model_fp16.onnx"
        if not model.is_file():
            raise GuidanceError(f"Depth Anything V2 model is missing: {model}")

        self._dll_handles = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in (
                runtime_root.parent / "frame_guidance" / "TensorRT",
                runtime_root.parent / "frame_guidance" / "DirectML",
            ):
                if directory.is_dir():
                    try:
                        self._dll_handles.append(os.add_dll_directory(str(directory)))
                    except OSError:
                        pass

        available = set(ort.get_available_providers())
        preferred = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [name for name in preferred if name in available]
        if not providers:
            raise GuidanceError(
                f"ONNX Runtime has no usable execution provider; available={sorted(available)}"
            )
        # ORT 1.24 can fail while applying SimplifiedLayerNormFusion to this
        # exported FP16 graph. Start with all graph rewrites disabled; this is
        # slightly slower to initialize but avoids a known invalid-node pass.
        # Basic optimization remains a fallback for older ORT builds where the
        # disable-all enum is unavailable.
        last_error = None
        levels = [
            getattr(ort.GraphOptimizationLevel, "ORT_DISABLE_ALL", None),
            getattr(ort.GraphOptimizationLevel, "ORT_ENABLE_BASIC", None),
        ]
        for level in levels:
            if level is None:
                continue
            try:
                options = ort.SessionOptions()
                options.graph_optimization_level = level
                self.session = ort.InferenceSession(
                    str(model), sess_options=options, providers=providers
                )
                self.optimization = str(level)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise GuidanceError(
                f"Could not initialize Depth Anything V2 with providers {providers}: {last_error}"
            ) from last_error
        inputs = self.session.get_inputs()
        if not inputs:
            raise GuidanceError("Depth Anything V2 ONNX model has no input")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.np = np
        self.cv2 = _load_cv2()
        self.model_path = model
        self.provider = ",".join(self.session.get_providers())
        LOGGER.info(
            "[TE DLSS5] Depth model ready: model=%s, providers=%s, graph_optimization=%s",
            model, self.provider, self.optimization,
        )

    def run(self, rgb):
        np = self.np
        cv2 = self.cv2
        # Match the model's long-side policy while keeping ViT patch dimensions
        # aligned.  This avoids the strong geometric distortion of a forced
        # square input on widescreen video.
        fixed_h = self.input_shape[-2] if len(self.input_shape) >= 4 else None
        fixed_w = self.input_shape[-1] if len(self.input_shape) >= 4 else None
        if isinstance(fixed_h, int) and isinstance(fixed_w, int) and fixed_h > 0 and fixed_w > 0:
            input_h, input_w = fixed_h, fixed_w
        else:
            scale = 336.0 / max(rgb.shape[0], rgb.shape[1])
            input_h = max(14, int(round(rgb.shape[0] * scale / 14.0)) * 14)
            input_w = max(14, int(round(rgb.shape[1] * scale / 14.0)) * 14)
        sample = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_AREA)
        sample = sample.astype(np.float32) / 255.0
        sample = (sample - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = np.transpose(sample, (2, 0, 1))[None, ...]
        result = self.session.run(None, {self.input_name: tensor})
        if not result:
            raise GuidanceError("Depth Anything V2 returned no output")
        depth = np.asarray(result[0], dtype=np.float32)
        while depth.ndim > 2:
            depth = depth[0]
        if depth.ndim != 2:
            raise GuidanceError(f"Unexpected depth output shape: {depth.shape}")
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        # DLSSNR expects a stable normalized inverse-relative depth signal.
        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        values = depth[finite]
        lo, hi = np.percentile(values, (1.0, 99.0))
        if hi <= lo + 1e-6:
            return np.zeros(depth.shape, dtype=np.float32)
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        return np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def close(self):
        # Drop the session before unloading provider DLL search paths.
        self.session = None
        for handle in self._dll_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._dll_handles.clear()


class _TorchDepthModel:
    """Adapter for the installed ControlNet-Aux PyTorch Depth Anything V2."""

    def __init__(self):
        np = _load_numpy()
        try:
            import torch
            import comfy.model_management as model_management
        except ImportError as exc:
            raise GuidanceError("PyTorch/ComfyUI model management is unavailable") from exc
        if not torch.cuda.is_available():
            raise GuidanceError("PyTorch CUDA is not available")

        try:
            from .te_depth_anything_v2 import (  # type: ignore
                DepthAnythingV2, model_configs,
            )
        except ImportError as exc:
            raise GuidanceError(
                "Could not import the TE-local Depth Anything V2 model"
            ) from exc

        filename = os.environ.get(
            "TE_DLSS5_DEPTH_CHECKPOINT", "depth_anything_v2_vits.pth"
        ).strip() or "depth_anything_v2_vits.pth"
        if filename not in model_configs:
            raise GuidanceError(
                f"Unsupported TE_DLSS5_DEPTH_CHECKPOINT={filename}; "
                f"choose one of {sorted(model_configs)}"
            )
        checkpoint = (
            Path(__file__).resolve().parent / "runtime" / "frame_guidance"
            / "DepthAnythingV2" / "model" / filename
        )
        if not checkpoint.is_file():
            raise GuidanceError(
                f"TE-local Depth Anything V2 checkpoint is missing: {checkpoint}"
            )
        try:
            model = DepthAnythingV2(**model_configs[filename])
            state = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state)
            self.device = model_management.get_torch_device()
            model.eval().to(self.device)
        except Exception as exc:
            raise GuidanceError(f"Could not load CUDA Depth Anything V2 checkpoint: {exc}") from exc

        self.np = np
        self.torch = torch
        self.model = model
        self.model_path = checkpoint
        self.provider = f"PyTorch/{self.device}"
        self.cv2 = _load_cv2()
        from .te_depth_anything_v2.util.transform import (  # type: ignore
            NormalizeImage, PrepareForNet, Resize,
        )
        from torchvision.transforms import Compose

        # Build the CPU-side preprocessing graph once instead of once per
        # frame. This matters for long video batches.
        try:
            input_size = int(os.environ.get("TE_DLSS5_DEPTH_INPUT_SIZE", "518"))
        except (TypeError, ValueError):
            input_size = 518
        input_size = max(224, min(1024, input_size))
        input_size -= input_size % 14
        if input_size < 224:
            input_size = 224
        self.input_size = input_size
        self.transform = Compose([
            Resize(width=input_size, height=input_size, resize_target=False,
                   keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method="lower_bound",
                   image_interpolation_method=self.cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        precision = os.environ.get("TE_DLSS5_DEPTH_FP16", "auto").strip().lower()
        if precision not in {"auto", "1", "on", "true", "yes", "0", "off", "false", "no"}:
            raise GuidanceError(f"Unknown TE_DLSS5_DEPTH_FP16={precision}; use auto or 0")
        self.autocast = precision not in {"0", "off", "false", "no"}
        self.device_type = torch.device(self.device).type
        LOGGER.info(
            "[TE DLSS5] Depth model ready: backend=%s, model=%s, device=%s, input=%s, autocast_fp16=%s",
            self.provider, checkpoint, self.device, self.input_size, self.autocast,
        )

    @staticmethod
    def _normalize(depth, np):
        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros(depth.shape, dtype=np.float32)
        values = depth[finite]
        lo, hi = np.percentile(values, (1.0, 99.0))
        if hi <= lo + 1e-6:
            return np.zeros(depth.shape, dtype=np.float32)
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        return np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def run(self, rgb):
        np = self.np
        torch = self.torch
        # ComfyUI IMAGE frames are RGB, matching the aux detector after its
        # external BGR-to-RGB conversion.
        image = np.ascontiguousarray(rgb[..., :3]).astype(np.float32) / 255.0
        array = self.transform({"image": image})["image"]
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.autocast and self.device_type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    depth = self.model(tensor, 1.0)
                    depth = torch.nn.functional.interpolate(
                        depth[:, None], (rgb.shape[0], rgb.shape[1]),
                        mode="bilinear", align_corners=True,
                    )[0, 0]
            else:
                depth = self.model(tensor, 1.0)
                depth = torch.nn.functional.interpolate(
                    depth[:, None], (rgb.shape[0], rgb.shape[1]),
                    mode="bilinear", align_corners=True,
                )[0, 0]
        return self._normalize(depth.float().detach().cpu().numpy(), np)

    def close(self):
        self.model = None
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass


def _create_depth_model(runtime_dir: Path):
    """Prefer CUDA PyTorch, with ONNX as the automatic fallback."""
    backend = os.environ.get("TE_DLSS5_DEPTH_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "torch", "onnx"}:
        raise GuidanceError(
            f"Unknown TE_DLSS5_DEPTH_BACKEND={backend}; use auto, torch, or onnx"
        )
    if backend in {"auto", "torch"}:
        try:
            return _TorchDepthModel()
        except Exception as exc:
            if backend == "torch":
                raise
            LOGGER.warning(
                "[TE DLSS5] CUDA Torch depth unavailable; falling back to ONNX: %s",
                exc,
            )
    return _DepthModel(runtime_dir)


class FrameGuidance:
    """Stateful guide producer for one sequential video stream."""

    def __init__(self, width: int, height: int, mode: str, runtime_dir: str, depth_interval: int = 4):
        self.width = int(width)
        self.height = int(height)
        self.mode = mode or "zero"
        self.depth_interval = max(1, min(8, int(depth_interval)))
        self.runtime_dir = Path(runtime_dir).expanduser()
        self.previous_gray = None
        self.previous_depth = None
        self._depth_residual_mean = 0.0
        self._depth_history_weight = 0.0
        self._reset_required = False
        self.frame_index = 0
        self.depth_model = None
        if self.mode not in {"zero", "motion", "depth", "motion_depth", "nvof", "nvof_depth"}:
            raise GuidanceError(f"Unknown guidance mode: {self.mode}")
        self.nvof = None
        try:
            self.nvof = _CudaNvof(width, height) if self.mode.startswith("nvof") else None
            if "depth" in self.mode:
                self.depth_model = _create_depth_model(self.runtime_dir)
        except Exception:
            if self.nvof is not None:
                self.nvof.close()
                self.nvof = None
            raise
        LOGGER.info(
            "[TE DLSS5] Guidance ready: mode=%s, motion=%s, depth=%s, confidence=%s, depth_interval=%s",
            self.mode, "CUDA NVOF" if self.nvof else ("OpenCV Farneback" if "motion" in self.mode else "disabled"),
            "Depth Anything V2" if self.depth_model else "disabled",
            "NVOF cost-derived R8" if self.nvof else "disabled",
            self.depth_interval,
        )

    def _motion(self, rgb):
        np = _load_numpy()
        cv2 = _load_cv2()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return np.zeros((self.height, self.width, 2), dtype=np.float16)
        # A large global change is usually a cut or seek.  Do not feed stale
        # temporal data into the neural filter across that boundary.
        delta = float(np.mean(cv2.absdiff(gray, self.previous_gray))) / 255.0
        if delta > 0.32:
            self.previous_gray = gray
            self.previous_depth = None
            self._reset_required = True
            return np.zeros((self.height, self.width, 2), dtype=np.float16)
        # Current -> previous displacement is the convention consumed by the
        # bridge.  The first frame intentionally has no temporal motion.
        flow = cv2.calcOpticalFlowFarneback(
            gray,
            self.previous_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self.previous_gray = gray
        return np.clip(flow, -32768.0, 32767.0).astype(np.float16)

    def _depth(self, rgb, motion, confidence=None):
        np = _load_numpy()
        cv2 = _load_cv2()
        if self.depth_model is None:
            return None
        warped = None
        valid = None
        if self.previous_depth is not None and motion is not None:
            yy, xx = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
            map_x = xx + motion[..., 0].astype(np.float32)
            map_y = yy + motion[..., 1].astype(np.float32)
            valid = (
                (map_x >= 0.0) & (map_x <= self.width - 1) &
                (map_y >= 0.0) & (map_y <= self.height - 1)
            )
            warped = cv2.remap(
                self.previous_depth,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        # Reusing a warped depth map between model evaluations is the same
        # temporal optimization used by real-time preprocessors. Interval 1
        # preserves the original per-frame behavior.
        if warped is not None and self.frame_index % self.depth_interval != 0:
            # Do not replicate history from outside the image after a large
            # motion vector. Invalid samples are neutralized instead of
            # smearing the last border pixel across a disocclusion.
            if valid is not None:
                warped = np.where(valid, warped, 0.0)
            self._depth_residual_mean = 0.0
            self._depth_history_weight = float(np.mean(valid)) if valid is not None else 1.0
            self.previous_depth = warped.astype(np.float32)
            return self.previous_depth

        current = self.depth_model.run(rgb)
        if warped is not None:
            if valid is not None:
                warped = np.where(valid, warped, 0.0)
            residual = np.abs(current - warped)
            residual_valid = valid if valid is not None else np.ones_like(current, dtype=bool)
            residual_values = residual[residual_valid]
            self._depth_residual_mean = float(np.mean(residual_values)) if residual_values.size else 0.0
            # Keep model updates responsive while damping per-frame depth noise.
            # Confidence and depth residual jointly decide how much history is
            # trusted; this rejects history at disocclusions and object edges.
            if confidence is not None:
                if isinstance(confidence, (bytes, bytearray, memoryview)):
                    expected = self.width * self.height
                    if len(confidence) != expected:
                        raise GuidanceError(
                            f"Unexpected confidence guide size {len(confidence)}; expected {expected}"
                        )
                    confidence = np.frombuffer(confidence, dtype=np.uint8).reshape((self.height, self.width))
                confidence_weight = np.asarray(confidence, dtype=np.float32) / 255.0
            else:
                confidence_weight = np.ones_like(current, dtype=np.float32)
            residual_weight = np.exp(-np.clip(residual, 0.0, 1.0) * 7.0)
            history_weight = 0.82 * confidence_weight * residual_weight
            if valid is not None:
                history_weight = np.where(valid, history_weight, 0.0)
            self._depth_history_weight = float(np.mean(history_weight))
            current = current * (1.0 - history_weight) + warped * history_weight
        self.previous_depth = current
        return current.astype(np.float32)

    def next(self, rgba: Any) -> tuple[bytes | None, bytes | None]:
        np = _load_numpy()
        frame = np.asarray(rgba)
        if frame.shape != (self.height, self.width, 4):
            raise GuidanceError(
                f"Expected RGBA frame {(self.height, self.width, 4)}, got {frame.shape}"
            )
        if self.mode == "zero":
            return None, None
        self.frame_index += 1
        rgb = frame[..., :3]
        confidence = None
        if self.nvof is not None:
            # NVOF keeps temporal hints internally. Reset that history at a
            # hard cut so vectors from the previous shot are never reused.
            gray = (rgb[..., 0].astype(np.float32) * 0.299 +
                    rgb[..., 1].astype(np.float32) * 0.587 +
                    rgb[..., 2].astype(np.float32) * 0.114)
            if self.previous_gray is not None:
                delta = float(np.mean(np.abs(gray - self.previous_gray))) / 255.0
                if delta > 0.32:
                    self.nvof.reset()
                    self.previous_depth = None
                    self._reset_required = True
            self.previous_gray = gray
            motion_bytes, confidence_bytes = self.nvof.next(frame.tobytes(order="C"))
            motion = np.frombuffer(motion_bytes, dtype=np.float16).reshape((self.height, self.width, 2))
            # The native bridge returns confidence as a packed R8 byte plane.
            # Decode it before depth fusion; passing the raw bytes to NumPy's
            # float conversion makes it try to parse the entire image buffer.
            confidence = np.frombuffer(confidence_bytes, dtype=np.uint8).reshape((self.height, self.width))
            if self.frame_index == 1 or self.frame_index % 30 == 0:
                LOGGER.info(
                    "[TE DLSS5] frame=%d motion=CUDA NVOF ok phase=%s confidence=ok mean=%.3f min=%.3f max=%.3f",
                    self.frame_index, "bootstrap" if self.frame_index == 1 else "temporal",
                    float(confidence.mean()) / 255.0,
                    float(confidence.min()) / 255.0, float(confidence.max()) / 255.0,
                )
        else:
            motion = self._motion(rgb) if "motion" in self.mode else None
            if motion is not None and (self.frame_index == 1 or self.frame_index % 30 == 0):
                LOGGER.info("[TE DLSS5] frame=%d motion=OpenCV Farneback ok", self.frame_index)
        depth = self._depth(rgb, motion, confidence) if "depth" in self.mode else None
        if depth is not None and (self.frame_index == 1 or self.frame_index % 30 == 0):
            depth_phase = "inference" if self.frame_index == 1 or self.frame_index % self.depth_interval == 0 else "warped_reuse"
            LOGGER.info(
                "[TE DLSS5] frame=%d depth=Depth Anything V2 %s range=[%.3f, %.3f] mean=%.3f residual=%.4f history_weight=%.3f",
                self.frame_index, depth_phase, float(depth.min()), float(depth.max()), float(depth.mean()),
                self._depth_residual_mean, self._depth_history_weight,
            )
        return (
            motion.tobytes(order="C") if motion is not None else None,
            depth.tobytes(order="C") if depth is not None else None,
        )

    def consume_reset(self) -> bool:
        value = self._reset_required
        self._reset_required = False
        return value

    def close(self):
        if self.nvof is not None:
            self.nvof.close()
            self.nvof = None
        if self.depth_model is not None:
            self.depth_model.close()
            self.depth_model = None
        if self.frame_index:
            LOGGER.info("[TE DLSS5] Guidance closed after %d frame(s)", self.frame_index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
