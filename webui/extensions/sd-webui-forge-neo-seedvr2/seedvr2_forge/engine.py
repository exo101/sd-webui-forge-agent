from __future__ import annotations

import gc
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .bootstrap import SEEDVR2_VERSION_NUMBER, installed_version, source_is_compatible
from .config import SeedVR2Config, models_root, require_model, vendor_seedvr2_root


class SeedVR2ImportError(RuntimeError):
    pass


class SeedVR2Interrupted(RuntimeError):
    pass


# SeedVR2's global model cache is process-global. Protect it from concurrent Forge jobs.
_INFERENCE_LOCK = threading.RLock()


def _pil_batch_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """PIL RGB images -> SeedVR2 [T,H,W,C] float16 CPU tensor in [0,1]."""
    if not images:
        raise ValueError("No input images")
    size = images[0].size
    arrays = []
    for image in images:
        rgb = image.convert("RGB")
        if rgb.size != size:
            raise ValueError(f"All SeedVR2 batch images must have the same size: {size} vs {rgb.size}")
        arrays.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays, axis=0)).to(dtype=torch.float16)


def _tensor_to_pil_batch(tensor: torch.Tensor) -> List[Image.Image]:
    """SeedVR2 [T,H,W,C] float tensor -> PIL RGB list."""
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim != 4 or tensor.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected SeedVR2 output shape: {tuple(tensor.shape)}")
    arr = (tensor.numpy() * 255.0 + 0.5).astype(np.uint8)
    return [Image.fromarray(frame[..., :3], mode="RGB") for frame in arr]


class SeedVR2ForgeEngine:
    """Thin Forge adapter around SeedVR2's official v2.5.x core pipeline.

    We intentionally do not import inference_cli.py because that module changes
    multiprocessing and CUDA environment state at import time. Instead, this
    adapter calls the same core functions used by the CLI.
    """

    def __init__(self, config: SeedVR2Config):
        self.config = config
        self._backend: Optional[Dict[str, Any]] = None
        self._debug = None

    def _ensure_vendor(self) -> Path:
        root = vendor_seedvr2_root()
        if not source_is_compatible(root):
            found = installed_version(root)
            if found is None:
                detail = "source is missing or incomplete"
            else:
                detail = f"found SeedVR2 {found}, adapter expects {SEEDVR2_VERSION_NUMBER}"
            raise SeedVR2ImportError(
                f"Official SeedVR2 {SEEDVR2_VERSION_NUMBER} source is not ready: {detail}.\n"
                "Re-run extensions/sd-webui-seedvr2/install.py. "
                "If GitHub is blocked, manually copy the v2.5.23 repository into vendor/seedvr2."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    def _import_backend(self) -> Dict[str, Any]:
        if self._backend is not None:
            return self._backend

        root = self._ensure_vendor()
        try:
            from src.core.generation_utils import (  # type: ignore
                compute_generation_info,
                load_text_embeddings,
                log_generation_start,
                prepare_runner,
                setup_generation_context,
            )
            from src.core.generation_phases import (  # type: ignore
                decode_all_batches,
                encode_all_batches,
                postprocess_all_batches,
                upscale_all_batches,
            )
            from src.optimization.memory_manager import clear_memory  # type: ignore
            from src.utils.debug import Debug  # type: ignore
            from src.utils.constants import __version__ as seedvr2_version  # type: ignore
        except Exception as e:
            raise SeedVR2ImportError(
                f"Failed to import SeedVR2 core from {root}: {type(e).__name__}: {e}"
            ) from e

        self._backend = {
            "setup_generation_context": setup_generation_context,
            "prepare_runner": prepare_runner,
            "compute_generation_info": compute_generation_info,
            "log_generation_start": log_generation_start,
            "load_text_embeddings": load_text_embeddings,
            "encode_all_batches": encode_all_batches,
            "upscale_all_batches": upscale_all_batches,
            "decode_all_batches": decode_all_batches,
            "postprocess_all_batches": postprocess_all_batches,
            "clear_memory": clear_memory,
            "Debug": Debug,
            "version": seedvr2_version,
        }
        return self._backend

    @staticmethod
    def _normalize_attention(mode: str) -> str:
        aliases = {
            "flash": "flash_attn_2",
            "flash_attn": "flash_attn_2",
            "xformers": "sdpa",  # SeedVR2 v2.5.x no longer exposes xformers as a named mode.
        }
        return aliases.get(mode, mode)

    def _forge_interrupt(self) -> None:
        try:
            from modules import shared
            if getattr(shared.state, "interrupted", False) or getattr(shared.state, "skipped", False):
                raise SeedVR2Interrupted("SeedVR2 processing interrupted from Forge")
        except ImportError:
            return

    def _progress(self, current: int, total: int, frames: int, message: str) -> None:
        self._forge_interrupt()
        try:
            from modules import shared
            shared.state.job = f"SeedVR2 - {message}"
            if total > 0:
                shared.state.job_no = int(current)
                shared.state.job_count = int(total)
        except Exception:
            pass

    def _make_context(self, backend: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config

        def optional_device(value: Optional[str]):
            return None if value in (None, "", "none") else value

        ctx = backend["setup_generation_context"](
            dit_device=cfg.device,
            vae_device=cfg.device,
            dit_offload_device=optional_device(cfg.dit_offload_device),
            vae_offload_device=optional_device(cfg.vae_offload_device),
            tensor_offload_device=optional_device(cfg.tensor_offload_device),
            debug=self._debug,
        )
        # Add Forge interruption support; upstream normally installs a ComfyUI callback here.
        ctx["interrupt_fn"] = self._forge_interrupt
        return ctx

    def _prepare_runner(self, backend: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        cfg = self.config
        model_dir = models_root()
        model_dir.mkdir(parents=True, exist_ok=True)
        require_model(cfg.dit_model)
        require_model(cfg.vae_model)

        # Upstream cache and BlockSwap constraints are easy to violate from a UI,
        # so fail early with a useful Forge-side message.
        if cfg.keep_model_cached:
            if cfg.dit_offload_device in ("", "none", None):
                raise ValueError("Keeping the DiT cached requires a DiT offload device (normally cpu).")
            if cfg.vae_offload_device in ("", "none", None):
                raise ValueError("Keeping the VAE cached requires a VAE offload device (normally cpu).")
        if int(cfg.blocks_to_swap) > 0 and cfg.dit_offload_device in ("", "none", None):
            raise ValueError("BlockSwap requires DiT offload device=cpu (or another CUDA device).")

        runner, cache_context = backend["prepare_runner"](
            dit_model=cfg.dit_model,
            vae_model=cfg.vae_model,
            model_dir=str(model_dir),
            debug=self._debug,
            ctx=ctx,
            dit_cache=bool(cfg.keep_model_cached),
            vae_cache=bool(cfg.keep_model_cached),
            dit_id=910001 if cfg.keep_model_cached else None,
            vae_id=910002 if cfg.keep_model_cached else None,
            block_swap_config={
                "blocks_to_swap": int(cfg.blocks_to_swap),
                "swap_io_components": bool(getattr(cfg, "swap_io_components", False)),
                "offload_device": (
                    None if cfg.dit_offload_device in ("", "none", None)
                    else cfg.dit_offload_device
                ),
            },
            encode_tiled=bool(cfg.encode_tiled),
            encode_tile_size=(int(cfg.encode_tile_size), int(cfg.encode_tile_size)),
            encode_tile_overlap=(int(cfg.encode_tile_overlap), int(cfg.encode_tile_overlap)),
            decode_tiled=bool(cfg.decode_tiled),
            decode_tile_size=(int(cfg.decode_tile_size), int(cfg.decode_tile_size)),
            decode_tile_overlap=(int(cfg.decode_tile_overlap), int(cfg.decode_tile_overlap)),
            tile_debug="false",
            attention_mode=self._normalize_attention(cfg.attention_mode),
            torch_compile_args_dit=None,
            torch_compile_args_vae=None,
        )
        ctx["cache_context"] = cache_context
        return runner, cache_context

    def process_tiles(self, pil_tiles: List[Image.Image]) -> List[Image.Image]:
        """Run SeedVR2 on a batch of equal-sized tiles.

        For the supplied workflow batch_size=1. We still accept a list so the outer
        Forge tiler can reuse the same adapter for future batch processing.
        """
        if not pil_tiles:
            return []

        with _INFERENCE_LOCK:
            backend = self._import_backend()
            self._debug = backend["Debug"](enabled=bool(self.config.debug), show_timestamps=True)

            frames = _pil_batch_to_tensor(pil_tiles)
            cfg = self.config
            resolution = min(pil_tiles[0].size)

            ctx = self._make_context(backend)
            runner, _cache_context = self._prepare_runner(backend, ctx)

            # Same ordering as the official CLI, but without importing inference_cli.py.
            ctx["text_embeds"] = backend["load_text_embeddings"](
                str(vendor_seedvr2_root()),
                ctx["dit_device"],
                ctx["compute_dtype"],
                self._debug,
            )

            frames, gen_info = backend["compute_generation_info"](
                ctx=ctx,
                images=frames,
                resolution=int(resolution),
                max_resolution=int(cfg.max_resolution),
                batch_size=int(cfg.batch_size),
                uniform_batch_size=bool(cfg.uniform_batch_size),
                seed=int(cfg.seed),
                prepend_frames=0,
                temporal_overlap=0,
                debug=self._debug,
            )
            backend["log_generation_start"](gen_info, self._debug)

            try:
                ctx = backend["encode_all_batches"](
                    runner,
                    ctx=ctx,
                    images=frames,
                    debug=self._debug,
                    batch_size=int(cfg.batch_size),
                    uniform_batch_size=bool(cfg.uniform_batch_size),
                    seed=int(cfg.seed),
                    progress_callback=self._progress,
                    temporal_overlap=0,
                    resolution=int(resolution),
                    max_resolution=int(cfg.max_resolution),
                    input_noise_scale=float(cfg.input_noise_scale),
                    color_correction=str(cfg.color_correction),
                )
                ctx = backend["upscale_all_batches"](
                    runner,
                    ctx=ctx,
                    debug=self._debug,
                    progress_callback=self._progress,
                    seed=int(cfg.seed),
                    latent_noise_scale=float(cfg.latent_noise_scale),
                    cache_model=bool(cfg.keep_model_cached),
                )
                ctx = backend["decode_all_batches"](
                    runner,
                    ctx=ctx,
                    debug=self._debug,
                    progress_callback=self._progress,
                    cache_model=bool(cfg.keep_model_cached),
                )
                ctx = backend["postprocess_all_batches"](
                    ctx=ctx,
                    debug=self._debug,
                    progress_callback=self._progress,
                    color_correction=str(cfg.color_correction),
                    prepend_frames=0,
                    temporal_overlap=0,
                    batch_size=int(cfg.batch_size),
                )

                result = ctx.get("final_video")
                if not isinstance(result, torch.Tensor):
                    raise TypeError(f"SeedVR2 returned invalid final_video: {type(result)!r}")
                return _tensor_to_pil_batch(result)
            finally:
                # Upstream phases perform model-specific offload/cleanup. Force a deep
                # cache cleanup when caching is disabled so Forge can reclaim VRAM.
                if not cfg.keep_model_cached:
                    try:
                        backend["clear_memory"](debug=self._debug, deep=True, force=True)
                    except Exception:
                        pass
                del frames, ctx, runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

    @property
    def upstream_version(self) -> str:
        backend = self._import_backend()
        return str(backend.get("version", "unknown"))

    def cleanup(self) -> None:
        if self.config.keep_model_cached:
            return
        backend = self._backend
        if backend is not None:
            try:
                backend["clear_memory"](debug=self._debug, deep=True, force=True)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
