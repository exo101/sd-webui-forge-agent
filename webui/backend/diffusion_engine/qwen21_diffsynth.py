"""
Qwen-Image-2.1 DiffSynth Backend

直接复用 DiffSynth-Studio 的 QwenImage21Pipeline 加载本地 int8 预量化权重
（DiT + TE + VAE），绕过 webui 的 k_diffusion 采样循环，用 DiffSynth 的
FlowMatchScheduler（Qwen-Image 模板）进行采样。

使用条件：
  - DiT: models/Stable-diffusion/qwen_image_2.1_int8_convrot_ds.safetensors
  - TE : models/text_encoder/qwen3vl_8b_int8_convrot_ds.safetensors
  - VAE: models/vae/qwen_image_2.1_vae_bf16_ds.safetensors
  - processor: models/Qwen/Qwen-Image-2.1/processor/
"""
import os
import time
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# int8 预量化的 exclude 列表（不量化的层，保持 bf16）
DIT_EXCLUDES = [
    "img_in",
    "modulation.1",
    "norm_out.linear",
    "proj_out",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "txt_in.in_layer",
    "txt_in.out_layer",
]


def _build_te_excludes():
    """TE 的 exclude：27 个 visual blocks 的 attn/mlp + merger 层。"""
    excludes = []
    for i in range(27):
        excludes.append(f"model.model.visual.blocks.{i}.attn.proj")
        excludes.append(f"model.model.visual.blocks.{i}.attn.qkv")
        excludes.append(f"model.model.visual.blocks.{i}.mlp.linear_fc1")
        excludes.append(f"model.model.visual.blocks.{i}.mlp.linear_fc2")
    for i in range(3):
        excludes.append(f"model.model.visual.deepstack_merger_list.{i}.linear_fc1")
        excludes.append(f"model.model.visual.deepstack_merger_list.{i}.linear_fc2")
    excludes.append("model.model.visual.merger.linear_fc1")
    excludes.append("model.model.visual.merger.linear_fc2")
    return excludes


TE_EXCLUDES = _build_te_excludes()


class Qwen21DiffSynthBackend:
    """封装 DiffSynth QwenImage21Pipeline 的构建与推理。"""

    def __init__(self, dit_path, te_path, vae_path, processor_path):
        self.dit_path = dit_path
        self.te_path = te_path
        self.vae_path = vae_path
        self.processor_path = processor_path
        self.pipeline = None
        self._patched = False

    def is_available(self):
        return all(os.path.exists(p) for p in [self.dit_path, self.te_path, self.vae_path, self.processor_path])

    def _patch_model_configs(self):
        """把本地权重的 hash 注册到 DiffSynth MODEL_CONFIGS。"""
        if self._patched:
            return
        from diffsynth.core.loader.file import hash_model_file
        import diffsynth.models.model_loader as ml
        from diffsynth.configs import model_configs as mc

        logger.info("[Qwen21-DS] hashing model files ...")
        h_dit = hash_model_file(self.dit_path)
        h_te = hash_model_file(self.te_path)
        h_vae = hash_model_file(self.vae_path)
        logger.info(f"[Qwen21-DS] dit={h_dit} te={h_te} vae={h_vae}")

        new_entries = [
            {"model_hash": h_dit, "model_name": "qwen_image_21_dit",
             "model_class": "diffsynth.models.qwen_image_21_dit.QwenImage21DiT"},
            {"model_hash": h_te, "model_name": "qwen_image_21_text_encoder",
             "model_class": "diffsynth.models.qwen_image_21_text_encoder.QwenImage21TextEncoder"},
            {"model_hash": h_vae, "model_name": "qwen_image_21_vae",
             "model_class": "diffsynth.models.qwen_image_21_vae.QwenImage21VAE"},
        ]
        existing = {c["model_hash"] for c in list(ml.MODEL_CONFIGS)}
        new_entries = [e for e in new_entries if e["model_hash"] not in existing]
        ml.MODEL_CONFIGS = list(ml.MODEL_CONFIGS) + new_entries
        mc.MODEL_CONFIGS = list(mc.MODEL_CONFIGS) + new_entries
        logger.info(f"[Qwen21-DS] MODEL_CONFIGS patched (+{len(new_entries)} entries)")
        self._patched = True

    def build_pipeline(self):
        """构建 DiffSynth pipeline（懒加载，首次调用时执行）。"""
        if self.pipeline is not None:
            return self.pipeline

        t0 = time.time()
        self._patch_model_configs()

        from diffsynth.core import ModelConfig
        from diffsynth.core.quant import QuantizeConfig
        from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

        # TE：disk offload。每次生成只在 prompt 编码时用到一次，按层从磁盘懒加载到 GPU，
        # 用完自动退回 meta —— 加载阶段零物化，常驻内存省 9.3GB（峰值从 ~40GB 降 ~9GB）
        # DiT：每个采样步都要用。显存充足（>=12GB，如 16GB 卡）时整模型（6.8GB int8）
        # 常驻 GPU —— CPU 逐层 offload 会让每步都卡在 PCIe 带宽上，是慢的主要原因；
        # 显存不足的小卡回退 CPU 常驻按层搬运
        # VAE：常驻 GPU
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        dit_on_gpu = gpu_mem_gb >= 12
        logger.info(f"[Qwen21-DS] 显存 {gpu_mem_gb:.1f}GB，DiT {'常驻 GPU' if dit_on_gpu else 'CPU offload'}")
        # 彻底抑制 DiffSynth 内部冗余日志（下载器配置框、comfy_kitchen 提示、逐层加载 JSON）
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
        import builtins
        _orig_print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            q_dit = QuantizeConfig(
                method="comfy_kitchen_int8_w8a8",
                load_prequantized=True,
                exclude_modules=DIT_EXCLUDES,
            )
            q_te = QuantizeConfig(
                method="comfy_kitchen_int8_w8a8",
                load_prequantized=True,
                exclude_modules=TE_EXCLUDES,
            )
            pipe = QwenImage21Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    # disk 模式开关是 offload_dtype="disk" 字符串（layers.py L123/L337），
                    # 传 dtype 会让 AutoWrappedModule/AutoWrappedLinear 的 disk_offload=False，
                    # onload 时 meta→cuda 直接崩；onload/preparing 仍须显式指定（否则 load_device 解析为 None）
                    ModelConfig(path=self.te_path, offload_device="disk", offload_dtype="disk",
                                onload_device="cuda", preparing_device="cuda", quantize=q_te),
                    # DiT 常驻 GPU 时不带 offload 参数（与 VAE 一致，落在 from_pretrained 的 device 上）
                    ModelConfig(path=self.dit_path, quantize=q_dit) if dit_on_gpu else
                    ModelConfig(path=self.dit_path, offload_device="cpu", offload_dtype=torch.bfloat16, quantize=q_dit),
                    ModelConfig(path=self.vae_path),
                ],
                processor_config=ModelConfig(path=self.processor_path),
            )
        finally:
            builtins.print = _orig_print
        # 兜底校验：diffsynth fetch_model 取不到时只打印 "No ... available. This is not an
        # error." 并返回 None（上面的静默会把它吞掉），若不在这里拦截，
        # 静默的 None 会一路流到采样 unit 才炸出难以定位的 AttributeError
        if pipe.dit is None or pipe.text_encoder is None or pipe.vae is None:
            missing = [n for n, m in (("dit", pipe.dit), ("text_encoder", pipe.text_encoder), ("vae", pipe.vae)) if m is None]
            raise RuntimeError(
                f"[Qwen21-DS] 模型池缺少组件: {missing}，DiT 权重可能不是 Qwen21 格式: {self.dit_path}"
            )
        # 回收构建期瞬态（DiT 物化 dict、mmap 缓冲等），压低峰值残留
        import gc
        gc.collect()
        logger.info(f"[Qwen21-DS] pipeline ready in {time.time() - t0:.0f}s")
        self.pipeline = pipe
        return pipe

    def generate(self, prompt, negative_prompt=" ", height=1024, width=1024,
                 seed=42, steps=40, cfg_scale=1.0, progress_callback=None,
                 reference_images=None):
        """调用 DiffSynth pipeline 生成图像，返回 PIL.Image。

        reference_images: PIL.Image 列表（多图参考插件的参考图 / i2i 首图），
        通过 pipeline 的 edit_image 传入 —— pipeline 内部会做 VL 多图
        prompt 嵌入（<imageN> token）和 VAE latent 前缀，参考图真正进入潜空间参与生成。
        """
        pipe = self.build_pipeline()

        kwargs = {}
        if reference_images:
            kwargs["edit_image"] = reference_images

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=steps,
            use_kv_cache=True,
            **kwargs,
        )
        return image


def find_diffsynth_weights(main_model_path):
    """根据主模型文件路径，定位对应的 DiffSynth 布局权重文件。

    主模型文件: .../Stable-diffusion/qwen_image_2.1_int8_convrot.safetensors
    返回 dit/te/vae/processor 路径字典，全部存在才返回，否则 None。
    """
    base = os.path.basename(main_model_path)
    # 只有 Qwen21 主模型才有 DiffSynth 布局。不校验的话，任意模型（如 SDXL）
    # 只要 4 个 companion 文件碰巧存在就会被误判，主模型被强制按 Qwen21 DiT
    # 加载，pipe.dit 静默为 None，生成时才报 'NoneType' 错误。
    if "qwen_image_2.1" not in base.lower():
        return None
    models_dir = os.path.dirname(os.path.dirname(os.path.abspath(main_model_path)))
    # 主模型同目录的 _ds 版本
    ds_name = base.replace("_convrot.", "_convrot_ds.")
    # 替换后文件名必须变化，否则 dit_path 指向主模型自身，布局无效
    if ds_name == base:
        return None
    dit_path = os.path.join(os.path.dirname(main_model_path), ds_name)

    te_path = os.path.join(models_dir, "text_encoder", "qwen3vl_8b_int8_convrot_ds.safetensors")
    vae_path = os.path.join(models_dir, "vae", "qwen_image_2.1_vae_bf16_ds.safetensors")
    processor_path = os.path.join(models_dir, "Qwen", "Qwen-Image-2.1", "processor")

    paths = {"dit_path": dit_path, "te_path": te_path, "vae_path": vae_path, "processor_path": processor_path}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    return None


def pil_to_decoded_tensor(image):
    """PIL.Image -> (C, H, W) float tensor in [-1, 1]。"""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.moveaxis(arr, -1, 0)  # HWC -> CHW
    t = torch.from_numpy(arr) * 2.0 - 1.0
    return t


def pixel_tensor_to_pil(t):
    """(1, H, W, C) [0,1] HWC 像素张量 -> PIL.Image (RGB)。

    参考图经插件 encode_first_stage 时保留的原始像素；VAE 侧的 4 通道
    RGBA 编码在 pipeline 内部自行完成，这里只需 RGB 像素。
    """
    from PIL import Image
    arr = t[0].clamp(0.0, 1.0).float().cpu().numpy()  # (H, W, C)
    if arr.shape[-1] >= 3:
        arr = arr[:, :, :3]
    return Image.fromarray((arr * 255).round().astype("uint8"), "RGB")


# ---------------------------------------------------------------------------
# Dummy 对象：DiffSynth 模式下 webui 核心代码仍会访问 vae.latent_channels /
# unet.model.predictor 等属性，用最小 dummy 满足接口
# ---------------------------------------------------------------------------

class _DummyPredictor:
    sigmas = None
    def set_sigmas(self, sigmas):
        self.sigmas = sigmas


class _DummyModel:
    predictor = _DummyPredictor()
    diffusion_model = None  # live preview 路径会访问，DiffSynth 模式下不使用


class _DummyUnet:
    model = _DummyModel()
    model_options = {"transformer_options": {}}


class _DummyVae:
    latent_channels = 64  # Qwen-Image-2.1 latent channels
    upscale_ratio = 16    # VAE scale factor


# ---------------------------------------------------------------------------
# 自定义 Sampler：完全绕过 webui 的 k_diffusion 循环，调用 DiffSynth pipeline
# ---------------------------------------------------------------------------

class DiffSynthSampler:
    """兼容 webui Sampler 接口的 DiffSynth 采样器。

    sample() 返回 DecodedSamples（[-1,1] 的 CHW 张量列表），
    webui 的 process_images_inner 会跳过 VAE 解码，直接走后续流程。
    """

    def __init__(self, funcname=None):
        self.funcname = funcname or "diffsynth"
        self.config = None
        self.stop_at = None
        self.last_latent = None

    def sample(self, p, x, conditioning, unconditional_conditioning, image_conditioning=None):
        from modules import shared
        from modules.processing import DecodedSamples
        from modules.sd_samplers_common import InterruptedException

        backend = p.sd_model.diffsynth_backend
        if backend is None:
            raise RuntimeError("Qwen21 DiffSynth backend not initialized")

        # 多图参考 / i2i 首图：插件已把参考图经 encode_first_stage(is_referencing=True)
        # 编码并存入 sd_model.ref_images（[0,1] HWC 像素）；i2i 首图存 sd_model.ini_image。
        # 这里取像素转 PIL 传给 pipeline 的 edit_image（VL 嵌入 + latent 前缀），
        # 与原生模式 "ini_latent 作为第一张参考" 的语义一致。
        sd_model = p.sd_model
        ref_pils = []
        if getattr(sd_model, "ini_image", None) is not None:
            ref_pils.append(pixel_tensor_to_pil(sd_model.ini_image))
            sd_model.ini_image = None
            sd_model.ini_latent = None
        for pix in list(getattr(sd_model, "ref_images", []) or []):
            ref_pils.append(pixel_tensor_to_pil(pix))

        # webui 采样参数
        prompt = p.prompt
        negative_prompt = p.negative_prompt or " "
        width = p.width
        height = p.height
        steps = p.steps or 40
        cfg_scale = p.cfg_scale or 1.0
        batch_size = p.batch_size

        # 进度条初始化（对齐 webui 的 launch_sampling）
        shared.state.sampling_steps = steps
        shared.state.sampling_step = 0
        shared.state.preview_step = 0

        results = DecodedSamples()
        if ref_pils:
            print(f"[Qwen21-DS] {len(ref_pils)} reference image(s) passed to pipeline as edit_image")
        for i in range(batch_size):
            seed = int(p.seeds[i]) if p.seeds and i < len(p.seeds) else p.seed

            # 中断检测
            if shared.state.interrupted or shared.state.skipped:
                raise InterruptedException

            image = backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                seed=seed,
                steps=steps,
                cfg_scale=cfg_scale,
                reference_images=ref_pils or None,
            )

            # 转成 webui 期望的 [-1, 1] CHW 张量
            t = pil_to_decoded_tensor(image)
            results.append(t)

            # 更新进度
            shared.state.sampling_step = steps
            shared.state.preview_step = steps
            shared.total_tqdm.update()

        return results
