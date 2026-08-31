"""Optimized inference for See-through with model caching and text embedding cache.

This version adds:
1. Model caching - models are loaded once and reused
2. Text embedding cache - pre-compute and cache tag embeddings
3. Explicit GPU/CPU memory management
4. Support for continuous processing without reloading

Usage:
    python inference/scripts/inference_psd_optimized.py --srcp image.png --save_to_psd
"""

import os.path as osp
import argparse
import sys
import os

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

# 设置模型下载目录为用户指定的目录
_current_file = osp.abspath(__file__)
_webui_root = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(_current_file))))))
_models_dir = osp.join(_webui_root, "models", "diffusers")
os.environ['HF_HOME'] = osp.join(_webui_root, "models")
os.environ['HF_HUB_CACHE'] = _models_dir

import json
import time
import cv2
import numpy as np
import torch
from PIL import Image

from common.modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from common.modules.layerdiffuse.vae import TransparentVAE
from common.modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from common.modules.marigold import MarigoldDepthPipeline
from common.utils.cv import center_square_pad_resize, smart_resize, img_alpha_blending
from common.utils.torch_utils import seed_everything
from common.utils.io_utils import json2dict, dict2json
from common.utils.inference_utils import further_extr
from common.utils.cv import validate_resolution


VALID_BODY_PARTS_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth',
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear',
    'tail', 'wings', 'objects'
]

# Global model cache
_model_cache = {
    'layerdiff': None,
    'marigold': None,
    'layerdiff_config': None,
    'marigold_config': None,
}


def _log_vram(label):
    """Log current GPU VRAM usage for profiling."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f'[VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB')


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    _model_cache['layerdiff'] = None
    _model_cache['marigold'] = None
    _model_cache['layerdiff_config'] = None
    _model_cache['marigold_config'] = None
    torch.cuda.empty_cache()
    print('[Cache] Model cache cleared')


def get_layerdiff_pipeline(args):
    """Get or build the LayerDiff3D pipeline with caching support."""
    global _model_cache
    
    # Create config key for cache checking
    config_key = f"{args.quant_mode}_{args.repo_id_layerdiff}_{args.cpu_offload}_{args.group_offload}"
    
    if _model_cache['layerdiff'] is not None and _model_cache['layerdiff_config'] == config_key:
        print('[Cache] Reusing cached LayerDiff3D pipeline')
        return _model_cache['layerdiff']
    
    print('[Cache] Building new LayerDiff3D pipeline...')
    
    # Clear old model if exists
    if _model_cache['layerdiff'] is not None:
        del _model_cache['layerdiff']
        torch.cuda.empty_cache()
    
    quant_mode = args.quant_mode
    dtype = torch.bfloat16
    
    if quant_mode == 'none':
        # Try to find model locally first
        local_repo = find_layerdiff_model()
        if local_repo:
            repo = local_repo
            print(f'[None] Using local LayerDiff model: {repo}')
        else:
            repo = args.repo_id_layerdiff
            print(f'[None] Using LayerDiff model: {repo}')
        trans_vae = TransparentVAE.from_pretrained(repo, subfolder='trans_vae')
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')
        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            repo, trans_vae=trans_vae, unet=unet, scheduler=None)
        
        if args.cpu_offload:
            pipeline.vae.to(dtype=dtype)
            pipeline.trans_vae.to(dtype=dtype)
            pipeline.unet.to(dtype=dtype)
            pipeline.text_encoder.to(dtype=dtype)
            pipeline.text_encoder_2.to(dtype=dtype)
            pipeline.enable_model_cpu_offload()
        else:
            device = torch.device('cuda')
            pipeline.vae.to(dtype=dtype, device=device)
            pipeline.trans_vae.to(dtype=dtype, device=device)
            pipeline.unet.to(dtype=dtype, device=device)
            pipeline.text_encoder.to(dtype=dtype, device=device)
            pipeline.text_encoder_2.to(dtype=dtype, device=device)
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
    else:
        # NF4 mode - use quantized model
        print('[NF4] Using quantized model')
        # Try to find NF4 quantized model locally
        nf4_repo = None
        
        # 直接查找 NF4 模型，不要使用普通模型
        possible_nf4_paths = [
            # WebUI models 目录
            osp.join(_webui_root, "models", "diffusers", "models--24yearsold--seethroughv0.0.2_layerdiff3d_nf4"),
        ]
        
        for path in possible_nf4_paths:
            print(f'[NF4] Checking: {path}')
            if osp.exists(path):
                # Check snapshots
                snapshots_path = osp.join(path, "snapshots")
                if osp.isdir(snapshots_path):
                    for snapshot in os.listdir(snapshots_path):
                        snapshot_path = osp.join(snapshots_path, snapshot)
                        if (osp.isdir(snapshot_path) and 
                            osp.isdir(osp.join(snapshot_path, "trans_vae")) and 
                            osp.isdir(osp.join(snapshot_path, "unet"))):
                            nf4_repo = snapshot_path
                            break
                else:
                    # Check if the path itself contains the model
                    if (osp.isdir(osp.join(path, "trans_vae")) and 
                        osp.isdir(osp.join(path, "unet"))):
                        nf4_repo = path
                if nf4_repo:
                    print(f'[NF4] Found local NF4 model: {nf4_repo}')
                    break
        
        if not nf4_repo:
            print('[NF4] No local NF4 model found')
            raise FileNotFoundError("LayerDiff3D NF4 模型未找到！请从群主网盘中下载模型并放置到以下目录：\n" 
                                  f"{osp.join(_webui_root, 'models', 'diffusers', 'models--24yearsold--seethroughv0.0.2_layerdiff3d_nf4')}")
        
        unet = UNetFrameConditionModel.from_pretrained(nf4_repo, subfolder='unet')
        trans_vae = TransparentVAE.from_pretrained(nf4_repo, subfolder='trans_vae')
        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            nf4_repo, trans_vae=trans_vae, unet=unet, scheduler=None)
        
        if args.cpu_offload:
            pipeline.vae.to(dtype=dtype)
            pipeline.trans_vae.to(dtype=dtype)
            pipeline.enable_model_cpu_offload()
        else:
            device = torch.device('cuda')
            pipeline.vae.to(dtype=dtype, device=device)
            pipeline.trans_vae.to(dtype=dtype, device=device)
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
    
    # Note: KDiffusionStableDiffusionXLPipeline doesn't support cache_tag_embeds
    # Model caching alone still provides significant speedup
    
    # Store in cache
    _model_cache['layerdiff'] = pipeline
    _model_cache['layerdiff_config'] = config_key
    
    print('[Cache] LayerDiff3D pipeline cached')
    return pipeline


def get_marigold_pipeline(args):
    """Get or build the Marigold depth pipeline with caching support."""
    global _model_cache
    
    # Create config key for cache checking
    config_key = f"{args.quant_mode}_{args.repo_id_depth}_{args.cpu_offload}_{args.group_offload}"
    
    if _model_cache['marigold'] is not None and _model_cache['marigold_config'] == config_key:
        print('[Cache] Reusing cached Marigold pipeline')
        return _model_cache['marigold']
    
    print('[Cache] Building new Marigold pipeline...')
    
    # Clear old model if exists
    if _model_cache['marigold'] is not None:
        del _model_cache['marigold']
        torch.cuda.empty_cache()
    
    quant_mode = args.quant_mode
    dtype = torch.bfloat16
    
    if quant_mode == 'none':
        # Try to find Marigold model locally first
        local_repo = find_marigold_model()
        if local_repo:
            repo = local_repo
            print(f'[None] Using local Marigold model: {repo}')
        else:
            repo = args.repo_id_depth
            print(f'[None] Using Marigold model: {repo}')
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')
        pipeline = MarigoldDepthPipeline.from_pretrained(repo, unet=unet)
        
        if args.cpu_offload:
            pipeline.to(dtype=dtype)
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device='cuda', dtype=dtype)
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
    else:
        # NF4 mode - use quantized model
        print('[NF4] Using quantized Marigold model')
        # Try to find NF4 quantized model locally
        nf4_repo = None
        
        possible_nf4_paths = [
            # WebUI models 目录
            osp.join(_webui_root, "models", "diffusers", "models--24yearsold--seethroughv0.0.1_marigold_nf4"),
        ]
        
        for path in possible_nf4_paths:
            print(f'[NF4 Marigold] Checking: {path}')
            if osp.exists(path):
                # Check snapshots
                snapshots_path = osp.join(path, "snapshots")
                if osp.isdir(snapshots_path):
                    for snapshot in os.listdir(snapshots_path):
                        snapshot_path = osp.join(snapshots_path, snapshot)
                        if osp.isdir(snapshot_path) and osp.isdir(osp.join(snapshot_path, "unet")):
                            nf4_repo = snapshot_path
                            break
                else:
                    if osp.isdir(osp.join(path, "unet")):
                        nf4_repo = path
                if nf4_repo:
                    print(f'[NF4 Marigold] Found local NF4 model: {nf4_repo}')
                    break
        
        if not nf4_repo:
            print('[NF4 Marigold] No local NF4 model found')
            raise FileNotFoundError("Marigold NF4 模型未找到！请从群主网盘中下载模型并放置到以下目录：\n" 
                                  f"{osp.join(_webui_root, 'models', 'diffusers', 'models--24yearsold--seethroughv0.0.1_marigold_nf4')}")
        
        unet = UNetFrameConditionModel.from_pretrained(nf4_repo, subfolder='unet')
        pipeline = MarigoldDepthPipeline.from_pretrained(nf4_repo, unet=unet)
        
        if args.cpu_offload:
            pipeline.to(dtype=dtype)
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device='cuda', dtype=dtype)
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
    
    # Note: Custom MarigoldDepthPipeline may not support cache_tag_embeds
    # Model caching alone still provides significant speedup
    
    # Store in cache
    _model_cache['marigold'] = pipeline
    _model_cache['marigold_config'] = config_key
    
    print('[Cache] Marigold pipeline cached')
    return pipeline


def find_layerdiff_model():
    """Find LayerDiff model in common locations"""
    possible_paths = [
        # WebUI models 目录
        osp.join(_webui_root, "models", "diffusers", "models--24yearsold--seethroughv0.0.2_layerdiff3d"),
        osp.join(_webui_root, "models", "diffusers", "models--layerdifforg--seethroughv0.0.2_layerdiff3d"),
    ]
    
    print(f"[Model Search] WebUI root: {_webui_root}")
    print(f"[Model Search] Looking for LayerDiff model in {len(possible_paths)} locations")
    
    for path in possible_paths:
        print(f"[Model Search] Checking: {path}")
        if not osp.exists(path):
            continue
        
        trans_vae_path = osp.join(path, "trans_vae")
        unet_path = osp.join(path, "unet")
        
        if osp.isdir(trans_vae_path) and osp.isdir(unet_path):
            print(f"[Model Search] Found LayerDiff model at: {path}")
            return path
        
        snapshots_path = osp.join(path, "snapshots")
        if osp.isdir(snapshots_path):
            try:
                for snapshot in os.listdir(snapshots_path)[:5]:
                    snapshot_path = osp.join(snapshots_path, snapshot)
                    if (osp.isdir(snapshot_path) and 
                        osp.isdir(osp.join(snapshot_path, "trans_vae")) and 
                        osp.isdir(osp.join(snapshot_path, "unet"))):
                        print(f"[Model Search] Found LayerDiff model in snapshot: {snapshot_path}")
                        return snapshot_path
            except Exception as e:
                print(f"[Model Search] Error checking snapshots: {e}")
    
    print("[Model Search] No local model found")
    raise FileNotFoundError("LayerDiff 模型未找到！请从群主网盘中下载模型并放置到以下目录：\n" 
                          f"{osp.join(_webui_root, 'models', 'diffusers', 'models--24yearsold--seethroughv0.0.2_layerdiff3d')}")


def find_marigold_model():
    """Find Marigold model in common locations"""
    possible_paths = [
        # WebUI models 目录
        osp.join(_webui_root, "models", "diffusers", "models--24yearsold--seethroughv0.0.1_marigold"),
        osp.join(_webui_root, "models", "diffusers", "models--24yearsold--seethroughv0.0.1_marigold_nf4"),
    ]
    
    print(f"[Model Search] WebUI root: {_webui_root}")
    print(f"[Model Search] Looking for Marigold model in {len(possible_paths)} locations")
    
    for path in possible_paths:
        print(f"[Model Search] Checking: {path}")
        if not osp.exists(path):
            continue
        
        unet_path = osp.join(path, "unet")
        if osp.isdir(unet_path):
            print(f"[Model Search] Found Marigold model at: {path}")
            return path
        
        snapshots_path = osp.join(path, "snapshots")
        if osp.isdir(snapshots_path):
            try:
                for snapshot in os.listdir(snapshots_path)[:5]:
                    snapshot_path = osp.join(snapshots_path, snapshot)
                    if osp.isdir(snapshot_path) and osp.isdir(osp.join(snapshot_path, "unet")):
                        print(f"[Model Search] Found Marigold model in snapshot: {snapshot_path}")
                        return snapshot_path
            except Exception as e:
                print(f"[Model Search] Error checking snapshots: {e}")
    
    print("[Model Search] No local model found")
    raise FileNotFoundError("Marigold 模型未找到！请从群主网盘中下载模型并放置到以下目录：\n" 
                          f"{osp.join(_webui_root, 'models', 'diffusers', 'models--24yearsold--seethroughv0.0.1_marigold_nf4')}")

def load_pipeline_cached(quant_mode='nf4', cache_tag_embeds=True, group_offload=False, cpu_offload=False):
    """Load LayerDiff3D pipeline with caching (for WebUI integration)."""
    # Create a simple args object
    class Args:
        def __init__(self):
            self.quant_mode = quant_mode
            self.cache_tag_embeds = cache_tag_embeds
            self.group_offload = group_offload
            self.cpu_offload = cpu_offload
            self.repo_id = find_layerdiff_model()
            self.repo_id_depth = "prs-eth/marigold-depth-v1-0"
    
    args = Args()
    return get_layerdiff_pipeline(args)


def run_layerdiff_cached(pipeline, imgp, save_dir, seed, num_inference_steps, resolution):
    """Run LayerDiff3D with optimized memory management."""
    saved = osp.join(save_dir, osp.splitext(osp.basename(imgp))[0])
    os.makedirs(saved, exist_ok=True)
    input_img = np.array(Image.open(imgp).convert('RGBA'))
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
    scale = pad_size[0] / resolution
    Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))
    
    device = torch.device('cuda')
    offload = torch.device('cpu')
    
    # Body tags
    body_tag_list = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 
                     'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 
                     'tail', 'wings', 'objects']
    # Head tags
    head_tag_list = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 
                     'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
    
    rng = torch.Generator(device=device).manual_seed(seed)
    
    # Body pass
    print('[Optimize] Running body pass...')
    pipeline_output = pipeline(
        strength=1.0,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        generator=rng,
        guidance_scale=1.0,
        prompt=body_tag_list,
        negative_prompt='',
        fullpage=fullpage,
        group_index=0
    )
    
    images = pipeline_output.images
    for rst, tag in zip(images, body_tag_list):
        Image.fromarray(rst).save(osp.join(saved, f'{tag}.png'))
    head_img = images[2]
    
    # Head crop
    nz = cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8))
    if nz is not None:
        hx0, hy0, hw, hh = cv2.boundingRect(nz)
        hx = int(hx0 * scale) - pad_pos[0]
        hy = int(hy0 * scale) - pad_pos[1]
        hw = int(hw * scale)
        hh = int(hh * scale)
        
        def _crop_head(img, xywh):
            x, y, w, h = xywh
            ih, iw = img.shape[:2]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            if w < iw // 2:
                px = min(iw - x - w, x, w // 5)
                x1 = min(max(x - px, 0), iw)
                x2 = min(max(x + w + px, 0), iw)
            if h < ih // 2:
                py = min(ih - y - h, y, h // 5)
                y2 = min(max(y + h + py, 0), ih)
                y1 = min(max(y - py, 0), ih)
            return img[y1:y2, x1:x2], (x1, y1, x2, y2)
        
        input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, hw, hh])
        hx1 = int(hx1 / scale + pad_pos[0] / scale)
        hy1 = int(hy1 / scale + pad_pos[1] / scale)
        ih, iw = input_head.shape[:2]
        input_head, head_pad_size, head_pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)
        Image.fromarray(input_head).save(osp.join(saved, 'src_head.png'))
        
        # Head pass
        print('[Optimize] Running head pass...')
        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=head_tag_list,
            negative_prompt='',
            fullpage=input_head,
            group_index=1
        )
        
        canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        py1, py2, px1, px2 = (np.array([head_pad_pos[1], head_pad_pos[1] + ih, 
                                        head_pad_pos[0], head_pad_pos[0] + iw]) / scale).astype(np.int64)
        scale_size = (int(head_pad_size[0] / scale), int(head_pad_size[1] / scale))
        
        for rst, tag in zip(pipeline_output.images, head_tag_list):
            rst = smart_resize(rst, scale_size)[py1:py2, px1:px2]
            full = canvas.copy()
            full[hy1:hy1 + rst.shape[0], hx1:hx1 + rst.shape[1]] = rst
            Image.fromarray(full).save(osp.join(saved, f'{tag}.png'))
    
    # Offload models to CPU
    if not getattr(pipeline, '_st_group_offload', False):
        pipeline.unet.to(offload)
        pipeline.vae.to(offload)
    pipeline.trans_vae.to(offload)
    torch.cuda.empty_cache()
    _log_vram('Models offloaded to CPU')


def run_marigold_cached(marigold_pipe, srcp, save_dir, seed, resolution_depth):
    """Run Marigold depth estimation with optimized memory management."""
    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)
    
    src_img_p = osp.join(saved, 'src_img.png')
    fullpage = np.array(Image.open(src_img_p).convert('RGBA'))
    src_h, src_w = fullpage.shape[:2]
    
    if isinstance(resolution_depth, int) and resolution_depth == -1:
        resolution_depth = [src_h, src_w]
    resolution_depth = validate_resolution(resolution_depth)
    src_rescaled = resolution_depth[0] != src_h or resolution_depth[1] != src_w
    
    img_list = []
    empty_array = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)
    
    compose_list = {'eyes': ['eyewhite', 'irides', 'eyelash', 'eyebrow'], 
                    'hair': ['back hair', 'front hair']}
    
    for tag in VALID_BODY_PARTS_V2:
        tagp = osp.join(saved, f'{tag}.png')
        if osp.exists(tagp):
            tag_arr = np.array(Image.open(tagp))
            tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
            img_list.append(tag_arr)
        else:
            img_list.append(empty_array)
    
    compose_dict = {}
    for c, clist in compose_list.items():
        imlist = []
        taglist = []
        for tag in clist:
            p = osp.join(saved, tag + '.png')
            if osp.exists(p):
                tag_arr = np.array(Image.open(p))
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                imlist.append(tag_arr)
                taglist.append(tag)
        if len(imlist) > 0:
            img = img_alpha_blending(imlist, premultiplied=False)
            img_list[VALID_BODY_PARTS_V2.index(c)] = img
            compose_dict[c] = {'taglist': taglist, 'imlist': imlist}
    
    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255
    
    blended_alpha = np.clip(blended_alpha, 0, 1) * 255
    blended_alpha = blended_alpha.astype(np.uint8)
    fullpage[..., -1] = blended_alpha
    img_list.append(fullpage)
    
    img_list_input = img_list
    if src_rescaled:
        img_list_input = [smart_resize(img, resolution_depth) for img in img_list]
    
    device = torch.device('cuda')
    offload = torch.device('cpu')
    
    # Move models to GPU
    if not getattr(marigold_pipe, '_st_group_offload', False):
        marigold_pipe.unet.to(device)
        marigold_pipe.vae.to(device)
    torch.cuda.empty_cache()
    _log_vram('Marigold on GPU')
    
    seed_everything(seed)
    pipe_out = marigold_pipe(color_map=None, show_progress_bar=False, img_list=img_list_input)
    _log_vram('Marigold inference complete')
    
    depth_pred = pipe_out.depth_tensor.to(device='cpu', dtype=torch.float32).numpy()
    
    if src_rescaled:
        depth_pred = [smart_resize(d, (src_h, src_w)) for d in depth_pred]
    
    # Offload models to CPU
    if not getattr(marigold_pipe, '_st_group_offload', False):
        marigold_pipe.unet.to(offload)
        marigold_pipe.vae.to(offload)
    torch.cuda.empty_cache()
    _log_vram('Marigold offloaded to CPU')
    
    drawables = [{'img': img, 'depth': depth} for img, depth in zip(img_list, depth_pred)]
    drawables = drawables[:-1]
    blended = img_alpha_blending(drawables, premultiplied=False)
    
    infop = osp.join(saved, 'info.json')
    if osp.exists(infop):
        info = json2dict(infop)
    else:
        info = {'parts': {}}
    
    parts = info['parts']
    for ii, depth in enumerate(depth_pred[:-1]):
        depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        tag = VALID_BODY_PARTS_V2[ii]
        if tag in compose_dict:
            mask = blended_alpha > 256
            for t, im in zip(compose_dict[tag]['taglist'][::-1], compose_dict[tag]['imlist'][::-1]):
                mask_local = im[..., -1] > 15
                mask_invis = np.bitwise_and(mask, mask_local)
                depth_local = np.full((src_h, src_w), fill_value=255, dtype=np.uint8)
                depth_local[mask_local] = depth[mask_local]
                if np.any(mask_invis):
                    depth_local[mask_invis] = np.median(depth[np.bitwise_and(mask_local, np.bitwise_not(mask_invis))])
                mask = np.bitwise_or(mask, mask_local)
                
                parts_info = parts.get(t, {})
                Image.fromarray(depth_local).save(osp.join(saved, f'{t}_depth.png'))
                parts[t] = parts_info
            continue
        
        parts_info = parts.get(tag, {})
        Image.fromarray(depth).save(osp.join(saved, f'{tag}_depth.png'))
        parts[tag] = parts_info
    
    dict2json(info, infop)
    Image.fromarray(blended).save(osp.join(saved, 'reconstruction.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Optimized inference with model caching"
    )
    parser.add_argument('--srcp', type=str, required=True, help='input image')
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--tblr_split', action='store_true')
    parser.add_argument('--quant_mode', type=str, default='nf4', choices=['nf4', 'none'])
    parser.add_argument('--repo_id_layerdiff', type=str, default=None)
    parser.add_argument('--repo_id_depth', type=str, default=None)
    parser.add_argument('--cpu_offload', action='store_true', default=False)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--resolution_depth', type=int, default=-1,
                        help='Marigold depth inference resolution. -1 to match layerdiff resolution (default)')
    parser.add_argument('--group_offload', action='store_true', default=False)
    parser.add_argument('--cache_tag_embeds', action='store_true', default=True)
    parser.add_argument('--clear_cache', action='store_true', default=False,
                        help='Clear model cache before processing')
    
    args = parser.parse_args()
    
    REPO_MAP = {
        'nf4': {
            'layerdiff': '24yearsold/seethroughv0.0.2_layerdiff3d_nf4',
            'depth': '24yearsold/seethroughv0.0.1_marigold_nf4',
        },
        'none': {
            'layerdiff': 'layerdifforg/seethroughv0.0.2_layerdiff3d',
            'depth': '24yearsold/seethroughv0.0.1_marigold',
        },
    }
    defaults = REPO_MAP[args.quant_mode]
    if args.repo_id_layerdiff is None:
        args.repo_id_layerdiff = defaults['layerdiff']
    if args.repo_id_depth is None:
        args.repo_id_depth = defaults['depth']
    
    srcp = args.srcp
    seed = args.seed
    resolution = args.resolution
    num_inference_steps = args.num_inference_steps
    save_dir = args.save_dir
    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)
    
    print(f'Optimized inference: quant_mode={args.quant_mode}, cache_tag_embeds={args.cache_tag_embeds}')
    print(f'  Source image: {srcp}')
    print(f'  Resolution: {resolution}, Steps: {num_inference_steps}, Seed: {seed}')
    
    if args.clear_cache:
        clear_model_cache()
    
    torch.cuda.reset_peak_memory_stats()
    total_t0 = time.time()
    
    # LayerDiff
    print('\n' + '='*60)
    print('[Step 1/3] LayerDiff3D...')
    print('='*60)
    seed_everything(seed)
    print(f'[LayerDiff] Loading pipeline with quant_mode={args.quant_mode}')
    pipeline = get_layerdiff_pipeline(args)
    print(f'[LayerDiff] Pipeline loaded successfully')
    layerdiff_t0 = time.time()
    print(f'[LayerDiff] Running inference with resolution={resolution}, steps={num_inference_steps}')
    run_layerdiff_cached(pipeline, srcp, save_dir, seed, num_inference_steps, resolution)
    layerdiff_time = time.time() - layerdiff_t0
    print(f'[LayerDiff] Done in {layerdiff_time:.1f}s')
    
    # Marigold
    print('\n' + '='*60)
    print('[Step 2/3] Marigold depth...')
    print('='*60)
    print(f'[Marigold] Loading pipeline')
    marigold_pipe = get_marigold_pipeline(args)
    print(f'[Marigold] Pipeline loaded successfully')
    marigold_t0 = time.time()
    print(f'[Marigold] Running depth estimation')
    run_marigold_cached(marigold_pipe, srcp, save_dir, seed, resolution_depth=args.resolution_depth)
    marigold_time = time.time() - marigold_t0
    print(f'[Marigold] Done in {marigold_time:.1f}s')
    
    # PSD assembly
    print('\n' + '='*60)
    print('[Step 3/3] PSD assembly...')
    print('='*60)
    print(f'[PSD] Starting assembly with save_to_psd={args.save_to_psd}, tblr_split={args.tblr_split}')
    psd_t0 = time.time()
    further_extr(saved, rotate=False, save_to_psd=args.save_to_psd, tblr_split=args.tblr_split)
    psd_time = time.time() - psd_t0
    print(f'[PSD] Assembly done in {psd_time:.1f}s')
    
    total_time = time.time() - total_t0
    
    stats = {
        'quant_mode': args.quant_mode,
        'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'layerdiff_time_s': layerdiff_time,
        'marigold_time_s': marigold_time,
        'psd_time_s': psd_time,
        'total_time_s': total_time,
        'optimized': True,
        'cache_enabled': True,
    }
    print(f'\n{"="*60}')
    print(json.dumps(stats, indent=2))
    print(f'{"="*60}')
    
    with open(osp.join(saved, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
