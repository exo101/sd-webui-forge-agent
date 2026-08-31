import argparse
from omegaconf import OmegaConf

from PIL import Image

import os
import os.path as osp
import numpy as np
from  tqdm import tqdm
from einops import reduce
import click
import cv2
import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from utils.io_utils import load_exec_list, find_all_imgs



def sam_parse_body_samples(config):

    from live2d.scrap_model import animal_ear_detected, Drawable, VALID_BODY_PARTS_V2
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything, init_model_from_pretrained
    from utils.visualize import visualize_segs_with_labels
    from modules.semanticsam import SemanticSam, Sam
    import torch


    seed_everything(42)

    config = OmegaConf.load(config)
    
    exec_list = config.exec_list
    ckpt = config.ckpt
    rank_to_worldsize = config.rank_to_worldsize
    save_dir = config.save_dir
    save_to_local = config.get('save_to_local', False)

    if not save_to_local:
        os.makedirs(save_dir, exist_ok=True)

    if osp.isdir(exec_list):
        exec_list = find_all_imgs(exec_list, abs_path=True)

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    model: SemanticSam = init_model_from_pretrained(
        pretrained_model_name_or_path=ckpt,
        module_cls=SemanticSam,
        download_from_hf=False,
        model_args=dict(class_num=19)
    ).to(device='cuda')

    model_name = osp.splitext(osp.basename(ckpt))[0]

    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:

            # instance_mask, crop_xyxy, score = load_detected_character(p)
            # if instance_mask is None:
            #     print(f'skip {p}, no character instance detected')
            #     continue
            
            # lmodel = Live2DScrapModel(p, crop_xyxy=crop_xyxy, pad_to_square=False)
            # lmodel.init_drawable_visible_map()
            # final_img = compose_from_drawables(lmodel.drawables)

            img = np.array(Image.open(p).convert('RGB'))
            with torch.inference_mode():
                preds = model.inference(img)[0]
                masks_np = (preds > 0).to(device='cpu', dtype=torch.bool).numpy()

            # save_tmp_img(visualize_segs_with_labels(masks_np, final_img[..., :3], VALID_BODY_PARTS_V1, reference_img=final_img[..., :3]))
            # print(f'save to ' + osp.join(model_dir, f'{model_name}_masks.json'))
            if save_to_local:
                saved = osp.dirname(p)
            else:
                saved = save_dir
            batch_save_masks(masks_np, osp.join(saved, f'{osp.basename(p)}_masks.json'))


        except Exception as e:
            raise
            print(f'Failed to process {p}: {e}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 默认 config 路径相对于脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'local_configs', 'evalsam_iter1.yaml')
    parser.add_argument('--config', type=str, default=default_config)
    # 兼容 see_through.py 传入的参数（这些参数实际用于 inference_psd_optimized.py）
    parser.add_argument('--srcp', type=str, default=None, help='(兼容) 输入图像路径，由 see_through.py 传入')
    parser.add_argument('--save_dir', type=str, default=None, help='(兼容) 输出目录，由 see_through.py 传入')
    parser.add_argument('--resolution', type=int, default=None, help='(兼容) 处理分辨率，由 see_through.py 传入')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='(兼容) 推理步数，由 see_through.py 传入')
    parser.add_argument('--seed', type=int, default=None, help='(兼容) 随机种子，由 see_through.py 传入')
    parser.add_argument('--quant_mode', type=str, default=None, help='(兼容) 量化模式，由 see_through.py 传入')
    parser.add_argument('--save_to_psd', action='store_true', default=None, help='(兼容) 保存为PSD，由 see_through.py 传入')
    parser.add_argument('--tblr_split', action='store_true', default=None, help='(兼容) 左右分离，由 see_through.py 传入')
    parser.add_argument('--cache_tag_embeds', action='store_true', default=None, help='(兼容) 缓存文本嵌入，由 see_through.py 传入')
    # 额外处理参数拼接成单个 token 的情况（如 --srcpD:\path 而不是 --srcp D:\path）
    # 通过 parse_known_args 忽略未知参数
    args, unknown = parser.parse_known_args()

    # 处理 unknown 参数中可能包含的 --key=value 或 --keyvalue 格式
    # 例如 --srcpD:\path 会被解析为一个未知 token
    # 尝试从中提取 srcp/save_dir/resolution 等值
    for token in unknown:
        if token.startswith('--'):
            # 去掉 -- 前缀
            stripped = token[2:]
            for known_arg in ['srcp', 'save_dir', 'resolution', 'num_inference_steps', 'seed', 'quant_mode']:
                if stripped.startswith(known_arg):
                    value = stripped[len(known_arg):]
                    if value.startswith('='):
                        value = value[1:]
                    if value:
                        setattr(args, known_arg, value)
                    break

    # 检查 config 文件是否存在
    config_path = args.config
    if not os.path.exists(config_path):
        # 如果提供了 --srcp，说明是从 see_through 管道调用的，但本脚本不支持直接按场景分割模式运行
        # 尝试在脚本同级的 local_configs 目录下查找
        alt_config = os.path.join(script_dir, 'local_configs', 'evalsam_iter1.yaml')
        if os.path.exists(alt_config):
            config_path = alt_config
        else:
            print(f"错误: 找不到配置文件 {config_path}")
            print(f"提示: {script_dir}/local_configs/ 目录不存在或缺少 evalsam_iter1.yaml")
            print("本脚本 (infer_sam.py) 是使用 SemanticSam 的人体部位解析脚本，")
            print("需要配置文件和模型检查点才能运行。")
            print("如需场景分割，请使用 see_through.py 中的 '场景分割 (SAM)' 模式（使用 scene_segmenter.py）。")
            sys.exit(1)

    sam_parse_body_samples(config_path)