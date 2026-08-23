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
from utils.cv import center_square_pad_resize, batch_load_masks
from live2d.scrap_model import VALID_BODY_PARTS_V2
from annotators.lama_inpainter import apply_inpaint



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--exec_list', type=str, default='workspace/datasets/eval_test.txt')
    

    args = parser.parse_args()

    config = args
    exec_list = load_exec_list(config.exec_list)

    save_dir = config.save_dir
    for p in tqdm(exec_list):
        saved = osp.join(save_dir, osp.splitext(osp.basename(p))[0])
        os.makedirs(saved, exist_ok=True)
        fullpage = center_square_pad_resize(np.array(Image.open(p).convert('RGB')), 1024)

        mask_list = batch_load_masks(p + '_masks.json')
        for tag in VALID_BODY_PARTS_V2:
            visible_mask = mask_list[VALID_BODY_PARTS_V2.index(tag)].astype(np.uint8) * 255
            visible_mask = center_square_pad_resize(visible_mask, 1024)
            inpainted = apply_inpaint(fullpage, visible_mask)
            inpainted = np.concatenate([inpainted, visible_mask[..., None]], axis=2)
            Image.fromarray(inpainted).save(saved + f'/{tag}.png')