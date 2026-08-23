import os.path as osp
import sys

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as TT
import math

from .pos_estimator import ModelFeatConcat, ModelFeatMatch
from .bg_segmenter import CharacterBGSegmenter
from utils.torch_utils import init_model_from_pretrained


def resize_min_dry(x, s=512):
    # returns size
    h,w = x[-2:]
    return (
        (s, int(w*s/h)),
        (int(h*s/w), s),
    )[w<h]


def pixel_rounder(n, mode):
    if mode==True or mode=='round':
        return round(n)
    elif mode=='ceil':
        return math.ceil(n)
    elif mode=='floor':
        return math.floor(n)
    else:
        return n


def pixel_ij(x, rounding=True):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return tuple(pixel_rounder(i, rounding) for i in (
        x if isinstance(x, tuple) or isinstance(x, list) else (x,x)
    ))


def resize_min(image: Image.Image, s=512, resample=Image.Resampling.LANCZOS, antialias=False):
    dry = resize_min_dry(image.size[::-1], s=s)
    return image.resize(dry[::-1], resample=resample)


def abbox(img: np.ndarray, thresh=0.5, allow_empty=False):
    # get bbox from alpha image, at threshold
    if isinstance(img, Image.Image):
        img = np.ndarray(img)

    if img.ndim == 2:
        img = img[None]

    assert len(img) in [1,4], 'image must be mode L or RGBA'
    a = img[-1] > thresh
    xlim = np.any(a, axis=1).nonzero()[0]
    ylim = np.any(a, axis=0).nonzero()[0]
    if len(xlim)==0 and allow_empty: xlim = np.asarray([0, a.shape[0]])
    if len(ylim)==0 and allow_empty: ylim = np.asarray([0, a.shape[1]])
    axmin,axmax = max(int(xlim.min()-1),0), min(int(xlim.max()+1),a.shape[0])
    aymin,aymax = max(int(ylim.min()-1),0), min(int(ylim.max()+1),a.shape[1])
    return [(axmin,aymin), (axmax-axmin,aymax-aymin)]


segmenter = None

def infer_segmentation(image: Image.Image, bbox_thresh=0.5, device='cuda'):
    assert isinstance(image, Image.Image)
    global segmenter
    if segmenter is None:
        # segmenter = CharacterBGSegmenter().to(device=device).eval()
        segmenter = init_model_from_pretrained(
                        'dreMaz/bizarre-pose-estimator',
                        CharacterBGSegmenter,
                        weights_name='character_bg_seg-epoch96.safetensors',
                        device=device).eval()

    ori_size = image.size
    image = resize_min(image, 256).convert('RGB')
    timg = TT.functional.to_tensor(image)[None].to(device)
    with torch.no_grad():
        out = segmenter(timg)
    segmentation = TT.functional.to_pil_image(out['softmax'][0,1].float().cpu()).resize(ori_size)

    bbox = abbox(np.array(segmentation, dtype=np.float32) / 255., thresh=bbox_thresh, allow_empty=True)
    return segmentation, bbox


def cropbox_inverse(origin_size, from_corner, from_size, to_size):
    # origin_size: original image size
    # from_corner/from_size/to_size: of cropbox to invert
    origin_size = pixel_ij(origin_size, rounding=False)
    from_corner = pixel_ij(from_corner, rounding=False)
    from_size = pixel_ij(from_size, rounding=False)
    to_size = pixel_ij(to_size, rounding=False)
    sx,sy = to_size[0]/from_size[0], to_size[1]/from_size[1]
    return [
        (-from_corner[0]*sx, -from_corner[1]*sy),
        (origin_size[0]*sx, origin_size[1]*sy),
        origin_size,
    ]


def cropbox(image: Image.Image, from_corner, from_size, to_size=None, resample='bilinear'):
    from_corner = pixel_ij(from_corner, rounding=True)
    from_size = pixel_ij(from_size, rounding=True)
    to_size = pixel_ij(to_size, rounding=True) if to_size!=None else from_size
    return TT.functional.resized_crop(
        image,
        from_corner[0],
        from_corner[1],
        from_size[0],
        from_size[1],
        to_size,
        interpolation=getattr(TT.functional.InterpolationMode, resample.upper()),
        antialias=True
    )


def cropbox_compose(cba, cbb):
    # compose two cropboxes
    fca,fsa,tsa = [pixel_ij(q, rounding=False) for q in cba]
    fcb,fsb,tsb = [pixel_ij(q, rounding=False) for q in cbb]
    sfx = fsa[0] / tsa[0]
    sfy = fsa[1] / tsa[1]
    fc = fca[0]+fcb[0]*sfx, fca[1]+fcb[1]*sfy
    fs = fsb[0]*sfx, fsb[1]*sfy
    ts = tsb
    return fc, fs, ts

def cropbox_sequence(cropboxes):
    # compose multiple cropboxes in sequence
    ans = cropboxes[-1]
    for c in range(len(cropboxes)-2, -1, -1):
        cb = cropboxes[c]
        ans = cropbox_compose(cb, ans)
    return ans


def resize_square_dry(x, s=512):
    # returns a forward cropbox
    h,w = x[-2:]
    from_corner = (
        (0, -(h-w)//2),
        (-(w-h)//2, 0),
    )[h<w]
    from_size = (max(h,w),)*2
    to_size = (s, s)
    return (from_corner, from_size, to_size)


def cropbox_points(pts, from_corner, from_size, to_size):
    # apply cropbox to points
    pts = np.asarray(pts)
    assert len(pts.shape)==2 and pts.shape[1]==2
    fc = pixel_ij(from_corner, rounding=False)
    fs = pixel_ij(from_size, rounding=False)
    ts = pixel_ij(to_size, rounding=False)
    fc = np.asarray(fc)[None,]
    sf = np.asarray([ts[0]/fs[0], ts[1]/fs[1]])[None,]
    return (pts-fc)*sf


pos_estimators = {}


from utils.visualize import coco_keypoints_ext


@torch.inference_mode()
def apply_pos_estimator(image: Image.Image, mask: np.ndarray=None, model_type='feat_concat',
                        flip_aug=False,
                        smoothing=0.1, pad_factor=1, size=256, padding=0.1, device='cuda', heat_map_aug=False, return_kps_as_dict=False):
    '''
    mask (np.ndarray shape: H, W): optional, range in 0~1
    '''
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)

    if mask is None:
        mask, bbox = infer_segmentation(image)
    else:
        assert isinstance(mask, np.ndarray)
        if mask.ndim == 3:
            assert mask.shape[-1] == 1
            mask = mask[..., 0]
        assert mask.ndim == 2
        bbox = abbox(mask, thresh=0.2)
        image = Image.composite(
            image, Image.new('RGB', image.size, color=0),
            Image.fromarray((mask * 255).astype(np.uint8))
        )

    global pos_estimators
    # if pos_estimator is None:
    if model_type not in pos_estimators:
        if model_type == 'feat_concat':
            pos_estimator = init_model_from_pretrained(
                'dreMaz/bizarre-pose-estimator',
                ModelFeatConcat,
                weights_name='feat_concat_plusdata.safetensors', device=device).eval()
        elif model_type == 'feat_match':
            pos_estimator = init_model_from_pretrained(
                'dreMaz/bizarre-pose-estimator',
                ModelFeatMatch,
                weights_name='feat_match_plusdata.safetensors', device=device).eval()
        else:
            raise Exception(f'invalid model type: {model_type}')
        pos_estimators[model_type] = pos_estimator
    else:
        pos_estimator = pos_estimators[model_type]

    ori_size = image.size
    to_pad = size * padding
    cb = cropbox_sequence([
        # crop to bbox, resize to square, pad sides
        [bbox[0], bbox[1], bbox[1]],
        resize_square_dry(bbox[1], size),
        [-to_pad*pad_factor/2, size +to_pad*pad_factor, size],
    ])
    icb = cropbox_inverse(ori_size[::-1], *cb)
    image = cropbox(image, *cb).convert('RGB')

    timg = TT.functional.to_tensor(image)[None].to(device=device)
    out = pos_estimator(timg, smoothing=smoothing, return_more=True, heat_map_aug=heat_map_aug)

    scores = out['keypoint_scores']
    # post-process keypoints
    kps = out['keypoints'][0].cpu().numpy()
    kps = cropbox_points(kps, *icb)

    if flip_aug:
        import torchvision
        out = pos_estimator(torchvision.transforms.functional.hflip(timg), smoothing=smoothing, return_more=True)
        kps_flip = out['keypoints'][0].cpu().numpy()
        kps_flip_ = kps_flip.copy()
        lst = np.arange(1, 17).reshape((-1, 2))
        for l in lst:
            kps_flip_[l] = kps_flip[l[::-1]]
        kps_flip = kps_flip_
        for k in kps_flip:
            k[1] = timg.shape[-1] - k[1]
        kps_flip = cropbox_points(kps_flip, *icb)
        kps = (kps + kps_flip) / 2

        # post-process keypoints
        # kps = out['keypoints'][0].cpu().numpy()

    return kps, scores, bbox
