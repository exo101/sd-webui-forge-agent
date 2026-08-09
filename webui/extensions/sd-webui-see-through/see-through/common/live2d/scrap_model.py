from importlib import metadata
from typing import List, Union
import shutil
import os.path as osp
import os
from pathlib import Path
import yaml
from functools import cached_property

from PIL import Image, ImageOps
import cv2
import numpy as np
from einops import reduce

from utils.io_utils import find_all_files_recursive, get_last_modified_file, json2dict, dict2json, pil_pad_square, imwrite
from utils.visualize import visualize_pos_keypoints, coco_parts, coco_parts_dict
from utils.cv import img_alpha_blending, bbox_intersection, rle2mask



MINIMUM_VISIBLE_ALPHA = 25


VALID_BODY_PARTS_V1 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'beard', 'mouth', 'neck', 'neckwear',
    'skin', 'topwear', 'handwear',
    'bottomwear', 'legwear', 'footwear', 
    'tail', 'wings'
]

VALID_BODY_PARTS_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 
    'tail', 'wings', 'objects'
]


VALID_BODY_PARTS_V3 = [
    'front hair', 'back hair', 'headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 
    'tail', 'wings', 'objects'
]

_tag_cvt = {
    'hairf': 'front hair',
    'hairb': 'back hair',
    'eyebg': 'eyewhite'
}


def pos_transform(pos, matrix, H, W):
    pos = pos * matrix[[0, 1], [0, 1]][None] / 2 + 0.5
    x = np.round(pos[..., 0] * W).astype(np.int32)
    y = np.round((1 - pos[..., 1]) * H).astype(np.int32)
    return np.stack([x, y], axis=1)


def load_drawable_vertex_info(drawablep: str, H, W):
    def _str2array(txt, dim=2):
        arr = np.fromstring(txt.replace('),(', ',').rstrip(')').lstrip('('),sep=',')
        if dim != 1:
            arr = arr.reshape((-1, dim))
        return arr
    
    srcp = osp.splitext(drawablep)[0] + '.json'
    if osp.exists(srcp):
        infos = json2dict(srcp)
        for k, v in infos.items():
            if isinstance(v, list):
                infos[k] = np.array(v)
        return infos

    srcp = osp.splitext(drawablep)[0] + '.txt'
    if not osp.exists(srcp):
        return None
    with open(srcp, 'r', encoding='utf8') as f:
        txtlines = f.read().split('\n')
    infos = {}
    for l in txtlines:
        if l.startswith('Positions='):
            infos['position'] = _str2array(l.replace('Positions=', ''))
        # elif l.startswith('UVs='):
        #     infos['UV'] = _str2array(l.replace('UVs=', ''))
        elif l.startswith('Matrix='):
            infos['matrix'] = _str2array(l.replace('Matrix=', ''), 4)
        elif l.startswith('Indices='):
            infos['ids'] = _str2array(l.replace('Indices=', ''), 1).astype(np.int32)
    infos['position'] = pos_transform(infos['position'], infos['matrix'], H, W)
    return infos


def vertex_info_from_metadata(meta_data, src_size, transform_matrix, frame_crop=None):
    if meta_data is None:
        return
    W, H = src_size
    infos = {}
    infos['ids'] = np.array(meta_data['vertex_indices']).astype(np.int64)
    infos['position'] = np.array(meta_data['vertex_pos']).reshape((-1, 2)).astype(np.float32)
    infos['position'] = pos_transform(np.array(infos['position']), transform_matrix, H, W)
    if frame_crop is not None:
        infos['position'][..., 0] -= frame_crop[0]
        infos['position'][..., 1] -= frame_crop[1]
    return infos

def animal_ear_detected(tag_list):
    for tag in tag_list:
        if 'animal_ears' in tag.lower():
            return True
    return False


def load_detected_character(srcd: str, pad_cropxyxy=True):
    '''
    load instance detected by AIS  
    return: instance_mask, xyxy, score  
    returns only one instance with max score, return None, None, None if no instance detected
    '''
    if osp.isfile(srcd):
        srcd = osp.dirname(srcd)
    instance_mask = None
    xyxy = None
    score = None
    instancep = osp.join(srcd, 'instances.json')
    if osp.exists(instancep):
        instance = json2dict(instancep)
        masks = instance['masks']
        if masks is not None and len(masks) > 0:
            max_ids = np.argmax(instance['scores'])
            score = instance['scores'][max_ids]
            bbox = instance['bboxes'][max_ids]
            xyxy = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            instance_mask = rle2mask(masks[max_ids])

    if pad_cropxyxy and instance_mask is not None:
        h, w = instance_mask.shape[:2]
        x1, y1, x2, y2 = xyxy
        iw, ih = x2 - x1, y2 - y1
        if iw < ih:
            p = (ih - iw) // 2
            x1 = max(x1 - p, 0)
            x2 = min(x2 + p, w)
        elif ih < iw:
            p = (iw - ih) // 2
            y1 = max(y1 - p, 0)
            y2 = min(y2 + p, h)
        xyxy = [x1, y1, x2, y2]

    return instance_mask, xyxy, score


def load_pos_estimation(model_dir, target='bizarre_pos.json'):
    p = osp.join(model_dir, target)
    if osp.exists(p):
        rst = json2dict(p)
        rst['pos'] = np.array(rst['pos'])
        rst['scores'] = np.array(rst['scores'])
        return rst
    else:
        return None


class ImageProcessor:

    def __init__(self,
        target_frame_size=None,
        crop_bbox=None,
        pad_to_square=True,
    ) -> None:
        self.target_frame_size = target_frame_size
        self.crop_bbox = crop_bbox
        self.pad_to_square = pad_to_square
        self.shift_before_scale = np.array([0, 0])
        self.shift_after_scale = np.array([0, 0])
        self.scale = np.array([1., 1.], dtype=np.float32)

    def final_pil_rgb(self):
        return Image.fromarray(self.final[..., :3])

    def final_pil_rgba(self):
        return Image.fromarray(self.final)

    def __call__(self, src: Union[Image.Image, np.ndarray, str], update_coords_modifiers=False) -> np.ndarray:

        if update_coords_modifiers:
            self.shift_before_scale = np.array([0, 0])
            self.shift_after_scale = np.array([0, 0])
            self.scale = np.array([1., 1.], dtype=np.float32)

        if isinstance(src, str):
            src = Image.open(src)
        if isinstance(src, np.ndarray):
            src = Image.fromarray(src)
        assert isinstance(src, Image.Image)
        if self.crop_bbox is not None:
            src = src.crop(self.crop_bbox)
            if update_coords_modifiers:
                self.shift_before_scale -= np.array(self.crop_bbox)[:2]
        if self.pad_to_square:
            src, padding = pil_pad_square(src)
            if update_coords_modifiers:
                self.shift_before_scale += np.array(padding)
        if self.target_frame_size is not None:
            if update_coords_modifiers:
                scale =  np.array(self.target_frame_size) / np.array([src.width, src.height])
                self.scale = self.scale * scale.astype(np.float32)
            # src = src.resize(self.target_frame_size, resample=Image.Resampling.NEAREST)
        # if self.to_np:
        src = np.array(src)
        if self.target_frame_size is not None:
            src = cv2.resize(src, dsize=self.target_frame_size, interpolation=cv2.INTER_AREA)
        return src

    def scale_coordinates(self, coordinates, to_int=False):
        '''
        coordinates: (N, 2) or (2,)
        '''

        assert isinstance(coordinates, (np.ndarray, list))
        if isinstance(coordinates, list):
            coordinates = np.array(coordinates, dtype=np.float32)
        else:
            coordinates = coordinates.astype(np.float32)

        one_element = True
        is_bbox = False
        if coordinates.ndim == 1:
            if len(coordinates) == 2:
                coordinates = coordinates[None]
            else:
                coordinates = coordinates.reshape(2, 2)
                is_bbox = True
                one_element = False
        else:
            assert coordinates.ndim == 2 and coordinates.shape[1] == 2
            one_element = False

        coordinates +=  self.shift_before_scale[None]
        coordinates *= self.scale[None]
        coordinates += self.shift_after_scale[None]

        if one_element:
            coordinates = coordinates[0]
        if is_bbox:
            coordinates = coordinates.flatten()

        if to_int:
            coordinates = np.round(coordinates).astype(np.int32)

        return coordinates


class Drawable:
    def __init__(self, 
                 img: np.ndarray = None, depth: int = -1, draw_order: int = -1, src_path: str = None, crop_xyxy=None, pad_drawable_img=True, final_size=None, 
                 seg_type='body_part_tag', src_size=None, meta_data=None, frame_crop=None, transform_matrix=None) -> None:
        self.img = img
        self.depth = depth  # 不要管这个值, 现在给的数据是错的
        self.draw_order = draw_order    # 绘制顺序
        self.src_path = src_path    # 源文件路径
        self.area = 0   # 面积
        self.img_processor: ImageProcessor = None
        self.final_visible_area = 0     # 可见部分面积
        self.final_visible_mask = None  # 可见部分 mask
        self.tag_stats = {}
        self.pos_stats = {}
        self.body_part_tag = None
        self.general_tags = None
        self.face_part_id = None
        self.face_part_stats = {}
        self.did = None
        self.pad_drawable_img = pad_drawable_img
        self._final_size = final_size
        self.idx = -1
        self._contours = None

        self._bbox = None
        self._final_visible_bbox = None
        self.src_size = src_size
        self.meta_data = meta_data
        self.frame_crop = frame_crop
        self.vertex_info = vertex_info_from_metadata(meta_data, self.src_size, transform_matrix, frame_crop)

        self.crop_xyxy= crop_xyxy
        
        if self.crop_xyxy is not None and not pad_drawable_img:
            self.x, self.y = self.crop_xyxy[:2]
        else:
            self.x = self.y = 0

        self.seg_type = seg_type

    @property
    def tag(self) -> str:
        return getattr(self, self.seg_type)
    
    def set_tag(self, value):
        if value is not None and value.lower() == 'none':
            value = None
        setattr(self, self.seg_type, value)
    
    def get_contours(self, simplify=True):
        if self._contours is None:
            self.get_img()
            cons, _ = cv2.findContours(self.img[..., -1].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if simplify:
                cnts = []
                for cnt in cons:
                    cnts.append(cv2.approxPolyDP(cnt, 3, True))
            else:
                cnts = cons
            self._contours = cnts
        return self._contours

    def get_vis_mask(self, global_xyxy=None, final_vis_mask=False):
        if final_vis_mask:
            mask = self.final_visible_mask
        else:
            mask = self.visible_mask
        if global_xyxy is None:
            return self.xyxy, mask
        bbox_i = bbox_intersection(self.xyxy, global_xyxy)
        if bbox_i is None:
            return None, None
        x1, y1, x2, y2 = bbox_i[0] - self.x, bbox_i[1] - self.y, bbox_i[2] - self.x, bbox_i[3] - self.y
        return bbox_i, mask[y1: y2, x1: x2]


    def mask_union_intersection(self, masks, global_xyxy=None, final_vis_mask=False):
        is_batch = True
        if masks.ndim == 2:
            masks = masks[None]
            is_batch = False
        bbox_i, vis_mask = self.get_vis_mask(global_xyxy=global_xyxy, final_vis_mask=final_vis_mask)
        if bbox_i is None:
            return None, None, None
        vis_mask = vis_mask[None]
        area = vis_mask.sum()
        if global_xyxy is not None:
            x, y = global_xyxy[0], global_xyxy[1]
            x1, y1, x2, y2 = bbox_i[0] - x, bbox_i[1] - y, bbox_i[2] - x, bbox_i[3] - y
        else:
            x1, y1, x2, y2 = bbox_i
        masks = masks[:, y1: y2, x1: x2]
        u_area = reduce(np.bitwise_or(vis_mask, masks), 'b h w -> b', 'sum').astype(np.float32)
        i_area = reduce(np.bitwise_and(vis_mask, masks), 'b h w -> b', 'sum').astype(np.float32)
        
        if not is_batch:
            u_area = u_area[0]
            i_area = i_area[0]
        
        return area, u_area, i_area

    
    def bitwise_and(self, mask, global_xyxy=None, final_vis_mask=False):
        bbox_i, vis_mask = self.get_vis_mask(global_xyxy=global_xyxy, final_vis_mask=final_vis_mask)
        if bbox_i is None:
            return np.zeros_like(mask)
        if global_xyxy is not None:
            x, y = global_xyxy[0], global_xyxy[1]
            x1, y1, x2, y2 = bbox_i[0] - x, bbox_i[1] - y, bbox_i[2] - x, bbox_i[3] - y
            mask = mask[y1: y2, x1: x2]
        return np.bitwise_and(mask, vis_mask)


    @property
    def x2(self):
        if isinstance(self.img, np.ndarray):
            return self.x + self.img.shape[1]
        return 0

    @property
    def y2(self):
        if isinstance(self.img, np.ndarray):
            return self.y + self.img.shape[0]
        return 0

    @property
    def xyxy(self):
        return [self.x, self.y, self.x2, self.y2]

    @property
    def xywh(self):
        return [self.x, self.y, self.x2 - self.x, self.y2 - self.y]

    @cached_property
    def visible_xyxy(self):
        x1, y1, x2, y2 = cv2.boundingRect(cv2.findNonZero(self.final_visible_mask.astype(np.uint8)))
        x1 += self.x
        x2 += x1
        y1 += self.y
        y2 += y1
        return [x1, y1, x2, y2]

    @property
    def final_size(self):
        '''
        it's the size of the final scene
        '''
        if self._final_size is None:
            raise Exception(f'Final size is not set for this drawable, calling Live2dScrapModel.init_basics should solve it')
        return self._final_size

    def set_img_processor(self, img_processor: callable):
        self.img_processor = img_processor

    def load_img(self, force_reload=False, img=None):
        if self.img is not None and not force_reload:
            return
        if img is None:
            img = Image.open(self.src_path)
            W, H = self.src_size
            # self.vertex_info = load_drawable_vertex_info(self.src_path, H, W)
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            else:
                assert isinstance(img, np.ndarray)

        if self.crop_xyxy is not None \
            and self.pad_drawable_img:
            # tw, th = self.transform_stats['src_size']
            th, tw = self._final_size
            x, y, x2, y2 = self.crop_xyxy
            img = ImageOps.expand(img, (x, y, tw - x2, th - y2))
            self.x = self.y = 0

        if self.img_processor is not None:
            if self.pad_drawable_img:
                img = self.img_processor(img)
            else:
                if self.img_processor.crop_bbox is not  None:
                    h, w = img.height, img.width
                    dx1, dy1 = self.x, self.y
                    dx2, dy2 = dx1 + w, dy1 + h
                    # x1, y1, x2, y2 = self.img_processor.crop_bbox
                    intersection = bbox_intersection([dx1, dy1, dx2, dy2], self.img_processor.crop_bbox)
                    if intersection is None:    # some weird boundary case
                        self.init_empty_img()
                        return
                    ix1, iy1, ix2, iy2 = intersection
                    cx1, cx2 = ix1 - dx1, ix2 - dx1
                    cy1, cy2 = iy1 - dy1, iy2 - dy1
                    if cx1 != 0 or cx2 != w or cy1 != 0 or cy2 != h:
                        self.x = ix1
                        self.y = iy1
                        img = img.crop((cx1, cy1, cx2, cy2))
                self.x, self.y = self.img_processor.scale_coordinates([self.x, self.y], to_int=True)
                if self.vertex_info is not None:
                    self.vertex_info['position'] = self.img_processor.scale_coordinates(self.vertex_info['position'], to_int=True)
                img = np.array(img)
                sw, sh = self.img_processor.scale
                h, w = img.shape[:2]
                tw = int(round(w * sw))
                th = int(round(h * sh))
                # tw = math.floor(w * sw)
                # th = math.floor(h * sh)
                if tw == 0 or th == 0:
                    self.init_empty_img()
                    return
                if h != th or w != tw:
                    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                    fsz = self.final_size
                    if th + self.y > fsz[0]:
                        img = img[:fsz[0] - self.y]
                    if tw + self.x > fsz[1]:
                        img = img[:, :fsz[1] - self.x]

        # if self.x == 0 and self.y == 0 and 

        visible_mask = img[..., -1] > MINIMUM_VISIBLE_ALPHA
        self.area = visible_mask.sum()
        if self.area > 0:
            self.visible_mask = visible_mask
            self.img = img
            bbox = cv2.boundingRect(cv2.findNonZero(self.visible_mask.astype(np.uint8)))
            self._bbox = np.array(bbox)
            self._bbox[0] += self.x
            self._bbox[1] += self.y
        else:
            self.visible_mask = 0
            self.img = 0
            self._bbox = [0, 0, 0, 0]
        self.final_visible_area = self.area

    def init_empty_img(self):
        self.area = self.visible_mask = self.final_visible_area = self.img = 0
        self._bbox = [0, 0, 0, 0]

    def get_img(self):
        if self.img is None:
            self.load_img()
        return self.img
    
    def get_bbox(self, xyxy=None):
        if self.img is None:
            self.load_img()
        bbox = self._bbox.copy()
        if xyxy is not None:
            bbox[0] = max(bbox[0] - xyxy[0], 0)
            bbox[1] = max(bbox[1] - xyxy[1], 0)
            # bbox[1] = 
            # bbox[1] -= xyxy[1]
        return bbox

    def to_local_pos(self, x: int, y: int):
        return [x - self.x, y - self.y]

    def get_final_visible_bbox(self):
        if self._final_visible_bbox is None:
            bbox = cv2.boundingRect(cv2.findNonZero(self.final_visible_mask.astype(np.uint8)))
            self._final_visible_bbox = np.array(bbox)
            self._final_visible_bbox[0] += self.x
            self._final_visible_bbox[1] += self.y
        return self._final_visible_bbox.copy()

    def set_final_visible_mask(self, visible_mask: np.ndarray):
        self.final_visible_mask = visible_mask
        self.final_visible_area = visible_mask.sum()

    def get_full_mask(self, final_visible_mask=False, xyxy=None):
        if final_visible_mask:
            mask = self.final_visible_mask
        else:
            mask = self.visible_mask
        if self.pad_drawable_img:
            full_mask = mask
        else:
            full_mask = np.zeros(self.final_size, dtype=bool)
            x1, y1, x2, y2 = self.xyxy
            full_mask[y1: y2, x1: x2] = mask
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            full_mask = full_mask[y1: y2, x1: x2]
        return full_mask


class Part:
    def __init__(self, directory: str, child_parts: dict = None, child_drawables: list = None, img: np.ndarray = None, root_directory=None) -> None:
        self.directory = directory
        if child_parts is None:
            child_parts = {}
        self.child_parts = child_parts
        if child_drawables is None:
            child_drawables = []
        self.child_drawables = child_drawables
        self.img = img
        self.area = 0
        self.root_directory = root_directory

    def get_sub_drawables(self) -> List[Drawable]:
        drawable_lst = []
        for part_name, child_part in self.child_parts.items():
            drawable_lst += child_part.get_sub_drawables()
        drawable_lst += self.child_drawables
        return drawable_lst

    def set_img_processor(self, img_processor: callable):
        self.img_processor = img_processor

    def load_img(self, force_reload=False):
        if self.img is not None and not force_reload:
            return
        drawable_lst = self.get_sub_drawables()
        drawable_lst.sort(key = lambda x: x.draw_order)
        img = compose_from_drawables(drawable_lst)
        self.area = img[..., -1].sum()
        self.img = img

    def get_img(self):
        if self.img is None:
            self.load_img()
        return self.img

    def write_composed_img(self, save_dir=None, recursive=False):
        if save_dir is None:
            save_dir = osp.join(self.root_directory, self.directory)
        img = self.get_img()
        Image.fromarray(img).save(osp.join(save_dir, osp.basename(self.directory) + '.png'))
        if recursive:
            for part_name, child_part in self.child_parts.items():
                child_part.write_composed_img(recursive=True)


def compose_from_drawables(drawables: List[Drawable], xyxy=None, output_type='numpy', premultiplied=True):
    '''
    drawables should be sorted beforehand
    Returns: uint8 rgb image with alpha channel
    '''
    if isinstance(drawables, (Drawable, np.ndarray)):
        drawables = [drawables]

    imgs = []
    final_sz = None
    for d in drawables:
        if isinstance(d, Drawable):
            img = {'img': d.get_img(), 'xyxy': d.xyxy}
            final_sz = d.final_size
            if d.area == 0:
                continue
        else:
            assert isinstance(d, np.ndarray)
            img = d
        imgs.append(img)

    return img_alpha_blending(imgs, xyxy=xyxy, output_type=output_type, final_size=final_sz, premultiplied=premultiplied)


def fix_drawable_rgbs(drawables: List[Drawable], xyxy=None, output_type='numpy', final_size=None):
    '''
    final_size: (h, w)
    '''
    import time
    t0 = time.time()

    if isinstance(drawables, (np.ndarray, dict)):
        drawables = [drawables]

    final_size = drawables[0].final_size
    # # infer final scene size
    if xyxy is not None:
        final_size = [xyxy[3] - xyxy[1], xyxy[2] - xyxy[0]]
        x1, y1, x2, y2 = xyxy
    else:
        x1, y1, x2, y2 = 0, 0, final_size[1], final_size[0]
    # elif final_size is None:
    #     d = drawables[0]
    #     if isinstance(d, dict):
    #         d = d['img']
    #     final_size = d.shape[:2]

    final_rgb = np.zeros((final_size[0], final_size[1], 3), dtype=np.float32)
    final_alpha = np.zeros_like(final_rgb[..., [0]])

    for drawable in drawables:
        drawable_img = drawable.get_img()
        dxyxy = drawable.xyxy
        dx1, dy1, dx2, dy2 = dxyxy

        if drawable_img.ndim == 3 and drawable_img.shape[-1] == 3:
            drawable_alpha = np.ones_like(drawable_img[..., [-1]])
        else:
            drawable_alpha = drawable_img[..., [-1]].astype(np.float32) / 255

        drawable_img = drawable_img[..., :3]

        if xyxy is not None:
            if dxyxy is None:
                drawable_img = drawable_img[y1: y2, x1: x2]
            else:
                intersection = bbox_intersection(xyxy, dxyxy)
                if intersection is None:
                    continue
                ix1, iy1, ix2, iy2 = intersection
                drawable_alpha = drawable_alpha[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                final_alpha[iy1-y1: iy2-y1, ix1-x1: ix2-x1] += drawable_alpha
                drawable_img = drawable_img[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1] = final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1] * (1-drawable_alpha) + drawable_img
                continue

        if dxyxy is None:
            final_alpha += drawable_alpha
            final_rgb = final_rgb * (1 - drawable_alpha) + drawable_img
        else:
            final_alpha[dy1: dy2, dx1: dx2] += drawable_alpha
            final_rgb[dy1: dy2, dx1: dx2] = final_rgb[dy1: dy2, dx1: dx2] * (1-drawable_alpha) + drawable_img
        drawable.img[..., :3] = np.clip(final_rgb[dy1: dy2, dx1: dx2], 0, 255).astype(np.uint8)
        # drawable.img[..., :3] = np.clip(drawable.img[..., :3] * drawable_alpha, 0, 255).astype(np.uint8)
        # from utils.io_utils import save_tmp_img
        # save_tmp_img((drawable_alpha * 255).astype(np.uint8))
        # save_tmp_img((final_rgb).astype(np.uint8))
        # pass
        


def compose_mask_from_drawables(drawables: List[Drawable], xyxy=None, final_visible_mask=False, output_type='numpy'):
    if xyxy is not None:
        x1, y1, x2, y2 = xyxy
    mask = np.zeros(drawables[0].final_size, dtype=bool)
    if xyxy is not None:
        mask = mask[y1: y2, x1: x2]
    if len(drawables) > 0:
        for d in drawables:
            d.get_img() # incase image not loaded
            if final_visible_mask:
                if d.final_visible_area == 0:
                    continue
                tmsk = d.final_visible_mask
            else:
                if d.area == 0:
                    continue
                tmsk = d.visible_mask
            dx1, dy1, dx2, dy2 = d.xyxy
            if xyxy is not None:
                if d.pad_drawable_img:
                    tmsk = tmsk[y1: y2, x1: x2]
                else:
                    intersection = bbox_intersection(d.xyxy, xyxy)
                    if intersection is None:
                        continue
                    ix1, iy1, ix2, iy2 = intersection
                    tmsk = tmsk[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                    mask[iy1-y1: iy2-y1, ix1-x1: ix2-x1] |= tmsk
                    continue
            if d.pad_drawable_img:
                mask |= tmsk
            else:
                mask[dy1: dy2, dx1: dx2] |= tmsk
    if output_type.lower() == 'pil':
        mask = Image.fromarray(mask)
    return mask


def init_drawable_visible_map(drawables: List[Drawable], final_alpha=None):
    # final_alpha = np.zeros_like(self.final[..., 0])

    opacity_map_lst = []
    for ii, drawable in enumerate(drawables):
        drawable_img = drawable.get_img()
        if drawable.area < 1:
            opacity_map_lst.append(None)
            continue
        if final_alpha is None:
            final_alpha = np.zeros(drawable.final_size).astype(np.float32)
        drawable_alpha = drawable_img[..., -1] / 255.
        if drawable.pad_drawable_img:
            opacity_map_lst.append(final_alpha + drawable_alpha)
            final_alpha = opacity_map_lst[-1]
        else:
            dx1, dy1, dx2, dy2 = drawable.xyxy
            final_alpha[dy1: dy2, dx1: dx2] += drawable_alpha
            opacity_map_lst.append(final_alpha[dy1: dy2, dx1: dx2].copy())

    for ii, opacity_map in enumerate(opacity_map_lst):
        drawable = drawables[ii]
        if opacity_map is None:
            drawable.final_visible_area = 0
            continue
        if drawable.pad_drawable_img:
            occulsion_mask = np.clip(final_alpha - opacity_map, 0, 1) > 0.95
        else:
            dx1, dy1, dx2, dy2 = drawable.xyxy
            occulsion_mask = np.clip(final_alpha[dy1: dy2, dx1: dx2] - opacity_map, 0, 1) > 0.9
        visible_mask = np.bitwise_and(np.bitwise_not(occulsion_mask), drawable.visible_mask)
        drawable.set_final_visible_mask(visible_mask)




class Live2DScrapModel:

    def __init__(
            self,
            model_directory: str,
            target_frame_size=None,
            crop_to_final=True,
            pad_to_square=True,
            skip_load=False,
            pad_drawable_img=False,
            crop_xyxy=None,
            seg_type='body_part_tag'
        ):
        if osp.isfile(model_directory):
            model_directory = osp.dirname(model_directory)
        self.directory = osp.normpath(model_directory)
        if isinstance(target_frame_size, int):
            target_frame_size = (target_frame_size, target_frame_size)

        self.final = None
        self.src_size = None
        self.target_frame_size = target_frame_size
        self.crop_to_final = crop_to_final
        self.crop_xyxy = crop_xyxy # will overwrite crop_to_final
        self.final_bbox = None
        self.pad_to_square = pad_to_square
        self.child_parts = {}
        self.child_drawables = []
        self.drawables: List[Drawable] = []
        self.processor  =  lambda x: x
        self.final_visible_mask: np.ndarray = None
        self.tag_stats = {}
        self.pos = None # note it's [N, (y, x)]
        self.facedet = None
        self.did2drawable: dict[str, Drawable] = {}
        self._img_ext = '.png'
        self.vismap_initialized = False
        self._face_parsing = None
        self._body_parsing = None
        self._body_parsing_src = None
        self.pad_drawable_img = pad_drawable_img
        self._valid_drawables = []
        self.seg_type = seg_type

        self.meta = None;

        self.frame_crop = None
        metap = osp.join(self.directory, 'meta.yaml')
        if osp.exists(metap):
            self.meta = json2dict(metap)
            self.frame_crop = self.meta.get('crop_xyxy', None)
            # self.crop_xyxy = self.meta.get('crop_xyxy', None)

        if not skip_load:
            self._load_model()

    def size(self):
        return self.final.shape[:2]

    def valid_drawables(self):
        return [d for d in self.drawables if d.area > 0]

    def get_child_part_by_name(self, part_name) -> Part:
        if part_name in self.child_parts:
            return self.child_parts[part_name]
        else:
            return None

    def get_part_by_path(self, p: str, init_missing_parts=False) -> Part:
        p = p.replace(self.directory, '')
        if p == '':
            return self
        parts = list(Path(p).parts)[1:]
        parent = self
        for ii, part_name in enumerate(parts):
            if part_name in parent.child_parts:
                parent = parent.child_parts[part_name]
            else:
                if init_missing_parts:
                    parent.child_parts[part_name] = Part(osp.join(*parts[:ii+1]), root_directory=self.directory)
                    parent = parent.child_parts[part_name]
                else:
                    return None
        return parent

    def init_drawable_visible_map(self):
        if self.vismap_initialized:
            return
        init_drawable_visible_map(self.drawables)
        self._valid_drawables.clear()
        didx = 0
        for d in self.drawables:
            if d.area > 0:
                self._valid_drawables.append(d)
            d.idx = didx
            didx += 1
        self.vismap_initialized = True

    def valid_parsing_list(self):
        parsing_lst = []
        for d in os.listdir(self.directory):
            if d.lower().endswith('.json'):
                if d.startswith('parsinglog_') or d in {'body_parsing.json', 'face_parsing.json', 'bodyparsingv3.json'}:
                    parsing_lst.append(d)
        return parsing_lst

    def face_detected(self):
        return self.facedet is not None and len(self.facedet) > 0

    def _load_model(self):
        self.vismap_initialized = False
        self.init_basics()
        self.init_drawables(img_ext=self._img_ext)

    def init_basics(self):
        final = get_last_modified_file(osp.join(self.directory, 'final'), ['.jxl', '.png', '.webp'])
        self._img_ext = osp.splitext(final)[-1]
        final = Image.open(final)
        self.final_bbox = None
        self.src_size = (final.width, final.height)
        if self.meta is not None:
            self.src_size = self.meta['canvas_size']
        if self.crop_xyxy is None:
            if self.crop_to_final:
                alpha = final.split()[-1]
                self.final_bbox = alpha.getbbox()
        else:
            self.final_bbox = self.crop_xyxy
        self.processor = ImageProcessor(self.target_frame_size, self.final_bbox, self.pad_to_square)

        self.final = final = self.processor(final, update_coords_modifiers=True)
        self.final_visible_mask = self.final[..., -1] > MINIMUM_VISIBLE_ALPHA

        self.facedet = None
    
        facedetp = osp.join(self.directory, 'facedet.json')
        if osp.exists(facedetp):
            self.facedet = json2dict(facedetp)
            for ii in range(len(self.facedet)):
                bbox = np.array(self.facedet[ii]['bbox'])
                bbox[:4] = self.processor.scale_coordinates(bbox[:4], to_int=True)
                self.facedet[ii]['bbox'] = bbox
                kpts = np.array(self.facedet[ii]['keypoints'])
                kpts[..., :2] = self.processor.scale_coordinates(kpts[..., :2], to_int=True)
                self.facedet[ii]['keypoints'] = kpts
            pass

        pos_path = osp.join(self.directory, 'pos.json')
        if osp.exists(pos_path):
            self.pos = json2dict(pos_path)
            self.pos['keypoints'] = self.processor.scale_coordinates(np.array(self.pos['keypoints'], dtype=np.float32)[..., ::-1])[..., ::-1]
            self.pos['bbox'] = self.processor.scale_coordinates(np.array(self.pos['bbox'], dtype=np.float32))
            self.pos['parts'] = {}
            for k, v in coco_parts_dict.items():
                self.pos['parts'][k] = [self.pos['keypoints'][v[0]], self.pos['keypoints'][v[1]]]


    def init_drawables(self, img_ext='.jxl', filter_drawables=False):
        self.did2drawable.clear()
        self.child_parts.clear()
        self.drawables.clear()
        drawable_transfrom_stats = osp.join(self.directory, 'transform_stats.json')
        if osp.exists(drawable_transfrom_stats):
            drawable_transfrom_stats = json2dict(drawable_transfrom_stats)
        else:
            drawable_transfrom_stats = {}

        drawable_plist = find_all_files_recursive(self.directory, ext={img_ext})

        if self.final is not None:
            final_sz = self.final.shape[:2]
        else:
            final_sz = None

        transform_matrix= None
        if self.meta is not None:
            transform_matrix = np.array(self.meta['transform_matrix']).reshape((-1, 4))

        for p in drawable_plist:
            drawable_name = osp.splitext(osp.basename(p))[0]
            if drawable_name == 'final':
                continue
            drawable_name_splits = drawable_name.split('-')
            if len(drawable_name_splits) < 3 or not drawable_name_splits[0].isdigit() or not drawable_name_splits[-1].isdigit():
                # print(f'{p} is not a valid drawable')
                continue
            draw_order = int(drawable_name_splits[0])
            depth = int(drawable_name_splits[-1])

            crop_xyxy = None
            drawable_tr = drawable_transfrom_stats.get(str(draw_order), None)
            if drawable_tr is not None:
                crop_xyxy = [x for x in drawable_tr['bbox']]
                crop_xyxy[2] += crop_xyxy[0]
                crop_xyxy[3] += crop_xyxy[1]

            drawable_metadata = None
            
            if self.meta is not None:
                drawable_metadata = self.meta['drawables'][draw_order]
                crop_xyxy = drawable_metadata.get("crop_xyxy", None)
                if crop_xyxy is not None and self.frame_crop is not None:
                    crop_xyxy = crop_xyxy.copy()
                    crop_xyxy[0] -= self.frame_crop[0]
                    crop_xyxy[1] -= self.frame_crop[1]
                    crop_xyxy[2] -= self.frame_crop[0]
                    crop_xyxy[3] -= self.frame_crop[1]

            self.drawables.append(
                Drawable(
                    depth=-1, draw_order=draw_order, src_path=osp.normpath(p), crop_xyxy=crop_xyxy, pad_drawable_img=self.pad_drawable_img, 
                    final_size=final_sz, seg_type=self.seg_type, src_size=self.src_size, meta_data=drawable_metadata, frame_crop=self.frame_crop, transform_matrix=transform_matrix)
            )
        self.drawables.sort(key=lambda x: x.draw_order)

        for drawable in self.drawables:
            drawable_parent_d = osp.dirname(drawable.src_path)
            part = self.get_part_by_path(drawable_parent_d, init_missing_parts=True)
            part.child_drawables.append(drawable)

        if len(self.drawables) > 0:
            for drawable in self.drawables[::-1]:
                did = osp.relpath(osp.realpath(drawable.src_path), osp.realpath(self.directory)).split(os.sep)
                did_src, bname = did[:-1], did[-1]
                bname_complete = osp.splitext(bname)[0]
                bname = bname_complete.split('-')[1:-1]
                bname = ''.join(bname)
                did = ('/'.join(did_src) + '/' + bname).lstrip('/')
                if did in self.did2drawable:
                    did = ('/'.join(did_src) + '/' + bname_complete).lstrip('/')
                drawable.did = did
                self.did2drawable[did] = drawable
                if self.processor is not None:
                    drawable.set_img_processor(self.processor)

    def set_seg_type(self, seg_type: str):
        for d in self.drawables:
            d.seg_type = seg_type
        self.seg_type = seg_type

    def update_tag_stats(self, score_map: np.ndarray, cls_idx: int, cls_name: str, filter_scoremap=True):
        if filter_scoremap:
            score_map = score_map * self.final_visible_mask
        score_sum = score_map.sum()
        contribution_sum = 0
        for drawable in self.drawables:
            if drawable.final_visible_area < 1:
                stats = {'contribution_score': 0, 'avg_score': 0}
            else:
                x1, y1, x2, y2 = drawable.xyxy
                drawable_scoremap = score_map[y1: y2, x1: x2] * drawable.final_visible_mask
                drawable_score = drawable_scoremap.sum()
                avg_score = drawable_score / drawable.final_visible_area
                contribution_score = drawable_score / score_sum
                stats = {'contribution_score': contribution_score, 'avg_score': avg_score}
                contribution_sum += contribution_score
            drawable.tag_stats[cls_name] = stats
        self.tag_stats[cls_name] = {'score_sum': score_sum, 'cls_idx': cls_idx, 'contribution_sum': contribution_sum}

    def get_drawable_parent(self, drawable: Drawable) -> Part:
        if isinstance(drawable, Drawable):
            drawable = drawable.src_path
        drawable = osp.normpath(drawable)
        part = self.get_part_by_path(drawable)
        return part

    def save_tag_stats(self, savep: str = None):
        if savep is None:
            savep = osp.join(self.directory, 'tag_stats.json')
        d = {'tag_info': self.tag_stats, 'drawable_tag_stats': [d.tag_stats for d in self.drawables]}
        dict2json(d, savep)

    def load_tag_stats(self, srcp: str = None):
        if srcp is None:
            srcp = osp.join(self.directory, 'tag_stats.json')
        if not osp.exists(srcp):
            return False
        d = json2dict(srcp)
        self.tag_stats.clear()
        self.tag_stats.update(d['tag_info'])
        assert len(self.drawables) == len(d['drawable_tag_stats'])
        for drawable, tag_stats in zip(self.drawables, d['drawable_tag_stats']):
            drawable.tag_stats.clear()
            drawable.tag_stats.update(tag_stats)
        return True
    
    def compose_drawables(self, drawables, mask_only=False, xyxy=None, final_visible_mask=False, output_type='numpy'):
        if len(drawables) > 0:
            drawables.sort(key=lambda x: x.draw_order)
            if mask_only:
                return compose_mask_from_drawables(drawables, xyxy=xyxy, final_visible_mask=final_visible_mask, output_type=output_type)
            return compose_from_drawables(drawables, xyxy=xyxy, output_type=output_type)
        else:
            h, w = self.final.shape[:2]
            if xyxy is not None:
                h, w = xyxy[3] - xyxy[1], xyxy[2] - xyxy[0]
            if mask_only:
                if output_type.lower() == 'pil':
                    return Image.fromarray(np.zeros((h, w), dtype=np.uint8))
                return np.zeros((h, w), dtype=bool)
            else:
                if output_type.lower() == 'pil':
                    return Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8))
                return np.zeros((h, w, 4), dtype=np.uint8)


    def compose_face_drawables(self, face_part_ids: Union[int, list, set], mask_only=False, xyxy=None, final_visible_mask=False, output_type='numpy'):
        drawables = []
        if isinstance(face_part_ids, int):
            face_part_ids = [face_part_ids]
        face_part_ids = set(face_part_ids)
        for d in self.drawables:
            if d.face_part_id in face_part_ids:
                drawables.append(d)
        return self.compose_drawables(drawables, mask_only=mask_only, xyxy=xyxy, final_visible_mask=final_visible_mask, output_type=output_type)


    def compose_bodypart_drawables(self, body_part_tags: Union[str, list, set], mask_only=False, xyxy=None, final_visible_mask=False, output_type='numpy', visible_area_check=True):
        drawables = []
        if isinstance(body_part_tags, str) or body_part_tags is None:
            body_part_tags = [body_part_tags]
        body_part_tags = set(body_part_tags)
        for d in self.drawables:
            if d.body_part_tag in body_part_tags and d.area > 0:
                drawables.append(d)
        return self.compose_drawables(drawables, mask_only=mask_only, xyxy=xyxy, final_visible_mask=final_visible_mask, output_type=output_type)


    def save_model_to(self, target_dir: str, crop_to_final=False, img_ext='.png', skip_invisible=False):
        if osp.exists(target_dir):
            # assert len(os.listdir(target_dir)) == 0, f'Target dir: {target_dir} is not empty!'
            pass
        else:
            os.makedirs(target_dir)
        
        tagp = osp.join(self.directory, 'tag_stats.json')
        if osp.exists(tagp):
            shutil.copy(tagp, osp.join(target_dir, 'tag_stats.json'))
        
        if self.pos is not None:
            pos = {k: v for k, v in self.pos.items()}
            if 'parts' in pos:
                pos.pop('parts')
            dict2json(pos, osp.join(target_dir, 'pos.json'))

        imwrite(osp.join(target_dir, 'final.png'), self.final, ext=img_ext)
        transform_stats = {}
        ndir_leading = len(self.directory.split(os.path.sep))
        for drawable in self.drawables:
            img = drawable.get_img()
            if drawable.area < 1:
                continue

            dsavep = drawable.src_path.split(os.path.sep)[ndir_leading:]
            dsavep = osp.join(target_dir, *dsavep)
            dsave_dir = osp.dirname(dsavep)
            if not osp.exists(dsave_dir):
                os.makedirs(dsave_dir)

            if crop_to_final:
                br = cv2.boundingRect(cv2.findNonZero(drawable.visible_mask.astype(np.uint8)))
                transform_stats[drawable.draw_order] = {'bbox': br}
                x, y, w, h = br
                img = img[y: y+h, x: x+w]
            imwrite(dsavep, img, ext=img_ext)
            if drawable.vertex_info is not None:
                dict2json(drawable.vertex_info, osp.splitext(dsavep)[0] + '.json')
            # vertexp = osp.splitext(drawable.src_path)[0] + '.txt'
            # if osp.exists(vertexp):
            #     shutil.copy(vertexp, osp.splitext(dsavep)[0] + '.txt')

        dict2json(transform_stats, osp.join(target_dir, 'transform_stats.json'))


    def brow_detected(self):
        return self.face_part_detected([2, 3])


    def face_part_detected(self, part_ids: Union[List[int], int]):
        single = isinstance(part_ids, int)
        if single:
            part_ids = [part_ids]
        part_detected = [False] * len(part_ids)
        partid2rstid = {pi: ii for pi, ii in zip(part_ids, list(range(len(part_ids))))}
        for drawable in self.drawables:
            for pi in part_ids:
                if drawable.face_part_id == pi:
                    part_detected[partid2rstid[pi]] = True
        if single:
            return part_detected[0]
        return (*part_detected, )
    

    def save_face_parsing(self, metadata=None, save_name=None, face_seg_xyxy=None):
        face_parsing = {'metadata': metadata, 'drawable': {}, 'xyxy': face_seg_xyxy}

        for d in self.drawables:
            if d.face_part_id is not None:
                face_parsing['drawable'][d.did] = d.face_part_id
        
        if save_name is None:
            save_name = 'face_parsing'
        if not save_name.lower().endswith('.json'):
            save_name = save_name + '.json'

        dict2json(face_parsing, osp.join(self.directory, save_name))


    def save_body_parsing(self, metadata=None, save_name=None, xyxy=None):

        if xyxy is None:
            xyxy = self.final_bbox

        body_parsing = {'metadata': metadata, 'drawable': {}, 'xyxy': xyxy}

        for d in self.drawables:
            if d.body_part_tag is not None:
                body_parsing['drawable'][d.did] = d.body_part_tag
        
        if save_name is None:
            if self._body_parsing_src is None:
                save_name = 'body_parsing'
            else:
                save_name = osp.basename(self._body_parsing_src)
        if not save_name.lower().endswith('.json'):
            save_name = save_name + '.json'

        dict2json(body_parsing, osp.join(self.directory, save_name))


    def save_tag_parsing(self, seg_type, save_name, metadata=None, xyxy=None):

        if xyxy is None:
            xyxy = self.final_bbox

        parsing = {'metadata': metadata, 'drawable': {}, 'xyxy': xyxy}

        for d in self.drawables:
            tag = getattr(d, seg_type)
            if tag is not None:
                parsing['drawable'][d.did] = tag

        if not save_name.lower().endswith('.json'):
            save_name = save_name + '.json'

        dict2json(parsing, osp.join(self.directory, save_name))


    def load_face_parsing(self, src: str = None):
        if src is None:
            src = osp.join(self.directory, 'face_parsing.json')
        if not osp.exists(src):
            print(f'failed to load face parsing file, {src} does not exist!')
            return False
        fd = json2dict(src)
        for d, v in fd['drawable'].items():
            d = d.lstrip('/')
            self.did2drawable[d].face_part_id = v
        self._face_parsing = fd
        return True

    def load_body_parsing(self, src: str = None):
        if src is None:
            src = osp.join(self.directory, 'body_parsing.json')
        else:
            if not osp.exists(src):
                src = osp.join(self.directory, src)
        if not osp.exists(src):
            print(f'failed to load body parsing file, {src} does not exist!')
            return False
        fd = json2dict(src)

        if 'metadata' in fd and fd['metadata'] is not None and 'tag_valid' in fd['metadata']:
            for k in list(fd['metadata']['tag_valid'].keys()):
                if k in _tag_cvt:
                    fd['metadata']['tag_valid'][_tag_cvt[k]] = fd['metadata']['tag_valid'].pop(k)

        for d, v in fd['drawable'].items():
            if v in _tag_cvt:
                v = _tag_cvt[v]
            d = d.lstrip('/')
            self.did2drawable[d].body_part_tag = v
        self._body_parsing = fd
        self._body_parsing_src = src
        return True

    @property
    def face_seg_xyxy(self):
        if self._face_parsing is not None and 'xyxy' in self._face_parsing:
            xyxy = self._face_parsing['xyxy']
            return self.processor.scale_coordinates(np.array(xyxy), to_int=True)
        return None
    

    def maxios_mindrawable(self, cls_id, thr = -1, ioa_thr=-1):
        min_d = None
        min_area = self.final.shape[0] * self.final.shape[1]
        max_ios = 0
        for d in self.drawables:
            if 'ios' not in d.face_part_stats:
                continue
            ios = d.face_part_stats['ios'][cls_id]
            if ios < thr:
                continue
            if d.face_part_stats['ioa'][cls_id] < ioa_thr:
                continue
            if ios > max_ios and d.final_visible_area < min_area:
                min_area = d.final_visible_area
                max_ios = ios
                min_d = d
            # if d.face_part_stats['ios'][cls_id] 

        return min_d


    def get_body_part_drawables(self, body_part_tag: str):
        if isinstance(body_part_tag, str):
            return [d for d in self.drawables if d.body_part_tag == body_part_tag]
        lst = []
        assert isinstance(body_part_tag, list)
        for d in self.drawables:
            if d.body_part_tag in body_part_tag:
                lst.append(d)
        return lst
    


def get_tag_voting_from_lmodel(src: str, seg_type='body_part_tag', parsing_src: str = 'parsinglog_sambody_iter1_step18k_masks.json'):
    if isinstance(src, Live2DScrapModel):
        lmodel = src
    else:
        lmodel = Live2DScrapModel(src, skip_load=True, seg_type=seg_type)
        lmodel.init_drawables()
        lmodel.load_body_parsing(parsing_src)
    did2tag = {}
    dir2tag = {}
    for d in lmodel.drawables:
        if d.did is None:
            continue
        did2tag[d.did] = d.tag
        ddir = osp.dirname(d.did)
        if ddir in dir2tag:
            dlist = dir2tag[ddir]
        else:
            dlist = {}
            dir2tag[ddir] = dlist
        dlist[osp.basename(d.did)] = d.tag

    return lmodel, dir2tag, did2tag


def get_common_prefix_exclude_digits(s1, s2):
    """
    Returns the longest common prefix of two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        str: The longest common prefix of s1 and s2.
    """
    common_prefix = ""
    min_length = min(len(s1), len(s2))

    for i in range(min_length):
        if s1[i] == s2[i] and not s1[i].isdigit():
            common_prefix += s1[i]
        else:
            break
    return common_prefix


def match_drawable_to_tag_voting(lmodel: Live2DScrapModel, dir2tag, did2tag, check_area=False):
    valid_n = 0
    rst_matching = {}
    src_matching = {}
    id_matched = []
    dir_matched = []
    for d in lmodel.drawables:
        if d.did is None:
            continue
        if check_area and d.area < 1:
            continue
        valid_n += 1
        src_matching[d.did] = d.tag
        if d.did in did2tag:
            # d.set_tag(did2tag[d.did])
            rst_matching[d.did] = did2tag[d.did]
            id_matched.append(d.did)
        elif osp.dirname(d.did) in dir2tag:
            dname = osp.basename(d.did)
            ddict = dir2tag[osp.dirname(d.did)]
            tag = ''
            tag_dist = -1
            for k, v in ddict.items():
                prefix = get_common_prefix_exclude_digits(dname, k)
                kd = k[len(prefix):]
                dnamed = dname[len(prefix):]
                if kd.isdigit() and dnamed.isdigit():
                    kd = int(kd)
                    dnamed = int(dnamed)
                    dist = abs(kd - dnamed)
                    if tag_dist < 0 or dist < tag_dist:
                        tag = v
                        tag_dist = dist
            if tag != '':
                # d.set_tag(tag)
                rst_matching[d.did] = tag
                dir_matched.append(d.did)

    print(f'total valid drawables: {valid_n}, id matched: {len(id_matched)} dir matched: {len(dir_matched)}')

    return src_matching, rst_matching, (id_matched, dir_matched)
