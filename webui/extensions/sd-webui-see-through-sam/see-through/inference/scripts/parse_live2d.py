from PIL import Image

import random
import os
import os.path as osp
import numpy as np
from  tqdm import tqdm
from einops import reduce
import click
import cv2
import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from utils.io_utils import load_exec_list, pil_pad_square, pil_ensure_rgb, imglist2imgrid, find_all_files_with_name, dict2json, json2dict, load_image, save_tmp_img
from live2d.scrap_model import Live2DScrapModel, compose_from_drawables, load_detected_character, init_drawable_visible_map, Drawable
from utils.visualize import pil_draw_text, visualize_segs, VALID_FACE_GROUPS, FACE_LABEL2NAME, visualize_facedet_output, LEFT_EYEBROW, RIGHT_EYEBROW, show_factorization_on_image
from utils.cv import mask2rle, rle2mask

exclude_cls = \
{
    '1girl',
    'smile',
    'simple_background',
    'white_background',
    'solo',
    'closed_mouth',
    'looking_at_viewer',
    'standing',
    'full_body',
    'virtual_youtuber',
    'tachi-e',
    'elf',
    'transparent_background',
    'blush',
    'straight-on',
    'looking_to_the_side',
    'expressionless',
    'holding',
}

@click.group()
def cli():
    """live2d scripts.
    """


@cli.command('build_live2d_exec_list')
@click.option('--srcd')
@click.option('--save_dir', default=None)
@click.option('--filter_p', default=None)
@click.option('--target_fno', default=-1)
@click.option('--num_chunk', default=-1)
@click.option('--save_name', default='exec_list')
def build_live2d_exec_list(srcd, save_dir, filter_p, target_fno, num_chunk, save_name):

    exec_list = find_all_files_with_name(srcd, name='final', exclude_suffix=True)

    tgt_list = []
    filter_set = set()
    if filter_p is not None:
        filter_set = set(load_exec_list(filter_p))
    for d in exec_list:
        if d in filter_set or osp.dirname(d) in filter_set:
            continue
        dname = osp.basename(osp.dirname(d))
        if target_fno > 0:
            fno = dname.split('-')[-1]
            if not fno.isdigit():
                print(f'{d} is not a valid path')
                continue
            fno = int(fno)
            if fno == target_fno:
                tgt_list.append(d)
        else:
            tgt_list.append(d)

    random.shuffle(tgt_list)
    print(f'num samples: {len(tgt_list)}')

    if save_dir is None:
        save_dir = srcd

    savep = osp.join(save_dir, f'{save_name}.txt')
    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(tgt_list))
        print(f'exec list saved to {savep}')

    if num_chunk > 0:
        world_size = num_chunk
        for ii in range(world_size):
            t = load_exec_list(tgt_list, ii, world_size=world_size)
            savep = osp.join(save_dir, f'{save_name}{ii}.txt')
            with open(savep, 'w', encoding='utf8') as f:
                f.write('\n'.join(t))
            print(f'exec list saved to {savep}')
            print(f'chunk {ii} num samples: {len(t)}')



@cli.command('further_extr')
@click.option('--exec_list')
@click.option('--rank_to_worldsize', default=None)
@click.option('--save_name', default=None)
def _further_extr(*args, **kwargs):
    further_extr(*args, **kwargs)

def further_extr(exec_list, rank_to_worldsize=None, save_name=None):
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)


    eye_mesh_dict = {
        '1-2-3-2-2+eyebgs-l': 'eyebgsl',
        '1-2-3-3-2+irides-l': 'iridesl',
        '1-2-3-1-2+eyelashs-l': 'eyelashsl',
        '1-2-3-2-1+eyebgs-r': 'eyebgsr',
        '1-2-3-3-1+irides-r': 'iridesr',
        '1-2-3-1-1+eyelashs-r': 'eyelashsr',
        '1-2-1-1+eyebrows-r': 'eyebrowr',
        '1-2-1-2+eyebrows-l': 'eyebrowl'
    }

    for p in tqdm(exec_list):
        try:
            fp = osp.join(p, 'face_parsing')
            parts_dict_exist = osp.exists(osp.join(fp, 'parts.json'))
            if parts_dict_exist:
                parts = json2dict(osp.join(fp, 'parts.json'))
            eye_parts = {}
            for k, n in eye_mesh_dict.items():
                imgp = osp.join(fp, k + '.png')
                if not osp.exists(imgp) or not parts_dict_exist:
                    eye_parts[n] = {'area': 0}
                    continue
                img = np.array(Image.open(osp.join(fp, k + '.png')))
                pd = parts[k]
                x, y, w, h = pd['x'], pd['y'], pd['w'], pd['h']
                mask = img[..., -1] > 15
                rect = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
                xyxy = [x, y, x + w, y + h]
                rect = [rect[0] + x, rect[1] + y, rect[2], rect[3]]
                rect[2] += rect[0]
                rect[3] += rect[1]
                eye_parts[n] = {
                    'img': img,
                    'xyxy': xyxy,
                    'mask': mask,
                    'rect': rect,
                    'area': np.sum(mask)
                }

            lmodel = Live2DScrapModel(p, pad_to_square=False, crop_to_final=False)

            lmodel.init_drawable_visible_map()
            lmodel.load_body_parsing()
            max_d = len(lmodel.drawables) + 1
            face_max_idx = -1
            face_min_idx = max_d
            neck_max_idx = -1
            neck_min_idx = max_d
            nose_max_idx = -1
            nose_min_idx = max_d
            mouth_max_idx = -1
            mouth_min_idx = max_d
            for d in lmodel.drawables:
                if d.body_part_tag == 'nose':
                    nose_max_idx = max(d.idx, nose_max_idx)
                    nose_min_idx = min(d.idx, nose_min_idx)
                elif d.body_part_tag == 'mouth':
                    mouth_max_idx = max(d.idx, mouth_max_idx)
                    mouth_min_idx = min(d.idx, mouth_min_idx)
            for d in lmodel.drawables:
                if d.body_part_tag == 'face':
                    tgt_max_idx = min(d.idx, nose_min_idx, mouth_min_idx)
                    face_max_idx = max(tgt_max_idx, face_max_idx)
                    face_min_idx = min(d.idx, face_min_idx)
            hair_split_idx = face_max_idx
            for d in lmodel.drawables:
                if d.body_part_tag is None or 'hair' not in d.body_part_tag:
                    continue
                if d.idx > hair_split_idx:
                    d.body_part_tag = 'front hair'
                else:
                    d.body_part_tag = 'back hair'

            eyel_xyxy = [lmodel.final.shape[1], lmodel.final.shape[0], 0, 0]
            eyer_xyxy = [lmodel.final.shape[1], lmodel.final.shape[0], 0, 0]
            for k in {'iridesl', 'eyebgsl', 'eyelashsl'}:
                if 'xyxy' in eye_parts[k]:
                    eyel_xyxy[0] = min(eyel_xyxy[0], eye_parts[k]['xyxy'][0])
                    eyel_xyxy[1] = min(eyel_xyxy[1], eye_parts[k]['xyxy'][1])
                    eyel_xyxy[2] = max(eyel_xyxy[2], eye_parts[k]['xyxy'][2])
                    eyel_xyxy[3] = max(eyel_xyxy[3], eye_parts[k]['xyxy'][3])
            for k in {'iridesr', 'eyebgsr', 'eyelashsr'}:
                if 'xyxy' in eye_parts[k]:
                    eyer_xyxy[0] = min(eyer_xyxy[0], eye_parts[k]['xyxy'][0])
                    eyer_xyxy[1] = min(eyer_xyxy[1], eye_parts[k]['xyxy'][1])
                    eyer_xyxy[2] = max(eyer_xyxy[2], eye_parts[k]['xyxy'][2])
                    eyer_xyxy[3] = max(eyer_xyxy[3], eye_parts[k]['xyxy'][3])

            for d in lmodel.drawables:
                if d.body_part_tag != 'eyes':
                    continue
                eye_tag = None
                score = 0.
                eye_scores = {}
                for ek, ed in eye_parts.items():
                    if ed['area'] == 0:
                        eye_scores[ek] = [None] * 4
                        continue
                    mask = ed['mask']
                    area, u_area, i_area = d.mask_union_intersection(mask, ed['xyxy'], final_vis_mask=True)
                    eye_scores[ek] = [area, u_area, i_area, ed['area']]
                irides_scores, bg_scores = None, None
                eyelash_scores = eyebrow_scores = None
                if eye_scores['iridesl'][0] is not None:
                    irides_scores = eye_scores['iridesl']
                    bg_scores = eye_scores['eyebgsl']
                elif eye_scores['iridesr'][0] is not None:
                    irides_scores = eye_scores['iridesr']
                    bg_scores = eye_scores['eyebgsr']
                if eye_scores['eyelashsr'][0] is not None:
                    eyelash_scores = eye_scores['eyelashsr']
                elif eye_scores['eyelashsl'][0] is not None:
                    eyelash_scores = eye_scores['eyelashsl']
                if eye_scores['eyebrowr'][0] is not None:
                    eyebrow_scores = eye_scores['eyebrowr']
                elif eye_scores['eyebrowl'][0] is not None:
                    eyebrow_scores = eye_scores['eyebrowl']
                iou_i = iou_b = iou_l = iou_br = -1
                scores = {'irides': 0, 'eyebg': 0, 'eyelash': 0, 'eyebrow': 0}
                if irides_scores is not None and bg_scores is not None and irides_scores[2] > 0 and bg_scores[2] > 0:
                    scores['irides'] = irides_scores[2] / irides_scores[1]
                    scores['eyebg'] = bg_scores[2] / bg_scores[1]
                if eyelash_scores is not None and eyelash_scores[2] > 0:
                    scores['eyelash'] = eyelash_scores[2] / eyelash_scores[1]
                if eyebrow_scores is not None and eyebrow_scores[2] > 0:
                    scores['eyebrow'] = eyebrow_scores[2] / eyebrow_scores[1]
                k = max(scores, key=scores.get)

                def rect_include(xyxy1, dict2):
                    if 'xyxy' not in dict2:
                        return False
                    xyxy2 = dict2['xyxy']
                    return xyxy1[0] > xyxy2[0] and xyxy1[1] > xyxy2[1] and xyxy1[2] < xyxy2[2] and xyxy1[3] < xyxy2[3]

                if scores[k] > 0:
                    d.body_part_tag = k
                else:
                    x1, y1, x2, y2 = d.xyxy
                    y = (y1 + y2) / 2
                    if y < eyel_xyxy[1] or y < eyer_xyxy[1]:
                        d.body_part_tag = 'eyebrow'
                    elif rect_include(d.xyxy, eye_scores['iridesl']) or rect_include(d.xyxy, eye_scores['iridesr']):
                        d.body_part_tag = 'irides'
                    elif rect_include(d.xyxy, eye_scores['eyebgsl']) or rect_include(d.xyxy, eye_scores['eyebgsr']):
                        d.body_part_tag = 'eyebg'
                    else:
                        d.body_part_tag = 'eyelash'

            if lmodel._body_parsing is not None:
                metadata = lmodel._body_parsing['metadata']
            else:
                metadata = {}
            lmodel.save_body_parsing(save_name=save_name, metadata=metadata)
        
            # hairf = lmodel.compose_bodypart_drawables('hairf')
            # hairb = lmodel.compose_bodypart_drawables('hairb')
            # irides = lmodel.compose_bodypart_drawables('irides')
            # eyebg = lmodel.compose_bodypart_drawables('eyebg')
            # eyelash = lmodel.compose_bodypart_drawables('eyelash')
            # eyebrow = lmodel.compose_bodypart_drawables('eyebrow')

            # save_tmp_img(
            #     imglist2imgrid([lmodel.final, hairf, hairb, irides, eyebg, eyelash, eyebrow], fix_size=512)
            # )
            # pass

        except Exception as e:
            raise
            print(f'failed to process {p}: {e}')
            continue

    

def propagate_invisible_parts(lmodel: Live2DScrapModel):
    voting_tree = {}
    for d in lmodel.drawables:
        if d.tag is None:
            continue
        parent = osp.dirname(d.did)
        if parent == '':
            parent = '_root'
        if parent not in voting_tree:
            voting_tree[parent] = {}
        if d.tag not in voting_tree[parent]:
            voting_tree[parent][d.tag] = 0
        voting_tree[parent][d.tag] += 1

    for d in lmodel.drawables:
        if d.tag is not None:
            continue
        parent = osp.dirname(d.did)
        target_tag = None
        while True:
            if parent == '':
                parent = '_root'
            if parent not in voting_tree:
                break
            if len(voting_tree[parent]) > 0:
                target_tag = max(voting_tree[parent], key=voting_tree[parent].get)
                break
            if parent == '_root':
                break
            parent = osp.dirname(parent)
        if target_tag is not None:
            voting_tree[parent][target_tag] += 1
            d.set_tag(target_tag)


def assign_tag_by_path(lmodel: Live2DScrapModel):
    did_contain_arms = False
    for d in lmodel.drawables:
        if d.did is None:
            continue
        if 'arm' in d.did.lower():
            did_contain_arms = True

    for d in lmodel.drawables:
        if d.did is None:
            continue
        did_lower = d.did.lower()
        if d.tag == 'objects':
            continue
        if d.tag is None:
            if 'hair' in did_lower:
                d.set_tag('hair')
            elif 'arm' in did_lower:
                d.set_tag('handwear')
            elif 'mouth' in did_lower:
                d.set_tag('mouth')
            elif 'body' in did_lower:
                if 'body2' in did_lower:
                    d.set_tag('bottomwear')
                # else:
                #     d.set_tag('topwear')
            elif 'face' in did_lower:
                d.set_tag('face')
            elif 'ear' in did_lower:
                d.set_tag('ears')
            elif 'eye' in did_lower:
                d.set_tag('eyes')
            elif 'leg' in did_lower:
                d.set_tag('legwear')
        elif d.tag == 'hair':
            if 'face' in did_lower:
                d.set_tag('face')
            elif 'arm' in did_lower:
                d.set_tag('handwear')
            elif 'body' in did_lower and 'hair' not in did_lower:
                d.set_tag('topwear')
        elif d.tag == 'handwear':
            if did_contain_arms:
                if 'body' in did_lower and 'arm' not in did_lower:
                    if 'body2' in did_lower:
                        d.set_tag('bottomwear')
                    # else:
                    #     d.set_tag('topwear')
            if 'hair' in did_lower:
                d.set_tag('hair')
        elif d.tag == 'topwear':
            if 'hair' in did_lower:
                d.set_tag('headwear')
            elif 'arm' in did_lower:
                d.set_tag('handwear')
        elif d.tag == 'bottomwear':
            if 'hair' in did_lower:
                d.set_tag('headwear')
        else:
            if 'arm' in did_lower:
                d.set_tag('handwear')
        if d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'ear' in did_lower:
            d.set_tag('ears')
        elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'neck' in did_lower:
            d.set_tag('neck')
        elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', None} and ('hand' in did_lower or 'arm' in did_lower):
            d.set_tag('handwear')
        elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', None} and 'eye' in did_lower:
            d.set_tag('eyes')
        elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', None} and 'mouth' in did_lower:
            d.set_tag('mouth')
        elif d.tag in {'legwear', 'topwear', 'bottomwear', 'hair', 'face', None} and 'nose' in did_lower:
            d.set_tag('nose')


@cli.command('label_l2d_wsamsegs')
@click.option('--exec_list')
@click.option('--save_dir', default='')
@click.option('--extr_more', is_flag=True, default=False, help='required if sam masks is 19 classes, further divide hair and eyes into sub parts')
@click.option('--rank_to_worldsize', default='', type=str)
def label_l2d_wsamsegs(exec_list, save_dir, extr_more, rank_to_worldsize):

    from live2d.scrap_model import Drawable, VALID_BODY_PARTS_V2
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything

    seed_everything(42)
    exec_listp = exec_list

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    mask_name = 'sam_masks.json'

    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:

            instance_mask, crop_xyxy, score = load_detected_character(p)
            # if instance_mask is None:
            #     print(f'skip {p}, no character instance detected')
            #     continue
            
            lmodel = Live2DScrapModel(p, crop_xyxy=crop_xyxy, pad_to_square=False)
            model_dir = lmodel.directory
            
            if lmodel._body_parsing is not None:
                metadata = lmodel._body_parsing['metadata']
            else:
                metadata = {}
            # feet_mask_valid = metadata['tag_valid']['footwear']

            masks_ann = json2dict(osp.join(model_dir, mask_name))
            sam_masks = [rle2mask(m, to_bool=True) for m in masks_ann]

            init_drawable_visible_map(lmodel.drawables)

            for tg in lmodel.drawables:
                if tg.final_visible_area < 1:
                    continue
                score_list = []
                for m in sam_masks:
                    area, u_area, i_area = tg.mask_union_intersection(m, final_vis_mask=True)
                    if i_area is None:
                        i_area = -1
                    score = i_area / tg.final_visible_area
                    score_list.append(score)
                best_match = np.argmax(np.array(score_list))
                best_match = VALID_BODY_PARTS_V2[best_match]
                tg.body_part_tag = best_match
                if tg.body_part_tag == 'legwear' and score_list[VALID_BODY_PARTS_V2.index('footwear') > 0.5]:
                    tg.body_part_tag = 'footwear'

            assign_tag_by_path(lmodel)
            propagate_invisible_parts(lmodel)
            lmodel.save_body_parsing(metadata=metadata, save_name='body_parsing')

        except Exception as e:
            raise
            print(f'Failed to process {p}: {e}')

    if extr_more:
        further_extr(exec_listp)


@cli.command('gradcam_heatmap')
@click.option('--image_file')
@click.option('--savep', default=None)
@click.option('--method', default='gradcam++')
@click.option('--model_type', default='eva')
@click.option('--gen_threshold', default=0.35)
@click.option('--eigen_smooth', is_flag=True, default=False)
@click.option('--aug_smooth', is_flag=True, default=False)
@click.option('--device', default='cuda')
def gradcam_heatmap(image_file, savep, method, model_type, gen_threshold, eigen_smooth, aug_smooth, device):

    from annotators.wdv3_tagger import apply_wdv3_tagger, get_tagger_and_transform
    from annotators.gradcam import apply_gradcam
    from pytorch_grad_cam.utils.image import show_cam_on_image

    if savep is None:
        os.makedirs('workspace', exist_ok=True)
        savep = osp.join('workspace', osp.basename(osp.dirname(image_file)) + '_' + model_type + '_' + method + '.png')

    img_input: Image.Image = Image.open(image_file)
    alpha = img_input.split()[-1]
    bbox = alpha.getbbox()

    # ensure image is RGB
    img_input = pil_ensure_rgb(img_input)
    img_input = img_input.crop(bbox)
    # pad to square with white background
    img_input, _ = pil_pad_square(img_input)
    img_input = img_input.resize((448, 448), resample=Image.Resampling.LANCZOS)

    caption, taglist, ratings, character, general = apply_wdv3_tagger(img_input, model_type=model_type, exclude_cls=exclude_cls, gen_threshold=gen_threshold)

    _, transform, labels = get_tagger_and_transform(model_type)
    inputs = transform(img_input).unsqueeze(0)
    inputs = inputs[:, [2, 1, 0]]

    imglist = []
    for k, v in tqdm(general.items()):
        grayscale_cam = apply_gradcam(inputs, v[1], method=method, model_type=model_type, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth, device=device)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(np.array(img_input)[..., ::-1] / 255., grayscale_cam)
        fontScale = 0.9
        cam_image = cv2.putText(cam_image, k, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        imglist.append(cam_image)
        # torch.cuda.empty_cache()

    rst = imglist2imgrid(imglist)
    Image.fromarray(rst[..., ::-1]).save(savep)
    print(f'result saved to {savep}')



@cli.command('infer_bizarre_tagger')
@click.option('--exec_list')
@click.option('--detected_instanec_only', default=False, is_flag=True)
@click.option('--rank_to_worldsize', default='', type=str)
def infer_bizarre_tagger(exec_list, detected_instanec_only, rank_to_worldsize):
    
    '''
    apply pos estimator: bizarre tagger
    '''

    from annotators.bizarre_tagger import apply_pos_estimator
    
    # model = LangSAM(sam_type="sam2.1_hiera_large")
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)


    for model_dir in tqdm(exec_list):
        try:
            if osp.isfile(model_dir):
                model_dir = osp.dirname(model_dir)
            crop_xyxy =  None
            if detected_instanec_only:
                instance_mask, crop_xyxy, score = load_detected_character(model_dir)
                if instance_mask is None:
                    print(f'skip {model_dir}, no character instance detected')
                    continue

            lmodel = Live2DScrapModel(model_dir, crop_xyxy=crop_xyxy, crop_to_final=True, pad_to_square=False)
            model_dir = lmodel.directory
            # ensure image is RGB
            img_input = pil_ensure_rgb(Image.fromarray(lmodel.final))
            kps, scores, bbox = apply_pos_estimator(img_input, mask=lmodel.final[..., -1].astype(np.float32) / 255.)

            save_rst = {'transform_stats': {'crop_xyxy': lmodel.final_bbox},  'pos': [k for k in kps], 'scores': scores}
            savep = osp.join(model_dir, 'bizarre_pos.json')
            dict2json(save_rst, savep)

        except Exception as e:
            #  raise e
             print(f'failed to process {model_dir}: {e}')


@cli.command('infer_langsam')
@click.option('--exec_list')
@click.option('--box_threshold', default=0.35, type=float)
@click.option('--text_threshold', default=0.25, type=float)
@click.option('--detected_instanec_only', default=False, is_flag=True)
@click.option('--rank_to_worldsize', default='', type=str)
@click.option('--skip_exists', default=False, is_flag=True)
def infer_langsam(exec_list, box_threshold, text_threshold, detected_instanec_only, rank_to_worldsize, skip_exists):
    import torch
    import gc

    from annotators.lang_sam import LangSAM
    
    model = LangSAM(sam_type="sam2.1_hiera_large")
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    prompts = ['face', 'hair', 'hand', 'feet', 'leg', 'dress', 'shirt', 'skirt', 'jacket', 'neck', 'outfit', 'shoes']
    prompt_list_head = ['mouth', 'nose', 'ears']
    # prompt_list_head = ['hairband', 'crown']

    # skip_exists = True

    for model_dir in tqdm(exec_list[0:]):
        try:
            if osp.isfile(model_dir):
                model_dir = osp.dirname(model_dir)
            crop_xyxy =  None
            if detected_instanec_only:
                instance_mask, crop_xyxy, score = load_detected_character(model_dir)
                if instance_mask is None:
                    print(f'skip {model_dir}, no character instance detected')
                    continue
            
            lmodel = Live2DScrapModel(model_dir, crop_xyxy=crop_xyxy, crop_to_final=True, pad_to_square=False)
            model_dir = lmodel.directory
            # ensure image is RGB
            img_input = pil_ensure_rgb(Image.fromarray(lmodel.final))

            
            savep = osp.join(model_dir, 'langsam_masks.json')
            if osp.exists(savep) and skip_exists:
                save_rst = json2dict(savep)
            else:
                save_rst = {'transform_stats': {'crop_xyxy': lmodel.final_bbox}, 'instances': {}}

            if skip_exists:
                prompt_list = [k for k in prompts if (k not in save_rst['instances'])]
            else:
                prompt_list = prompts
            if len(prompt_list) > 0:
                rst = model.predict_multi_prompts(img_input, prompt_list, box_threshold=box_threshold, text_threshold=text_threshold)
                for p, ins in zip(prompt_list, rst):
                    masks = [np.squeeze(m, 0) if m.ndim == 3 else m for m in ins['masks']]
                    masks = [mask2rle(m) for m in masks]
                    ins['boxes'] = [b for b in ins['boxes']] 
                    ins['masks'] = masks
                    save_rst['instances'][p] = ins

            if skip_exists:
                prompt_list = [k for k in prompt_list_head if (k not in save_rst['instances'])]
            else:
                prompt_list = prompt_list_head

            crop_head_for_head_prompt = True
            if len(prompt_list) > 0:
                
                head_crop = head_pad = None
                head_input = img_input
                h, w = img_input.height, img_input.width
                if crop_head_for_head_prompt and lmodel.face_detected():
                    facedet = lmodel.facedet[0]
                    x1, y1, x2, y2 = facedet['bbox'][:4]
                    p = int(round(max(x2 - x1, y2 - y1) * 1.0))
                    if p > 0:
                        head_crop = [max(x1 - p, 0), max(y1 - p, 0), min(x2 + p, w), min(y2 + p, h)]
                        hw, hh = head_crop[2] - head_crop[0], head_crop[3] - head_crop[1]
                        head_pad = [head_crop[0], head_crop[1], w - head_crop[2], h - head_crop[3]]
                        if np.all(np.array(head_pad) == 0) or hw <= 0 or hh <= 0:
                            head_pad = None
                        else:
                            head_input = head_input.crop(head_crop)
                        
                rst = model.predict_multi_prompts(head_input, prompt_list, box_threshold=box_threshold, text_threshold=text_threshold)

                for p, ins in zip(prompt_list, rst):
                    masks = [np.squeeze(m, 0) if m.ndim == 3 else m for m in ins['masks']]
                    if head_pad is not None:
                        masks = [cv2.copyMakeBorder(m.astype(np.uint8), head_pad[1], head_pad[3], head_pad[0], head_pad[2], value=0, borderType=cv2.BORDER_CONSTANT) for m in masks]
                    masks = [mask2rle(m) for m in masks]
                    ins['boxes'] = [b for b in ins['boxes']] 
                    ins['masks'] = masks
                    save_rst['instances'][p] = ins

                # from utils.visualize import visualize_segs_with_labels
                # from utils.cv import rle2mask
                # masks = []
                # for p in prompt_list:
                #     msk = [rle2mask(m) for m in save_rst['instances'][p]['masks']]
                #     if len(msk) > 0:
                #         msk = np.logical_or.reduce(np.stack(msk, 0), axis=0)
                #     else:
                #         msk = np.zeros_like(lmodel.final[..., 0])
                #     masks.append(msk)
                # t = json2dict(osp.join(model_dir, 'general_tags.json'))
                # print(t.keys())
                # print(has_animal_ear(t.keys()))
                # save_tmp_img(visualize_segs_with_labels(masks, lmodel.final, prompt_list))
                # pass
            
            savep = osp.join(model_dir, 'langsam_masks.json')
            dict2json(save_rst, savep)

            # pad to square with white background   
        except Exception as e:
            # raise
            print(f'failed to process {model_dir}: {e}')
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

@cli.command('parse_live2d')
@click.option('--exec_list')
@click.option('--method', default='gradcam')
@click.option('--model_type', default='eva')
@click.option('--gen_threshold', default=0.3)
@click.option('--eigen_smooth', is_flag=True, default=False)
@click.option('--aug_smooth', is_flag=True, default=False)
@click.option('--save_gradcam_heatmap', is_flag=True, default=False)
@click.option('--device', default='cuda')
@click.option('--tag_only', default=False, is_flag=True)
@click.option('--detected_instanec_only', default=False, is_flag=True)
@click.option('--rank_to_worldsize', default='', type=str)
def parse_live2d(exec_list, method, model_type, gen_threshold, eigen_smooth, aug_smooth, save_gradcam_heatmap, device, tag_only, detected_instanec_only, rank_to_worldsize):

    from annotators.wdv3_tagger import apply_wdv3_tagger, get_tagger_and_transform
    from annotators.gradcam import apply_gradcam
    from pytorch_grad_cam.utils.image import show_cam_on_image

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    for model_dir in tqdm(exec_list):
        try:
            crop_xyxy =  None
            if detected_instanec_only:
                instance_mask, crop_xyxy, score = load_detected_character(model_dir)
                if instance_mask is None:
                    print(f'skip {model_dir}, no character instance detected')
                    continue
            
            model = Live2DScrapModel(model_dir, target_frame_size=448, crop_to_final=True, pad_to_square=True, crop_xyxy=crop_xyxy, pad_drawable_img=False)
            model_dir = model.directory
            # ensure image is RGB
            img_input = pil_ensure_rgb(Image.fromarray(model.final))
            # pad to square with white background
            caption, taglist, ratings, character, general = apply_wdv3_tagger(img_input, model_type=model_type, exclude_cls=exclude_cls, gen_threshold=gen_threshold)
            dict2json(general, osp.join(model_dir, 'general_tags.json'))
            if tag_only:
                continue
            model.init_drawable_visible_map()
            _, transform, labels = get_tagger_and_transform(model_type)
            inputs = transform(img_input).unsqueeze(0)
            inputs = inputs[:, [2, 1, 0]]

            gradcam_heatmap_vis = []
            for cls_name, v in general.items():
                cls_score, cls_idx = v[0], v[1]
                score_map = apply_gradcam(inputs, cls_idx, method=method, model_type=model_type, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth, device=device)
                model.update_tag_stats(score_map[0], cls_idx, cls_name, filter_scoremap=True)

                if save_gradcam_heatmap:
                    cam_image = show_cam_on_image(np.array(img_input)[..., ::-1] / 255., score_map[0])
                    fontScale = 0.9
                    cam_image = cv2.putText(cam_image, cls_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                    gradcam_heatmap_vis.append(cam_image)

            if save_gradcam_heatmap:
                gradcam_heatmap_vis = imglist2imgrid(gradcam_heatmap_vis, cols=6)
                savep = osp.join(model_dir, 'heatmap_' + model_type + '_' + method + '.png')
                Image.fromarray(gradcam_heatmap_vis[..., ::-1]).save(savep)

            model.save_tag_stats()
        except Exception as e:
            print(f'failed to process {model_dir}: {e}')
        pass

        # # assign drawable to the tag with maximum
        # avgscore_lst = []
        # for tag, tag_info in model.tag_stats.items():
        #     avgscore_map = np.zeros_like(model.final[..., 0]).astype(np.float32)
        #     for drawable in model.drawables:
        #         if drawable.final_visible_area < 1:
        #             continue
        #         x1, y1, x2, y2 = drawable.xyxy
        #         avgscore_map[y1: y2, x1: x2] += drawable.final_visible_mask.astype(np.float32) * drawable.tag_stats[tag]['avg_score']
        #     avgscore_lst.append(avgscore_map)

        # avgscore_lst = np.stack(avgscore_lst).clip(0, 1)
        # concept_labels = list(model.tag_stats.keys())
        # vis = show_factorization_on_image(model.final[..., :3] / 255., avgscore_lst, concept_labels=concept_labels, image_weight=0.1, visible_mask=model.final_visible_mask[..., None])
        # Image.fromarray(vis).save(osp.join(model_dir, 'segmentation_' + model_type + '_' + method + '.png'))


@cli.command('dump_body_tags')
@click.option('--src_dir', default='workspace/tags_raw/bodyparts')
@click.option('--savep', default='workspace/tagcluster_bodypart.json')
def dump_body_tags(src_dir, savep):
    from utils.io_utils import json2dict, dict2json

    spliters = [',', '|']
    tag_set_cleaned = {}

    sets_duplicated = {}

    for d in os.listdir(src_dir):
        p = osp.join(src_dir, d)
        with open(p, 'r', encoding='utf8') as f:
            lines = f.read().split('\n')

        lines_lst = []
        for l in lines:
            l = l.strip().lower()
            if l.startswith('#'):
                continue
            for s in spliters:
                l = l.split(s)[0].strip()
            if len(l) > 0:
                l = '_'.join(l.split(' '))
                lines_lst.append(l)
        tag_set_cleaned[d] = lines_lst


    dict2json(tag_set_cleaned, savep)


@cli.command('facedet')
@click.option('--exec_list')
@click.option('--twopass', is_flag=True, default=False)
@click.option('--rank_to_worldsize', default='', type=str)
@click.option('--skip_exists', default=False, is_flag=True)
def facedet(exec_list, twopass, rank_to_worldsize, skip_exists):

    from annotators import anime_face_detector

    if exec_list.endswith('.json') or exec_list.endswith('.json.gz'):
        exec_list = json2dict(exec_list)
        exec_list = list(exec_list.keys())
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    device = 'cuda'  #@param ['cuda:0', 'cpu']
    model = 'yolov3'  #@param ['yolov3', 'faster-rcnn']
    detector = anime_face_detector.create_detector(model, device=device)

    if skip_exists:
        new_exec_list = []
        for srcp in exec_list:
            if osp.isfile(srcp):
                srcp = osp.dirname(srcp)
            if osp.exists(osp.join(srcp, 'facedet.json')):
                print(f'skip {srcp} due to result exists')
                continue
            new_exec_list.append(srcp)
        exec_list = new_exec_list

    for srcp in tqdm(exec_list):
        try:
            if osp.isfile(srcp):
                srcp = osp.dirname(srcp)
            savep = osp.join(srcp, 'facedet.json')

            lmodel = Live2DScrapModel(srcp, crop_to_final=True, pad_to_square=False)
            image = Image.fromarray(lmodel.final)
            if twopass:
                lmodel.init_drawable_visible_map()

            image = pil_ensure_rgb(image)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            preds = detector(image)
            if len(preds) > 0:
                pred = max(preds, key=lambda x: x['bbox'][-1])

                bbox = pred['bbox']
                keypoints = pred['keypoints']

                if twopass:
                    bbox_input = bbox.copy()
                    # bbox_input[..., :4] = lmodel.processor.scale_coordinates(bbox_input[..., :4].reshape((2, 2))).flatten()
                    x1, y1, x2, y2, _ = np.round(bbox_input).astype(np.int32)
                    # vis_face_det = visualize_facedet_output(model.final, [facedet])[y1: y2, x1: x2, :3]
                    # vis_face_det = np.concatenate([vis_face_det, np.full_like(vis_face_det[..., [0]], fill_value=255)], axis=2)

                    xyxy = [x1, y1, x2, y2]
                    valid_drawables = []
                    for drawable in lmodel.drawables:
                        if drawable.final_visible_area < 1:
                            continue
                        bbox_i, vis_mask = drawable.get_vis_mask(xyxy, final_vis_mask=True)
                        if bbox_i is None or vis_mask.sum() / drawable.final_visible_area < 0.8:
                            continue
                        valid_drawables.append(drawable)
                    face_crop = compose_from_drawables(valid_drawables, xyxy=xyxy)
                    facedet2 = detector(face_crop, boxes=[np.array([0, 0, x2-x1, y2-y1, 1])])
                    keypoints2 = facedet2[0]['keypoints']
                    px1 = x1 + lmodel.final_bbox[0]
                    py1 = y1 + lmodel.final_bbox[1]
                    keypoints2[:, 0] += px1
                    keypoints2[:, 1] += py1
                    keypoints[LEFT_EYEBROW] = keypoints2[LEFT_EYEBROW]
                    keypoints[RIGHT_EYEBROW] = keypoints2[RIGHT_EYEBROW]
                    # Image.fromarray(face_crop).save('local_tst.png')
                    pass

                bbox[-1] = np.round(bbox[-1] * 100)
                bbox[:-1] = np.round(bbox[:-1])
                pred['bbox'] = bbox.astype(np.int32)
                if lmodel.final_bbox is not None:
                    pred['bbox'][[0, 2]] += lmodel.final_bbox[0]
                    pred['bbox'][[1, 3]] += lmodel.final_bbox[1]

                    # print(lmodel.final_bbox)
                    # save_tmp_img(imread(osp.join(srcp, 'final.jxl'))[pred['bbox'][1]: pred['bbox'][3], pred['bbox'][0]: pred['bbox'][2]])
                    # break
                keypoints[:, 2] = np.round(keypoints[:, 2] * 100)
                keypoints[:, :2] = np.round(keypoints[:, :2])
                pred['keypoints'] = [k for k in keypoints.astype(np.int32)]
            
            else:
                pass
            dict2json(preds, savep)
        except Exception as e:
            print(f'failed to process {srcp}: {e}')



def hflip_aug_mask(mask: np.ndarray, x: int, aug=True):
    '''
    mask: (h, w) or (c, h, w)
    '''
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = np.logical_or.reduce(mask, axis=0)
    h, w = mask.shape[:2]
    mid = w // 2
    if x <= mid:
        x1 = 0
        x2 = x * 2
    else:
        x2 = w
        x1 = w - (w - x) * 2
    mask_or = mask[:, x1: x2]
    if aug:
        mask[:, x1: x2] = np.bitwise_or(mask_or, mask_or[:, ::-1])
    mask_l = mask.copy()
    mask_l[:, mid:] = 0
    mask[:, :mid] = 0
    # imglist2imgrid([mask_l.astype(np.uint8) * 255, mask.astype(np.uint8) * 255], output_type='pil').save('local_tst.png')
    return mask_l, mask


def part_morph_transform(masks, target_cls, ksize=1, op='dilate'):
    mask = masks[target_cls].astype(np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
    if op == 'dilate':
        mask = cv2.dilate(mask, element)
    else:
        mask = cv2.erode(mask, element)
    masks[target_cls] = mask.astype(bool)


def split_lr_part(lmodel: Live2DScrapModel, target_ids: list):
    from sklearn.cluster import MiniBatchKMeans, KMeans
    eye_xs = []
    eye_drawables = []
    for d in lmodel.drawables:
        if d.face_part_id in target_ids:
            dx, dy, dw, dh = cv2.boundingRect(cv2.findNonZero(d.final_visible_mask.astype(np.uint8)))
            dx += d.x
            dy += d.y
            eye_xs.append(dx + dw / 2)
            eye_drawables.append(d)
    if len(eye_drawables) < 2:
        return False
    eye_xs = np.array(eye_xs)
    eye_xs_mean = np.mean(eye_xs) + 1e-6
    eye_xs = eye_xs[:, None] / eye_xs_mean - 1
    rst = KMeans(2, max_iter=50).fit(eye_xs)
    labels = rst.predict(eye_xs)
    if rst.cluster_centers_[0, 0] > rst.cluster_centers_[1, 0]:
        labels = 1 - labels
    for d, l in zip(eye_drawables, labels):
        d.face_part_id = target_ids[l]
    return True


def split_lr_mask(masks: np.ndarray, split_channels):
    ms = masks[split_channels]
    ms = np.logical_or.reduce(ms, axis=0)
    xs = np.where(ms > 0)

    pass
        

def find_brow(lmodel: Live2DScrapModel, brow_id, face_xyxy=None):
    btop = 100000
    eye_id = brow_id + 2
    eye_mask = lmodel.compose_face_drawables(eye_id, mask_only=True, final_visible_mask=True, xyxy=face_xyxy)
    ex, ey, ew, eh = cv2.boundingRect(cv2.findNonZero(eye_mask.astype(np.uint8)))
    tgt_brow = None
    for d in lmodel.drawables:
        if d.face_part_id != eye_id:
            continue
        dx, dy, dw, dh = d.get_bbox(xyxy=face_xyxy)
        if dy < btop and dw / ew > 0.5:
            tgt_brow = d
            btop = dy
    if tgt_brow is not None:
        tgt_brow.face_part_id = brow_id
        return True
    return False


@cli.command('facedet_sam')
@click.option('--exec_list')
@click.option('--ckpt')
@click.option('--mask_decoder', default='mlp_variant')
@click.option('--class_num', default=19)
@click.option('--save_segs', default=False, is_flag=True)
@click.option('--save_preview', default=False, is_flag=True)
@click.option('--rank_to_worldsize', default='', type=str)
@click.option('--skip_exists', default=False, is_flag=True)
def facedet_sam(exec_list, ckpt, mask_decoder, class_num, save_segs, save_preview, rank_to_worldsize, skip_exists):

    from modules.semanticsam import SemanticSam, Sam
    from utils.torch_utils import init_model_from_pretrained
    from utils.cv import batch_save_masks
    import torch
    
    if exec_list.endswith('.json') or exec_list.endswith('.json.gz'):
        exec_list = json2dict(exec_list)
        exec_list = list(exec_list.keys())

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    device = 'cuda'  #@param ['cuda:0', 'cpu']

    # if model is None:
    sam: SemanticSam = init_model_from_pretrained(
        pretrained_model_name_or_path=ckpt,
        module_cls=SemanticSam,
        model_args={'mask_decoder': mask_decoder, 'class_num': class_num},
        device=device
    ).eval()

    head_pad_ratio = 0.4

    if skip_exists:
        new_exec_list = []
        for srcp in exec_list:
            if osp.isfile(srcp):
                srcp = osp.dirname(srcp)
            if osp.exists(osp.join(srcp, 'face_parsing.json')):
                print(f'skip {srcp} due to result exists')
                continue
            new_exec_list.append(srcp)
        exec_list = new_exec_list

    for sidx, srcp in enumerate(tqdm(exec_list[0:])):
        # 5 12
        try:
            if osp.isfile(srcp):
                srcp = osp.dirname(srcp)

            lmodel_dir = srcp
            lmodel = Live2DScrapModel(lmodel_dir, crop_to_final=True, pad_to_square=False)
            if not lmodel.face_detected():
                print(f'skip {srcp} due to no face detected')
                continue
            lmodel.init_drawable_visible_map()

            fh, fw = lmodel.final.shape[:2]

            facedet = lmodel.facedet[0]
            x1, y1, x2, y2 = facedet['bbox'][:4]
            # save_tmp_img(lmodel.final[y1: y2, x1: x2])

            head_pad = 0
            if head_pad_ratio != 0:
                head_pad =  head_pad_ratio * (y2 - y1)
                head_pad = int(round(head_pad))
                facedet['bbox'][:2] -= head_pad
                facedet['bbox'][2:4] += head_pad
                facedet['bbox'] = np.clip(facedet['bbox'], 0, min(fh, fw))
                x1, y1, x2, y2, _ = facedet['bbox']
                # x1 -= head_pad ; y1 -= head_pad ; x2 += head_pad ; y2 += head_pad

            face_xyxy = [x1, y1, x2, y2]
            image = lmodel.final
            face_image = image[y1: y2, x1: x2, :3]
            ch, cw = face_image.shape[:2]

            save_tmp_img(face_image)
            with torch.inference_mode():
                preds = sam.inference(face_image)[0]
                masks_np = (preds > 0).to(device='cpu', dtype=torch.bool).numpy()

            if save_segs:
                batch_save_masks(masks_np, osp.join(lmodel_dir, 'faceseg.json'), compress='gzip')
                batch_save_masks(masks_np[[10, 11]], osp.join(lmodel_dir, 'faceseg_nosemouth.json'), compress='gzip')
            
            # save_tmp_img(visualize_segs(masks_np[[1]], src_img=np.array(face_image), image_weight=0.3))

            part_morph_transform(masks_np, 11, ksize=2)
            part_morph_transform(masks_np, 4, ksize=3)
            part_morph_transform(masks_np, 5, ksize=3)
            part_morph_transform(masks_np, 7, ksize=2)
            part_morph_transform(masks_np, 8, ksize=2)
            
            neck_detected = False
            seg_areas = reduce(masks_np, 'b h w -> b', 'sum') + 1e-6
            for drawable in lmodel.drawables:
                if drawable.final_visible_area < 1:
                    continue
                area, u_area, i_area = drawable.mask_union_intersection(masks_np, face_xyxy, final_vis_mask=True)
                if u_area is None or area == 0 or np.all(i_area[1:] == 0):
                    continue
                u_area += 1e-6
                drawable.face_part_stats = {
                    'union': u_area, 'intersection': i_area, 'iou': i_area / u_area, 'ioa': i_area / area, 'area': area, 'ios': i_area / seg_areas
                }
                drawable.face_part_id = np.argmax(drawable.face_part_stats['ioa'][1:]) + 1
                if drawable.face_part_id == 14:
                    neck_detected = True

            base_face_mask = lmodel.compose_face_drawables(1, mask_only=True, xyxy=face_xyxy)
            bx, by, bw, bh = cv2.boundingRect(cv2.findNonZero(base_face_mask.astype(np.uint8)))
            by2 = by + bh
            bx2 = bw + bx

            base_face_mask_vis = cv2.cvtColor(base_face_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
            base_face_mask_vis = cv2.rectangle(base_face_mask_vis, (bx, by), (bx2, by2), color=(0, 255, 0), thickness=4)
            
            eyew = cv2.boundingRect(cv2.findNonZero(
                lmodel.compose_face_drawables([4, 5], mask_only=True, xyxy=face_xyxy).astype(np.uint8)
            ))[2]
            eye_detected = eyew > 1
            if eyew / bw > 0.5:
                split_lr_part(lmodel, (4, 5))
            leye_mask = lmodel.compose_face_drawables(4, mask_only=True, xyxy=face_xyxy)
            reye_mask = lmodel.compose_face_drawables(5, mask_only=True, xyxy=face_xyxy)

            leye_x, leye_y, leye_w, leye_h = cv2.boundingRect(cv2.findNonZero(leye_mask.astype(np.uint8)))
            reye_x, reye_y, reye_w, reye_h = cv2.boundingRect(cv2.findNonZero(reye_mask.astype(np.uint8)))
            eyel, eyer = min(leye_x, reye_x), max(leye_x + leye_w, reye_x + reye_w)
            eyew = eyer - eyel

            brow_potentials = []

            leye_detected, reye_detected = lmodel.face_part_detected([4, 5])
            beye_detected = leye_detected and reye_detected

            # re-assign eye lids & brows
            base_face_draworder = 10000
            for d_id, drawable in enumerate(lmodel.drawables):
                if drawable.area < 1:
                    continue
                if drawable.face_part_id == 14:
                    if drawable.face_part_stats['ioa'][16] + 0.1 > drawable.face_part_stats['ioa'][14]:
                        drawable.face_part_id = 16
                if not neck_detected and drawable.face_part_id == 16 and drawable.face_part_stats['ioa'][14] > 0.15:
                    drawable.face_part_id = 14

                dx, dy, dw, dh = drawable.get_bbox(xyxy=face_xyxy)
                dx2 = dx + dw
                dy2 = dy + dh
                if drawable.face_part_id == 16 and dy < by + bh / 2:
                    if drawable.face_part_stats['ioa'][17] > 0.15:
                        drawable.face_part_id = 17

                if not drawable.face_part_id in {None, 1, 17, 4, 5}:
                    continue

                # check if hair drawable is actually background
                if drawable.face_part_id == 17:
                    if drawable.face_part_stats['ioa'][0] > 0.7 and drawable.face_part_stats['ioa'][17] < 0.3:
                        drawable.face_part_id = None


                if drawable.face_part_id == 1 and dw / bw > 0.7 and dh > bw > 0.7:
                    if drawable.draw_order < base_face_draworder:
                        base_face_draworder = drawable.draw_order

                if not (dx > bx and dx2 < bx2 and dy > by and dy2 < by2):
                    continue
                
                if dy > max(leye_y + leye_h, reye_y + reye_h):
                    continue

                if drawable.face_part_id == 17 and drawable.draw_order >= base_face_draworder:
                    if dw / bw > 0.4 or dh / bh > 0.2:
                        continue

                # re-assign glass
                if drawable.face_part_id in {4, 5} and beye_detected:
                    if eye_detected and (dw / eyew > 0.6 or drawable.face_part_stats['ioa'][6] > 0.4):
                        drawable.face_part_id = 6
                        continue
                if dw > dh:
                    brow_potentials.append(drawable)

            facedrawable_wo_prehair = []
            for drawable in lmodel.drawables:
                # skip face-covering hairs
                if drawable.face_part_id == 17 and drawable.draw_order >= base_face_draworder:
                    continue
                # skip glass, neck, hat, cloth
                if drawable.face_part_id in {6, 14, 16, 18}:
                    continue

                dx, dy, dw, dh = drawable.get_bbox(xyxy=face_xyxy)
                dx2 = dx + dw ; dy2 = dy + dh
                ix = min(dx2, bx2) - max(dx, bx) 
                iy = min(dy2, by2) - max(dy, by)
                if dw > 0 and dh > 0 and ix / dw > 0.3 and iy / dh > 0.8:
                    facedrawable_wo_prehair.append(drawable)
                
            for drawable in lmodel.drawables:
                if drawable.face_part_id is None:
                    continue
                if drawable.face_part_id == 1:
                    # re-assgin mouth tags
                    if drawable.face_part_stats['ioa'][11] > 0.3:
                        drawable.face_part_id = 11
                

            base_face_drawables: list[Drawable] = None
            base_face_drawables = set(facedrawable_wo_prehair + brow_potentials)
            base_face_drawables = list(base_face_drawables)
            base_face_drawables.sort(key=lambda x: x.draw_order)
            # reinit for those covered by hairs 
            init_drawable_visible_map(base_face_drawables)
            # pil_ensure_rgb(compose_from_drawables(base_face_drawables[10:12], xyxy=face_xyxy, output_type='pil')).save('local_tst.png')

            base_face = compose_from_drawables(base_face_drawables, xyxy=face_xyxy)
            base_face = np.array(pil_ensure_rgb(Image.fromarray(base_face)))
            with torch.inference_mode():
                preds = sam.inference(base_face)[0]
                masks_np2 = (preds > 0).to(device='cpu', dtype=torch.bool).numpy()

            if save_segs:
                batch_save_masks(masks_np2, osp.join(lmodel_dir, 'faceseg2.json'), compress='gzip')

            masks_np2[[1, 10, 11]] = masks_np[[1, 10, 11]]
            masks_np2[[2, 3, 7, 8]] = np.bitwise_or(masks_np2[[2, 3, 7, 8]], masks_np[[2, 3, 7, 8]])
            if beye_detected:
                masks_np2[[4, 5]] = masks_np[[4, 5]]

            seg_areas = reduce(masks_np2, 'b h w -> b', 'sum') + 1e-6
            hair_mask = lmodel.compose_face_drawables([17], mask_only=True, xyxy=face_xyxy)
            for didx, drawable in enumerate(base_face_drawables):

                if didx in [10, 11]:
                    continue
                if drawable.final_visible_area < 1:
                    continue

                area, u_area, i_area = drawable.mask_union_intersection(masks_np2, face_xyxy, final_vis_mask=True)
                if u_area is None or area == 0 or np.all(i_area[1:] == 0):
                    continue
                u_area += 1e-6
                i_area[17] = 0
                if np.all(i_area[1:] == 0):
                    continue
                ori_face_part_stats = drawable.face_part_stats
                drawable.face_part_stats = {
                    'union': u_area, 'intersection': i_area, 'iou': i_area / u_area, 'ioa': i_area / area, 'area': area, 'ios': i_area / seg_areas
                }
                if len(ori_face_part_stats) > 0:
                    drawable.face_part_stats['ioa'][17] = ori_face_part_stats['ioa'][17]
                    drawable.face_part_stats['ioa'][1] = ori_face_part_stats['ioa'][1]
                face_part_id = np.argmax(drawable.face_part_stats['ioa'][1:]) + 1
                face_part_id = int(face_part_id)

                # fix brows & ears attached to hair
                if face_part_id == 17:
                    if drawable.face_part_stats['ioa'][2] > 0.5:
                        face_part_id = 2
                    elif drawable.face_part_stats['ioa'][3] > 0.5:
                        face_part_id = 3
                    elif drawable.face_part_stats['ioa'][7] > 0.5:
                        face_part_id = 7
                    elif drawable.face_part_stats['ioa'][8] > 0.5:
                        face_part_id = 8
                
                # fix brows & ears attached to face
                if face_part_id == 1:
                    if drawable.face_part_stats['ioa'][2] > 0.3:
                        face_part_id = 2
                    if drawable.face_part_stats['ioa'][3] > 0.3:
                        face_part_id = 3

                # assign drawables not classified as hair in the first pass
                if drawable.face_part_id != 17:
                    drawable.face_part_id = face_part_id
                # drawable was classiified as hair in the first pass and classified as ear now
                elif face_part_id in {7, 8} and drawable.draw_order <= base_face_draworder:
                    drawable.face_part_id = face_part_id
                elif face_part_id in {2, 3}:
                    drawable.face_part_id = face_part_id

                if drawable.face_part_id in {1, 17}:
                    dx, dy, dw, dh = drawable.get_bbox(xyxy=face_xyxy)
                    dx2 = dx + dw
                    dy2 = dy + dh
                    if not (dx > bx and dx2 < bx2 and dy > by and dy2 < by2):
                        continue
                    if dy > max(leye_y + leye_h, reye_y + reye_h):
                        continue
                    # if (dx - bx_mid) * (dx2 - bx_mid) < 0:
                    #     continue
                    # eye lids
                    if dw < max(leye_w, reye_w) * 1.1 and dh < max(leye_h / 2, reye_h / 2):
                        if drawable.face_part_stats['ioa'][4] > 0.4 or np.any(drawable.bitwise_and(masks_np[4], face_xyxy)):
                            drawable.face_part_id = 4
                        elif drawable.face_part_stats['ioa'][5] > 0.4 or np.any(drawable.bitwise_and(masks_np[5], face_xyxy)):
                            drawable.face_part_id = 5


                if drawable.face_part_id == 11 and drawable.get_bbox(xyxy=face_xyxy)[3] / bh > 0.4:
                    if np.any(drawable.bitwise_and(hair_mask, face_xyxy)):
                        drawable.face_part_id = 17
                    
            if eyew / bw > 0.5:
                split_lr_part(lmodel, (4, 5))
            lbrow_detected, rbrow_detected = lmodel.brow_detected()
            if lbrow_detected ^ rbrow_detected:
                brow_mask = lmodel.compose_face_drawables([2, 3], mask_only=True, xyxy=face_xyxy)
                brx, bry, brw, brh = cv2.boundingRect(cv2.findNonZero(brow_mask.astype(np.uint8)))
                if eye_detected and brw / eyew > 0.6:
                    split_lr_part(lmodel, (2, 3))
                    lbrow_detected = rbrow_detected = True
            if not lbrow_detected:
                lbrow_detected = find_brow(lmodel, 2)
            if not rbrow_detected:
                rbrow_detected = find_brow(lmodel, 3)

            if lbrow_detected and rbrow_detected:
                brow_mask = lmodel.compose_face_drawables([2, 3], mask_only=True, xyxy=face_xyxy)
                brx, bry, brw, brh = cv2.boundingRect(cv2.findNonZero(brow_mask.astype(np.uint8)))
                if eye_detected and brw / eyew > 0.6:
                    split_lr_part(lmodel, (2, 3))
                # lmodel.compose_face_drawables([2], xyxy=face_xyxy, output_type='pil', mask_only=True).save('local_tst.png')
                # lmodel.compose_face_drawables([3], xyxy=face_xyxy, output_type='pil', mask_only=True).save('local_tst.png')
                pass

            lear_detected, rear_detected = lmodel.face_part_detected([7, 8])
            if lear_detected ^ rear_detected:
                ear_mask = lmodel.compose_face_drawables([7, 8], mask_only=True, xyxy=face_xyxy)
                brx, bry, brw, brh = cv2.boundingRect(cv2.findNonZero(ear_mask.astype(np.uint8)))
                if brw / bw > 0.75:
                    split_lr_part(lmodel, (7, 8))
            elif lear_detected:
                ear_mask = lmodel.compose_face_drawables([7, 8], mask_only=True, xyxy=face_xyxy)
                brx, bry, brw, brh = cv2.boundingRect(cv2.findNonZero(ear_mask.astype(np.uint8)))
                if brw / bw > 0.75:
                    split_lr_part(lmodel, (7, 8))

            neck_base_mask = None
            face_mask = lmodel.compose_face_drawables(1, mask_only=True, final_visible_mask=True, xyxy=face_xyxy)
            base_neck_ids = set()
            for d in lmodel.drawables:
                if d.face_part_id != 14:
                    continue
                d_mask = d.get_full_mask(final_visible_mask=True, xyxy=face_xyxy)
                if np.any(np.bitwise_and(face_mask, d_mask)):
                    if neck_base_mask is None:
                        neck_base_mask = d_mask
                    else:
                        neck_base_mask = np.bitwise_or(neck_base_mask, d_mask)
                    base_neck_ids.add(d.draw_order)
                    continue
            if neck_base_mask is not None:
                # save_tmp_img(neck_base_mask, mask2img=True)
                for d in lmodel.drawables:
                    if d.face_part_id != 14 or d.draw_order in base_neck_ids:
                        continue
                    area, u_area, i_area = d.mask_union_intersection(neck_base_mask[None], face_xyxy, final_vis_mask=True)
                    if i_area[0] / (d.final_visible_area + 1e-6) < 0.95:
                        d.face_part_id = None

            # fix hat assigned as cloth
            for d in lmodel.drawables:
                if d.face_part_id == 16:
                    msk = d.get_full_mask(True, face_xyxy)
                    if np.any(msk):
                        ymean = np.mean(np.where(msk)[0])
                        if ymean < 1 / 3 * ch:
                            d.face_part_id = 18

            nose_detected = lmodel.face_part_detected(10)
            mouth_detected = lmodel.face_part_detected(11)
            # if not nose_detected:
            #     for d in lmodel.drawables:
            #         if d.face_part_id != 1:
            #             continue
            #         if d.face_part_stats['ioa'][10] > 0.4:
            #             d.face_part_id = 10
            #             nose_detected = True

            if not nose_detected:
                d = lmodel.maxios_mindrawable(10, 0.7, ioa_thr=0.3)
                if d is not None:
                    d.face_part_id = 10
                    # save_tmp_img(compose_from_drawables([d], xyxy=face_xyxy))
                    pass

            if save_preview:
                vis_face_det = visualize_facedet_output(lmodel.final, [facedet])[y1: y2, x1: x2, :3]
                rst_preds = visualize_segs(masks_np, src_img=np.array(face_image), image_weight=0.3)
                rst_preds2 = visualize_segs(masks_np2, src_img=np.array(base_face), image_weight=0.3)
                vis_list = [face_image, rst_preds, vis_face_det, base_face, rst_preds2]
                for flabel, flist in VALID_FACE_GROUPS.items():
                    rst = lmodel.compose_face_drawables(face_part_ids=flist, xyxy=face_xyxy)
                    rst = Image.fromarray(rst)
                    pil_draw_text(rst, flabel, point=(0, 0), stroke_width=2, font_size=int(bw / 5))
                    rst = pil_ensure_rgb(rst)
                    vis_list.append(rst)
                # save_tmp_img(masks_np[11], mask2img=True)
                imglist2imgrid(vis_list, cols=4, output_type='pil').save(osp.join(lmodel.directory, 'face_parsing_preview.jpg'), q=95)
                # imglist2imgrid(vis_list, cols=4, output_type='pil').save('local_tst.png')
                pass

            if lmodel.final_bbox is not None:
                face_xyxy[0] += lmodel.final_bbox[0]
                face_xyxy[2] += lmodel.final_bbox[0]
                face_xyxy[1] += lmodel.final_bbox[1]
                face_xyxy[3] += lmodel.final_bbox[1]
            lmodel.save_face_parsing(metadata=FACE_LABEL2NAME, face_seg_xyxy=face_xyxy)
            pass

        except Exception as e:
            raise
            print(f'failed to process {srcp}: {e}')



@cli.command('instance_segmentation')
@click.option('--exec_list')
@click.option('--rank_to_worldsize', default='', type=str)
def instance_segmentation(exec_list, rank_to_worldsize):
    
    from annotators.animeinsseg.instance_segmentation import apply_instance_segmentation
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    for p in tqdm(exec_list):
        img = Image.open(p)
        img = np.array(pil_ensure_rgb(img))
        instances = apply_instance_segmentation(img)
        
        instances_dict = {
            'masks': instances.masks if instances.masks is not None else [],
            'scores': instances.scores,
            'bboxes': instances.bboxes
        }
        instances_dict['masks'] = [mask2rle(m) for m in instances_dict['masks'] if instances_dict['masks'] is not None]

        d = osp.dirname(p)
        savep = osp.join(d, 'instances.json')
        dict2json(instances_dict, savep)



@cli.command('sam_infer_l2d')
@click.option('--exec_list')
@click.option('--ckpt', default='sam_l2d_19cls_iter2_18k')
@click.option('--rank_to_worldsize', default='', type=str)
def sam_infer_l2d(exec_list, ckpt, rank_to_worldsize):

    from live2d.scrap_model import animal_ear_detected, Drawable, VALID_BODY_PARTS_V1, VALID_BODY_PARTS_V2
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything, init_model_from_pretrained
    from utils.visualize import visualize_segs_with_labels
    from modules.semanticsam import SemanticSam, Sam
    import torch


    if ckpt == 'sam_l2d_19cls_iter2_18k':
        model: SemanticSam = init_model_from_pretrained(
            pretrained_model_name_or_path='24yearsold/l2d_sam_iter2',
            weights_name='checkpoint-18000.pt',
            module_cls=SemanticSam,
            download_from_hf=True,
            model_args=dict(class_num=19)
        ).to(device='cuda')
    else:
        raise Exception(f'Invalid ckpt: {ckpt}')

    seed_everything(42)

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:

            instance_mask, crop_xyxy, score = load_detected_character(p)
            # if instance_mask is None:
            #     print(f'skip {p}, no character instance detected')
            #     continue
            
            lmodel = Live2DScrapModel(p, crop_xyxy=crop_xyxy, pad_to_square=False)
            lmodel.init_drawable_visible_map()
            final_img = compose_from_drawables(lmodel.drawables)

            model_dir = lmodel.directory

            with torch.inference_mode():
                preds = model.inference(final_img[..., :3])[0]
                masks_np = (preds > 0).to(device='cpu', dtype=torch.bool).numpy()

            batch_save_masks(masks_np, osp.join(model_dir, 'sam_masks.json'))

        except Exception as e:
            print(f'Failed to process {p}: {e}')



@cli.command('infer_synsample_tags')
@click.option('--exec_list')
@click.option('--tags', default='objects,fullpage')
@click.option('--rank_to_worldsize', default='', type=str)
def infer_synsample_tags(exec_list, tags, rank_to_worldsize):
    
    # from annotators.animeinsseg.instance_segmentation import apply_instance_segmentation
    from annotators.wdv3_tagger import apply_wdv3_tagger
    import os.path as osp
    tags = tags.split(',')
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    img_ext = '.png'

    tagcluster_bodypart = json2dict('assets/tagcluster_bodypart_v2.json')
    tag2generaltag = {}
    for general_tag, tlist in tagcluster_bodypart.items():
        for t in tlist:
            if t in tag2generaltag and tag2generaltag[t] != general_tag:
                print(f'conflict tag def: {t} - {general_tag}, ' + tag2generaltag[t])
            tag2generaltag[t] = general_tag
    valid_taglst = set(list(tag2generaltag.keys()) + ['smile'])

    for p in tqdm(exec_list):

        pbase = osp.splitext(p)[0]

        for t in tags:
            if t == 'fullpage':
                p = pbase
            else:
                p = pbase + f'_{t}'

            try:
                imgp = p + img_ext
                if not osp.exists(imgp):
                    # raise Exception(f'{imgp}')
                    continue
                img = Image.open(p + img_ext)
                img = pil_ensure_rgb(img)
                img_input = img.resize((448, 448), resample=Image.Resampling.LANCZOS)

                caption, taglist, ratings, character, general = apply_wdv3_tagger(img_input, exclude_cls=exclude_cls)

                # img_input.save('local_tst.png')
                # general = [t for t in general if t in valid_taglst]
                
                general_tags = ','.join([t for t in general])

                savep = p + '.txt'
                # print(general_tags)
                with open(savep, 'w', encoding='utf8') as f:
                    f.write(general_tags)
                # print(savep)
            
            except Exception as e:
                print(f'failed on {p}: ', e)
        # return

        
if __name__ == '__main__':
    cli()