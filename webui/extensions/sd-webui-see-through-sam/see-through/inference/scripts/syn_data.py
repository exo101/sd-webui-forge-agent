import os
import random
import os.path as osp
import numpy as np
from pathlib import Path
import shutil
import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from PIL import Image
from  tqdm import tqdm
import click
import cv2

from utils.cv import mask2rle, rle2mask, mask_xyxy
from utils.io_utils import load_exec_list, find_all_files_recursive, find_all_files_with_name, pil_ensure_rgb, imglist2imgrid, imread, imwrite, json2dict, save_tmp_img, dict2json
from utils.sampler import NameSampler
from utils.visualize import visualize_segs, visualize_segs_with_labels, visualize_pos_keypoints
from live2d.scrap_model import Live2DScrapModel, VALID_BODY_PARTS_V1, VALID_BODY_PARTS_V2, compose_mask_from_drawables, init_drawable_visible_map, load_detected_character, load_pos_estimation


@click.group()
def cli():
    """live2d data processing related scripts.
    """


def get_unique_render_lst(exec_list):
    unique_lst = []
    processed_models = set()
    unique_src_to_models = dict()
    for p in tqdm(exec_list):
        modeld = osp.dirname(p)
        if modeld not in processed_models:
            processed_models.add(modeld)
        else:
            continue
        plist = sub_render_parts([p])
        mlist = [Live2DScrapModel(p, skip_load=True) for p in plist]
        for m in mlist:
            m.init_drawables()
        unique_mlist = [mlist[4]]
        for m in mlist:
            is_unique = True
            mklist = list(m.did2drawable.keys())
            mklist.sort()
            for um in unique_mlist:
                umklist = list(um.did2drawable.keys())
                umklist.sort()
                if mklist == umklist:
                    srcp = um.directory
                    is_unique = False
                    break

            tgtp = m.directory
            if is_unique:
                unique_mlist.append(m)
                srcp = m.directory
            if srcp not in unique_src_to_models:
                unique_src_to_models[srcp] = []
            unique_src_to_models[srcp].append(tgtp)
                
        unique_mlist = [m.directory for m in unique_mlist]
        unique_lst += unique_mlist

    return unique_lst, unique_src_to_models


@cli.command('get_tgt_list')
@click.option('--src_dir')
@click.option('--savep', default=None)
def get_tgt_list(src_dir, savep):
    if savep is None:
        savep = osp.join('workspace/datasets', osp.basename(src_dir) + '.txt')

    valid_list = []
    for f in find_all_files_recursive(src_dir, ext={'.json'}):
        tgtf = f.rstrip('.json') + '.png'
        if osp.exists(tgtf):
            valid_list.append(tgtf)
    print(len(valid_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(valid_list))


@cli.command('get_png_list')
@click.option('--src_dir')
@click.option('--savep', default=None)
def get_png_list(src_dir, savep):
    if savep is None:
        savep = osp.join('workspace/datasets', osp.basename(src_dir) + '.txt')

    valid_list = []
    for f in find_all_files_recursive(src_dir, ext={'.png'}):
        valid_list.append(f)
    print(len(valid_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(valid_list))


@cli.command('check_unique_rst')
@click.option('--exec_list')
@click.option('--savep', default=None)
def check_unique_rst(exec_list, savep):
    if savep is None:
        savep = exec_list
    exec_list = load_exec_list(exec_list)
    exec_list, unique_src_to_models = get_unique_render_lst(exec_list)
    print(len(exec_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(exec_list))
    dict2json(unique_src_to_models, savep + '.json')


@cli.command('compress_live2d')
@click.option('--src_dir')
@click.option('--save_dir')
@click.option('--ext', default='.jxl')
@click.option('--disable_crop', is_flag=True, default=False)
def compress_live2d(src_dir, save_dir, ext, disable_crop):

    src_dir = osp.normpath(src_dir)
    model_final_list = find_all_files_with_name(src_dir, 'final')

    crop = not disable_crop
    if save_dir is None:
        save_dir = src_dir +  f'_{ext}'
        if crop:
            save_dir += '_crop'
    save_dir = osp.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    ndir_leading = len(src_dir.split(os.path.sep))
    for model_f in tqdm(model_final_list, desc=f'saving to {save_dir}'):
        model_dir = osp.dirname(model_f)
        model_save_dir = model_dir.split(os.path.sep)[ndir_leading:]
        model = Live2DScrapModel(model_dir, crop_to_final=crop, pad_to_square=False)
        model.save_model_to(osp.join(save_dir, *model_save_dir), 
                            crop_to_final=crop, img_ext=ext)


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

    with open(osp.join(save_dir, f'{save_name}.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(tgt_list))

    if num_chunk > 0:
        world_size = num_chunk
        for ii in range(world_size):
            t = load_exec_list(tgt_list, ii, world_size=world_size)
            with open(osp.join(save_dir, f'{save_name}{ii}.txt'), 'w', encoding='utf8') as f:
                f.write('\n'.join(t))
            print(f'chunk {ii} num samples: {len(t)}')




@cli.command('render_face_samples')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--save_dir')
@click.option('--rank_to_worldsize', default='', type=str)
def render_face_samples(exec_list, bg_list, save_dir, rank_to_worldsize):

    TARGET_FRAME_SIZE = 2048

    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, img_bbox, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything
    from utils.visualize import FACE_LABEL2NAME

    def _compose_face_samples(lmodel: Live2DScrapModel):
        '''
        todo: save complete part
        '''
        face_xyxy = lmodel.face_seg_xyxy
        face_h, face_w = face_xyxy[3] - face_xyxy[1], face_xyxy[2] - face_xyxy[0]
        all_face_labels = list(FACE_LABEL2NAME.keys())
        face_final = lmodel.compose_face_drawables(list(FACE_LABEL2NAME.keys()), xyxy=face_xyxy)
        # save_tmp_img(face_final, 'local_tmp.png')

        part_mask_list = []
        # segmap = np.zeros((face_h, face_w), dtype=np.int32)
        alphas = np.zeros((face_h, face_w), dtype=np.int32)
        for ii in range(1, len(all_face_labels)):
            m  = lmodel.compose_face_drawables(ii, mask_only=True, xyxy=face_xyxy, final_visible_mask=True).astype(np.uint8)
            # save_tmp_img(m, mask2img=True)
            part_mask_list.append(m)

        mask_bg = np.bitwise_not(np.bitwise_or.reduce(np.stack(part_mask_list).astype(bool), axis=0))
        part_mask_list.insert(0, mask_bg.astype(np.uint8))
        
        nose_detected, mouth_detected = lmodel.face_part_detected([10, 11])
        tp = osp.join(lmodel.directory, 'faceseg_nosemouth.json.gz')
        if osp.exists(tp) and (not nose_detected or not mouth_detected):
            nose_mouth = batch_load_masks(tp)
            if not nose_detected:
                part_mask_list[10] = nose_mouth[0]
                part_mask_list[1][np.where(nose_mouth[0] > 0)] = 0
            if not mouth_detected:
                part_mask_list[11] = nose_mouth[1]
                part_mask_list[1][np.where(nose_mouth[1] > 0)] = 0

        bx, by, bw, bh = cv2.boundingRect(cv2.findNonZero(part_mask_list[0].astype(np.uint8)))
        by2 = by + bh
        bx2 = bw + bx

        # DONT DELETE THESE!!!!
        
        # depth_lower = 100000
        # depth_upper = -1
        # for d_id, drawable in enumerate(lmodel.drawables):
        #     if drawable.area < 1 or not drawable.face_part_id == 1:
        #         continue

        #     dx, dy, dw, dh = drawable.get_bbox(xyxy=face_xyxy)
        #     dx2 = dx + dw
        #     dy2 = dy + dh

        #     # check if hair drawable is actually background
        #     if drawable.face_part_id == 17:
        #         if drawable.face_part_stats['ioa'][0] > 0.7 and drawable.face_part_stats['ioa'][17] < 0.3:
        #             drawable.face_part_id = None

        #     if drawable.face_part_id == 1 and dw / bw > 0.7 and dh > bw > 0.7:
        #         if drawable.draw_order < depth_lower:
        #             depth_lower = drawable.draw_order
        #         if drawable.draw_order > depth_upper:
        #             depth_upper = drawable.draw_order

        # depth_buffer = np.zeros((face_h, face_w), dtype=np.uint8)
        # base_depth = 1
        # mask = np.zeros_like(depth_buffer, dtype=bool)
        # valid_face_ids = set(range(1, 19))
        # for d in lmodel.drawables:
        #     if d.area < 1 or d.face_part_id not in valid_face_ids:
        #         continue
        #     if np.any(d.bitwise_and(mask, face_xyxy)):
        #         base_depth += 1
        #     m = d.get_full_mask(xyxy=face_xyxy)
        #     mask |= m 
        #     d.depth = base_depth
        #     depth_buffer[np.where(m)] = base_depth
        
        # depth = (depth_buffer / np.max(depth_buffer) * 255).astype(np.uint8)
        # save_tmp_img(depth)


        # base_face_mask = compose_from_drawables([d for d in lmodel.drawables if \
        #                                          drawable.draw_order >= depth_lower and drawable.draw_order > depth_upper])
        # for drawable in lmodel.drawables:
        #     if drawable.draw_order < depth_lower or drawable.draw_order > depth_upper:
        #         continue

        # segmap = segmap.astype(np.uint8)
        # lmodel.compose_face_drawables([4, 5], output_type='pil').save('local_tst.png')
        # save_tmp_img(face_final)
        # save_tmp_img(segmap == 1, mask2img=True)
        # save_tmp_img(segmap == 4, mask2img=True)

        return True, part_mask_list, face_final
    
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(42)

    hist_match_prob = 0.2
    quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': quantize_prob})
    
    if exec_list.endswith('.json'):
        new_exec_list = []
        exec_list = json2dict(exec_list)
        for k, vs in exec_list.items():
            for v in vs:
                new_exec_list.append({v: k})
        exec_list = new_exec_list
        pass
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    VALID_FACE_SET = set(range(19))
    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:
            face_parsingp = None
            if isinstance(p, dict):
                for k, v in p.items():
                    p = k
                    face_parsingp = osp.join(v, 'face_parsing.json')
            lmodel = Live2DScrapModel(p)
            model_dir = lmodel.directory

            if face_parsingp is None:
                face_parsingp = osp.join(model_dir, 'face_parsing.json')
                if not osp.exists(face_parsingp):
                    face_parsingp = '-'.join(model_dir.split('-')[:-1]) + '-4'
                    face_parsingp = osp.join(face_parsingp, 'face_parsing.json')
            if not osp.exists(face_parsingp):
                print(f"skip {p} due to face parsing result not found")
                continue
            
            lmodel.load_face_parsing(face_parsingp)
            face_drawables = [d for d in lmodel.drawables if d.face_part_id in VALID_FACE_SET]
            init_drawable_visible_map(face_drawables)

            is_valid, labels, face_final = _compose_face_samples(lmodel,)
            mask_list = labels
            if not is_valid:
                continue
            # save_tmp_img(labels[0], mask2img=True)

            bgp = random.choice(bg_list)
            fh, fw = face_final.shape[:2]
            bg = imread(bgp)
            bgh, bgw = bg.shape[:2]
            target_bg_size = min(bgh, bgw, TARGET_FRAME_SIZE)
            fsize = max(fh, fw)
            if fsize * 2 < target_bg_size:
                target_bg_size = random.randint(fsize * 2, target_bg_size)
            bg = resize_short_side_to(bg, target_bg_size)
            bg = random_crop(imread(bgp), (fh, fw))
            # save_tmp_img(bg)
            
            color_correct = color_correction_sampler.sample()
            if color_correct == 'hist_match':
                fgbg_hist_matching([face_final], bg)

            face_wbg = img_alpha_blending([bg, face_final])
            
            if color_correct == 'quantize':
                mask = face_final[..., -1] > 35
                # cv2.imshow("mask", mask)
                face_wbg[..., :3] = quantize_image(face_wbg[..., :3], random.choice([12, 16, 32]), 'kmeans', mask=mask)[0]
            
            d = osp.abspath(model_dir).replace('\\', '/').rstrip('/').replace('.', '_DOT_')
            d1 = d.split('/')[-1]
            d2 = d.split('/')[-3]
            savename = d2 + '____' + d1
            savep = osp.join(save_dir, savename)
            # save_tmp_img(face_wbg)
            imwrite(savep, face_wbg, quality=97, ext='.jpg')
            batch_save_masks(mask_list, savep + '.json', compress='gzip')
            # print(f'finished {savep}')
        except Exception as e:
            # raise
            print(f'Failed to process {p}: {e}')


@cli.command('get_tgt_list')
@click.option('--src_dir')
@click.option('--savep', default=None)
def get_tgt_list(src_dir, savep):
    if savep is None:
        savep = osp.join('workspace/datasets', osp.basename(src_dir) + '.txt')

    valid_list = []
    for f in find_all_files_recursive(src_dir, ext={'.json'}):
        tgtf = osp.splitext(f)[0] + '.png'
        if osp.exists(tgtf):
            valid_list.append(tgtf)
    print('valid samples: ', len(valid_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(valid_list))
    print(f'exec_list saved to {savep}')

@cli.command('render_body_samples')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--mask_name', default=None)
@click.option('--save_dir', default='')
@click.option('--rank_to_worldsize', default='', type=str)
@click.option('--save_suffix', default='.png', type=str)
def render_body_samples(exec_list, bg_list, mask_name, save_dir, rank_to_worldsize, save_suffix):

    from live2d.scrap_model import animal_ear_detected, Drawable, ImageProcessor, compose_from_drawables, VALID_BODY_PARTS_V3
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything

    seed_everything(42)

    hist_match_prob = 0.35
    # quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': 0.})
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    tagcluster_bodypart = json2dict('common/assets/tagcluster_bodypart_v2.json')
    tag2generaltag = {}
    for general_tag, tlist in tagcluster_bodypart.items():
        for t in tlist:
            if t in tag2generaltag and tag2generaltag[t] != general_tag:
                print(f'conflict tag def: {t} - {general_tag}, ' + tag2generaltag[t])
            tag2generaltag[t] = general_tag

    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    render_sample = save_dir != ''

    MAX_TGT_SIZE = 1280
    target_tag_list = VALID_BODY_PARTS_V3 + ['head']

    invalid_lst: list[int] = [2094, 1389, 627, 477, 280, 480]


    for ii, p in enumerate(tqdm(exec_list)):
        try:
            lmodel = Live2DScrapModel(p)
            load_success = lmodel.load_body_parsing(mask_name)
            if not load_success:
                print(f'failed to load body parsing, skip: {p}')
                continue

            metadata = lmodel._body_parsing['metadata']
            if metadata is None:
                metadata = {}
            is_valid = metadata.get('is_valid', True)
            is_incomplete = metadata.get('is_incomplete', False)
            is_cleaned = metadata.get('cleaned', False)
            tag_valid = metadata.get('tag_valid', {})
            object_valid = True
            foot_valid = True

            if not is_valid:
                continue

            # if is_incomplete:
            #     continue

            # keep_bg = random.random() < 0.3
            keep_bg = False

            if not is_valid:
                continue
            
            valid_drawables: list[Drawable] = []
            body_drawables: list[Drawable] = []
            h, w = lmodel.size()
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for d in lmodel.drawables:
                d.get_img()
                if d.area < 1:
                    continue
                if not keep_bg and d.body_part_tag not in target_tag_list:
                    continue
                valid_drawables.append(d)
                if d.body_part_tag in target_tag_list:
                    body_drawables.append(d)
                dxyxy = d.xyxy
                x_min = min(x_min, dxyxy[0])
                x_max = max(x_max, dxyxy[2])
                y_min = min(y_min, dxyxy[1])
                y_max = max(y_max, dxyxy[3])

            if keep_bg:
                x_min = y_min = 0
                x_max = w
                y_max = h

            ch, cw = y_max - y_min, x_max - x_min
            scale = min(MAX_TGT_SIZE / max(ch, cw), 1)
            nh, nw = ch, cw
            if scale < 1:
                nh = int(round(nh * scale))
                nw = int(round(nw * scale))
            new_processor = ImageProcessor(target_frame_size=[nw, nh], crop_bbox=[x_min, y_min, x_max, y_max], pad_to_square=False)
            lmodel.final = new_processor(lmodel.final, update_coords_modifiers=True)
            lmodel.final_bbox = [
                new_processor.crop_bbox[0] + x_min,
                new_processor.crop_bbox[1] + y_min,
                new_processor.crop_bbox[0] + x_max,
                new_processor.crop_bbox[1] + y_max
            ]
            for d in valid_drawables:
                d.set_img_processor(new_processor)
                d._final_size = [nh, nw]
                d.load_img(force_reload=True, img=d.img)
            
            h, w = lmodel.size()
            depth_buffer = np.zeros((h, w), dtype=np.uint16)
            base_depth = 1
            init_drawable_visible_map(valid_drawables)
            # part_mask_list, body_final = _compose_body_samples(lmodel)
            
            part_mask_list = []
            if not keep_bg:
                body_final = lmodel.compose_bodypart_drawables(target_tag_list)
            else:
                body_final = compose_from_drawables(valid_drawables)

            for tag in target_tag_list:
                m  = lmodel.compose_bodypart_drawables(tag, mask_only=True, final_visible_mask=True).astype(np.uint8)
                part_mask_list.append(m)
            
            mask = np.zeros((h, w), dtype=bool)
            for d in body_drawables:
                m = d.get_full_mask()
                if np.any(d.bitwise_and(mask, [0, 0, w, h])):
                    base_depth += 1
                    mask = m
                else:
                    mask |= m                
                d.depth = base_depth
                depth_buffer[np.where(m)] = base_depth
            
            depth_dtype = np.uint8
            if base_depth > 255:
                depth_dtype = np.uint16
            depth_buffer = depth_buffer.astype(depth_dtype)

            d = osp.abspath(lmodel.directory).replace('\\', '/').rstrip('/').replace('.', '_DOT_')
            d1 = d.split('/')[-1]
            d2 = d.split('/')[-3]
            savename = d2 + '____' + d1
            savep = osp.join(save_dir, savename)

            masks = part_mask_list
            foot_msk_idx = target_tag_list.index('footwear')
            object_msk_idx = target_tag_list.index('objects')
            leg_msk_idx = target_tag_list.index('legwear')
            masks[leg_msk_idx] = masks[leg_msk_idx] | masks[foot_msk_idx]
            px = py = 0
            
            final_img = body_final

            bgp = random.choice(bg_list)
            fh, fw = final_img.shape[:2]
            bg = imread(bgp)
            fsize = min(max(h, w), MAX_TGT_SIZE)
            fsze_max = int(round(fsize * 1.5))
            target_bg_size = random.randint(fsize, fsze_max)
            bg = resize_short_side_to(bg, target_bg_size)

            target_bg_w = target_bg_h = target_bg_size
            if fh > fw:
                target_bg_w = random.randint(fw, target_bg_size)
            elif fw > fh:
                target_bg_h = random.randint(fh, target_bg_size)

            bg = random_crop(imread(bgp), (target_bg_h, target_bg_w))
            
            px = py = 0
            if fh != target_bg_h or fw != target_bg_w:
                if fh != target_bg_h:
                    py = random.randint(0, target_bg_h - fh)
                if fw != target_bg_w:
                    px = random.randint(0, target_bg_w - fw)
                blank_final = np.zeros((target_bg_h, target_bg_w, 4), np.uint8)
                blank_final[py: py + fh, px: px + fw] = final_img
                final_img = blank_final

                depth_blank = np.zeros((target_bg_h, target_bg_w), dtype=depth_dtype)
                depth_blank[py: py + fh, px: px + fw] = depth_buffer
                depth_buffer = depth_blank

                for mi, m in enumerate(masks):
                    blank = np.zeros((target_bg_h, target_bg_w), bool)
                    blank[py: py + fh, px: px + fw] = m
                    masks[mi] = blank
            fh, fw = final_img.shape[:2]

            color_correct = color_correction_sampler.sample()

            if color_correct == 'hist_match':
                fgbg_hist_matching([final_img], bg, fg_only=True)

            wbg = img_alpha_blending([bg, final_img])
            wbg[..., -1] = final_img[..., -1]


            fh, fw = wbg.shape[:2]

            # save_tmp_img(visualize_segs_with_labels(masks, wbg[..., :3], tag_list=target_tag_list, image_weight=0.1))
            imwrite(savep, wbg, quality=100, ext=save_suffix)
            imwrite(savep + '_depth', depth_buffer, quality=100, ext='.png')
            
            mask_meta_list = [{} for _ in range(len(target_tag_list))] # dont use [{}] * len
            mask_meta_list[foot_msk_idx]['is_valid'] = foot_valid
            mask_meta_list[object_msk_idx]['is_valid'] = object_valid
            batch_save_masks(masks, savep + '.json', mask_meta_list=mask_meta_list)
            del masks
            # del wbg
            del depth_buffer

            sample_ann = {'cleaned': is_cleaned, 'is_incomplete': is_incomplete, 'tag_info': {k: {'valid': True, 'exists': False} for k in VALID_BODY_PARTS_V2}, 'final_size': wbg.shape[:2]}
            tag_info = sample_ann['tag_info']
            # tag_info['footwear']['valid'] = foot_valid
            # tag_info['objects']['valid'] = object_valid

            for ii, tag in enumerate(target_tag_list):
                # if tag == 'footwear' and not foot_valid:
                #     continue
                # if tag == 'objects' and not object_valid:
                #     continue
                if tag == 'head':
                    drawables = lmodel.get_body_part_drawables(['face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'nose', 'mouth'])
                else:
                    drawables = lmodel.get_body_part_drawables(tag)
                # if tag == 'legwear':
                #     drawables += lmodel.get_body_part_drawables('footwear')
                drawables = [d for d in drawables if d.area >= 1]
                if len(drawables) == 0:
                    continue
                init_drawable_visible_map(drawables)
                x_min, x_max, y_min, y_max = fw, 0, fh, 0
                for d in drawables:
                    dxyxy = d.xyxy
                    x_min = min(x_min, dxyxy[0])
                    x_max = max(x_max, dxyxy[2])
                    y_min = min(y_min, dxyxy[1])
                    y_max = max(y_max, dxyxy[3])
                
                xyxy = [x_min, y_min, x_max, y_max]
                dh, dw = y_max - y_min, x_max - x_min
                part_final = compose_from_drawables(drawables, xyxy=xyxy)
                imwrite(savep + f'_{tag}', part_final, quality=100, ext='.png')
                
                depth_buffer = np.zeros((dh, dw), dtype=depth_dtype)
                for d in drawables:
                    dxyxy = d.xyxy
                    m = d.final_visible_mask
                    depth_buffer[dxyxy[1] - y_min: dxyxy[3] - y_min, dxyxy[0] - x_min: dxyxy[2] - x_min][np.where(m)] = d.depth
                
                xyxy = [x_min + px, y_min + py, x_max + px, y_max + py]
                imwrite(savep + f'_{tag}_depth', depth_buffer, quality=100, ext='.png')
                if tag not in tag_info:
                    tag_info[tag] = {}
                tag_info[tag]['exists'] = True
                tag_info[tag]['xyxy'] = xyxy

                blank = np.zeros_like(wbg)
                blank[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]] = part_final
                # save_tmp_img(wbg)
                # save_tmp_img(img_alpha_blending([wbg, blank]))
                # pass
                
            dict2json(sample_ann, savep + '_ann.json')

        except Exception as e:
            raise
            print(f'Failed to process {p}: {e}')





if __name__ == '__main__':
    cli()