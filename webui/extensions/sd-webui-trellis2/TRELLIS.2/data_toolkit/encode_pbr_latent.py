import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import torch
import numpy as np
import pandas as pd
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)

def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()

def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory to save the pbr voxel files')
    parser.add_argument('--pbr_latent_root', type=str, default=None,
                        help='Directory to save the pbr latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Sparse voxel resolution')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str,
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root
    opt.pbr_latent_root = opt.pbr_latent_root or opt.root

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_{opt.resolution}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name, 'new_records'), exist_ok=True)
    
    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{opt.resolution}', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{opt.resolution}','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name, 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name,'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_voxelized'] == True]
        if 'pbr_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['pbr_latent_encoded'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
        tqdm(total=len(metadata), desc="Filtering existing objects") as pbar:
        def check_sha256(sha256):
            if os.path.exists(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name, f'{sha256}.npz')):
                coords = np.load(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name, f'{sha256}.npz'))['coords']
                records.append({'sha256': sha256, 'pbr_latent_encoded': True, 'pbr_latent_tokens': coords.shape[0]})
            pbar.update()
        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)
    existing_sha256 = set(r['sha256'] for r in records)
    print(f'Found {len(existing_sha256)} processed objects')
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]
        
    print(f'Processing {len(metadata)} objects...')
    
    sha256s = list(metadata['sha256'].values)
    load_queue = Queue(maxsize=32)
    with ThreadPoolExecutor(max_workers=32) as loader_executor, \
         ThreadPoolExecutor(max_workers=32) as saver_executor:

        def loader(sha256):
            try:
                attrs = ['base_color', 'metallic', 'roughness', 'alpha']                
                coords, attr = o_voxel.io.read_vxz(
                    os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{opt.resolution}', f'{sha256}.vxz'),
                    num_threads=4
                )
                feats = torch.concat([attr[k] for k in attrs], dim=-1) / 255.0 * 2 - 1
                x = sp.SparseTensor(
                    feats.float(),
                    torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
                )
                load_queue.put((sha256, x))
            except Exception as e:
                print(f"[Loader Error] {sha256}: {e}")
                load_queue.put((sha256, None))

        loader_executor.map(loader, sha256s)
        
        def saver(sha256, pack):
            save_path = os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name, f'{sha256}.npz')
            np.savez_compressed(save_path, **pack)
            records.append({'sha256': sha256, 'pbr_latent_encoded': True, 'pbr_latent_tokens': pack['coords'].shape[0]})
            
        for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
            try:
                sha256, voxels = load_queue.get()
                if voxels is None:
                    print(f"[Skip] {sha256}: Failed to load input")
                    continue
                
                num_voxels = voxels.feats.shape[0]

                # NaN/Inf
                if not (is_valid_sparse_tensor(voxels)):
                    print(f"[Skip] {sha256}: NaN/Inf in input")
                    continue

                z = encoder(voxels.cuda())
                torch.cuda.synchronize()

                if not torch.isfinite(z.feats).all():
                    print(f"[Skip] {sha256}: Non-finite latent in z.feats")
                    clear_cuda_error()
                    continue

                pack = {
                    'feats': z.feats.cpu().numpy().astype(np.float32),
                    'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, pack)

            except Exception as e:
                print(f"[Error] {sha256} ({num_voxels} voxels): {e}")
                clear_cuda_error()
                continue
            
        saver_executor.shutdown(wait=True)
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_name, 'new_records', f'part_{opt.rank}.csv'), index=False)
