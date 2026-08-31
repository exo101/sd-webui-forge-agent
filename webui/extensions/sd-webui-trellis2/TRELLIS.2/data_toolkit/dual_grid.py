import os
import sys
import importlib
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import o_voxel
from easydict import EasyDict as edict
from functools import partial


def _dual_grid_mesh(file, metadatum, mesh_dump_root, root):
    sha256 = metadatum['sha256']
    try:
        pack = {'sha256': sha256}
        data = None
        for res in opt.resolution:
            need_process = False

            # check if already processed
            if os.path.exists(os.path.join(root, f'dual_grid_{res}', f'{sha256}.vxz')):
                try:
                    info = o_voxel.io.read_vxz_info(os.path.join(root, f'dual_grid_{res}', f'{sha256}.vxz'))
                    pack[f'dual_grid_converted_{res}'] = True
                    pack[f'dual_grid_size_{res}'] = info['num_voxel']
                except Exception as e:
                    print(f'Error reading {sha256}.vxz: {e}')
                    need_process = True
            else:
                need_process = True

            # process mesh
            if need_process:
                if data is None:
                    with open(os.path.join(mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle'), 'rb') as f:
                        dump = pickle.load(f)
                    start = 0
                    vertices = []
                    faces = []
                    for obj in dump['objects']:
                        if obj['vertices'].size == 0 or obj['faces'].size == 0:
                            continue
                        vertices.append(obj['vertices'])
                        faces.append(obj['faces'] + start)
                        start += len(obj['vertices'])
                    vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
                    faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
                    vertices_min = vertices.min(dim=0)[0]
                    vertices_max = vertices.max(dim=0)[0]
                    center = (vertices_min + vertices_max) / 2
                    scale = 0.99999 / (vertices_max - vertices_min).max()
                    vertices = (vertices - center) * scale
                    assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
                    data = {'vertices': vertices, 'faces': faces}

                voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
                    **data,
                    grid_size=res,
                    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
                    face_weight=1.0,
                    boundary_weight=0.2,
                    regularization_weight=1e-2,
                    timing=False,
                )
                dual_vertices = dual_vertices * res - voxel_indices
                assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1+1e-3), 'dual_vertices out of range'
                dual_vertices = torch.clamp(dual_vertices, 0, 1)
                dual_vertices = (dual_vertices * 255).type(torch.uint8)
                intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
                
                o_voxel.io.write_vxz(
                    os.path.join(root, f'dual_grid_{res}', f'{sha256}.vxz'),
                    voxel_indices,
                    {'vertices': dual_vertices, 'intersected': intersected},
                )

                pack[f'dual_grid_converted_{res}'] = True
                pack[f'dual_grid_size_{res}'] = len(dual_vertices)

        return pack
    except Exception as e:
        print(f'Error processing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to load mesh dumps')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory to save dual grids')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--resolution', type=str, default=256)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = [int(x) for x in opt.resolution.split(',')]
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.dual_grid_root = opt.dual_grid_root or opt.root

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
    for res in opt.resolution:
        if os.path.exists(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'metadata.csv')):
            dual_grid_metadata = pd.read_csv(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'metadata.csv')).set_index('sha256')
            dual_grid_metadata = dual_grid_metadata.rename(columns={'dual_grid_converted': f'dual_grid_converted_{res}', 'dual_grid_size': f'dual_grid_size_{res}'})
            metadata = metadata.combine_first(dual_grid_metadata)
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['mesh_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        for res in opt.resolution:
            if f'dual_grid_converted_{res}' in metadata.columns:
                mask |= metadata[f'dual_grid_converted_{res}'] != True
            else:
                mask[:] = True
                break
        metadata = metadata[mask]
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
    
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_dual_grid_mesh, root=opt.dual_grid_root, mesh_dump_root=opt.mesh_dump_root)
    dual_grids = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Dual griding')
    if 'error' in dual_grids.columns:
        errors = dual_grids[dual_grids[f'error'].notna()]
        with open('errors.txt', 'w') as f:
            f.write('\n'.join(errors['sha256'].tolist()))
    for res in opt.resolution:
        if f'dual_grid_converted_{res}' in dual_grids.columns:
            dual_grid_metadata = dual_grids[dual_grids[f'dual_grid_converted_{res}'] == True]
            if len(dual_grid_metadata) > 0:
                dual_grid_metadata = dual_grid_metadata[['sha256', f'dual_grid_converted_{res}', f'dual_grid_size_{res}']]
                dual_grid_metadata = dual_grid_metadata.rename(columns={f'dual_grid_converted_{res}': 'dual_grid_converted', f'dual_grid_size_{res}': 'dual_grid_size'})
                dual_grid_metadata.to_csv(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'new_records', f'part_{opt.rank}.csv'), index=False)
    