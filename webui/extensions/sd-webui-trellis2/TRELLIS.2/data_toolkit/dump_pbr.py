import os
import shutil
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import tempfile


BLENDER_LINK = 'https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
    os.system(f'{BLENDER_PATH} -b --python {os.path.join(os.path.dirname(__file__), "blender_script", "install_pillow.py")}')


def _dump_pbr(file_path, metadatum, root):
    sha256 = metadatum['sha256']
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = os.path.join(tmp_dir, f'{sha256}.pickle')
        output_path = os.path.join(root, 'pbr_dumps', f'{sha256}.pickle')
        args = [
            BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'dump_pbr.py'),
            '--',
            '--object', os.path.expanduser(file_path),
            '--output_path', os.path.expanduser(temp_path)
        ]
        if file_path.endswith('.blend'):
            args.insert(1, file_path)
        
        call(args, stdout=DEVNULL, stderr=DEVNULL)
        
        if os.path.exists(temp_path):
            shutil.move(temp_path, output_path)
            return {'sha256': sha256, 'pbr_dumped': True}
        else:
            if os.path.exists(temp_path + '_error.txt'):
                with open(temp_path + '_error.txt', 'r') as f:
                    error_msg = f.read()
                raise ValueError(f'Failed to dump PBR. File {file_path}. Error message: {error_msg}')
            else:
                raise ValueError(f'Failed to dump PBR. File {file_path}.')

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to save the mesh dumps')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root

    os.makedirs(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'new_records'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'pbr_dumped' in metadata.columns:
            metadata = metadata[metadata['pbr_dumped'] != True]
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
    sha256_list = os.listdir(os.path.join(opt.pbr_dump_root, 'pbr_dumps'))
    sha256_list = [os.path.splitext(f)[0] for f in sha256_list if f.endswith('.pickle')]
    for sha256 in sha256_list:
        records.append({'sha256': sha256, 'pbr_dumped': True})
    print(f'Found {len(sha256_list)} dumped PBRs')
    metadata = metadata[~metadata['sha256'].isin(sha256_list)]
       
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_dump_pbr, root=opt.pbr_dump_root)
    pbr_dumped = dataset_utils.foreach_instance(metadata, opt.download_root, func, max_workers=opt.max_workers, desc='Dumping PBR')
    pbr_dumped = pd.concat([pbr_dumped, pd.DataFrame.from_records(records)])
    pbr_dumped.to_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'new_records', f'part_{opt.rank}.csv'), index=False)
