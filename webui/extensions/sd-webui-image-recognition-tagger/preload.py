from pathlib import Path
from argparse import ArgumentParser

from modules.paths_internal import models_path

default_ddp_path = Path(models_path, 'deepdanbooru')


def preload(parser: ArgumentParser):
    parser.add_argument(
        '--deepdanbooru-projects-path',
        type=str,
        help='Path to directory with DeepDanbooru project(s).',
        default=default_ddp_path
    )
