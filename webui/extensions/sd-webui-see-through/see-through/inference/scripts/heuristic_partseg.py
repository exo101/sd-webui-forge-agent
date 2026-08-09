import os.path as osp
import argparse
import sys
import os
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from utils.inference_utils import seg_wdepth, seg_wlr, seg_wdepth_psd, seg_wlr_psd
import click

@click.group()
def cli():
    """live2d scripts.
    """


@cli.command('seg_wdepth')
@click.option('--srcp')
@click.option('--target_tags', default=None, help='tags to split seperate by \",\"')
def seg_wdepth_(srcp, target_tags):
    if osp.isfile(srcp):
        seg_wdepth_psd(srcp, target_tags)
    else:
        seg_wdepth(srcp)


@cli.command('seg_wlr')
@click.option('--srcp')
@click.option('--target_tags', default=None, help='tags to split seperate by \",\"')
def seg_wlr_(srcp, target_tags):
    if osp.isfile(srcp):
        seg_wlr_psd(srcp, target_tags)
    else:
        seg_wlr(srcp)


if __name__ == '__main__':
    cli()