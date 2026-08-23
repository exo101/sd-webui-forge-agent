# animeinsseg

**Conda env:** `live2d_ann_mmdet3`

## Setup

```bash
conda create -n live2d_ann_mmdet3 python=3.12 -c conda-forge
conda activate live2d_ann_mmdet3
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install numpy==1.26.4 accelerate==1.7.0 pillow-jxl-plugin opencv-python==4.10.0.84 \
  openmim==0.3.9 ipykernel einops
pip install setuptools==80.9.0
pip install git+https://github.com/dmMaze/mmdetection@dev-3.x#egg=mmdet
```

## Known Issues

- Dependency hell — cannot be expressed as a single conda YAML. See `../../ISSUES.md`.
