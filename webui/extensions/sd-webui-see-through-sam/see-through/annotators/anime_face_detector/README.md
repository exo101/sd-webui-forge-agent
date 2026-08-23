# anime_face_detector

**Conda env:** `live2d_ann_mmpose`

## Setup

```bash
conda create -n live2d_ann_mmpose python=3.10 -c conda-forge
conda activate live2d_ann_mmpose
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install openmim==0.3.6 numpy==1.26.4 opencv-python==4.10.0.84
mim install mmcv-full==1.7.0
mim install mmdet==2.28.2
mim install mmpose==0.29.0
```

## Known Issues

- Env compatibility with `common/` not verified. See `../../ISSUES.md`.
