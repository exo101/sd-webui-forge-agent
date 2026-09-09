# annotators

Annotator plugins for the inference pipeline. Installed as a Python package
via the unified `requirements.txt` at repo root (`-e ./annotators`).

## Package Structure

| Module | Purpose | Extra required |
|--------|---------|---------------|
| `wdv3_tagger` | WDv3 image tagging | (base) |
| `gradcam` | GradCAM heatmap generation | (base) |
| `lama_inpainter` | LaMa inpainting | (base) |
| `bizarre_tagger` | Bizarre pose estimation + BG segmentation | `[bizarre_tagger]` |
| `lang_sam` | Language-guided SAM segmentation | `[lang_sam]` |
| `animeinsseg` | Anime instance segmentation (mmdet3) | `[animeinsseg]` |
| `anime_face_detector` | Anime face detection (legacy) | Separate `ann_mmpose` env |

## Optional extras

The base annotators install with the unified env. For heavier extras:

```bash
# Body parsing + language SAM (~10 min, needs C++ compiler)
pip install -e annotators[bizarre_tagger,lang_sam]

# Instance segmentation (needs CUDA toolkit for mmcv build)
pip install -e annotators[animeinsseg]

# Or use the pre-configured tier file:
pip install -r requirements-inference-mmdet.txt

# Everything
pip install -e annotators[all]
```

### Anime face detection (separate env)

`anime_face_detector` requires legacy dependencies (PyTorch 1.13, mmcv-full 1.7)
incompatible with the unified env. Set up a dedicated conda environment:

```bash
conda create -n ann_mmpose python=3.10
conda activate ann_mmpose
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install openmim==0.3.6 numpy==1.26.4 opencv-python==4.10.0.84
mim install mmcv-full==1.7.0
mim install mmdet==2.28.2
mim install mmpose==0.29.0
pip install -e ./common
```

Run from the repo root:
```bash
conda run --no-capture-output -n ann_mmpose \
  python inference/scripts/parse_live2d.py facedet \
  --exec_list workspace/datasets/.../exec_list.txt
```
