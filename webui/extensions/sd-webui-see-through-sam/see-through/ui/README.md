# ui

Desktop UI sub-codebase: Qt6 application for Live2D model annotation and management.

Uses the `see_through` conda env. See the [root README](../README.md) for setup.

## Usage

**Always run from the repo root as the working directory:**

```bash
cd /path/to/see-through
conda activate see_through
python ui/ui/launch.py
```

The `assets` symlink must exist at repo root (see root README).
Without it, the UI will crash on startup with `FileNotFoundError`.

### Workspace setup

The UI expects project data under `workspace/datasets/`. Each project is a directory
containing annotated Live2D model folders:

```
workspace/
└── datasets/
    └── <project_name>/
        ├── exec_list.txt              # List of model paths (one per line)
        └── <model_name>/
            ├── final.jxl              # Source image (JXL format)
            ├── final.json             # Project metadata (auto-created by UI)
            ├── instances.json         # Instance annotations
            └── *_masks.json           # Segmentation masks (from inference)
```

To open a project, use **File > Open** and select the `exec_list.txt` or project `.json`
file, or launch directly with:

```bash
python ui/ui/launch.py --proj workspace/datasets/<project_name>
```

Refer to [CubismPartExtr](https://github.com/shitagaki-lab/CubismPartExtr) for how to
prepare workspace data from Live2D model files.

### Windows

Double-click `launch_ui_win.bat` from the `ui/` directory.

### Headless testing (Xvfb)

For visual testing on headless Linux:

```bash
sudo bash ui/install_system_deps.sh
Xvfb :99 -screen 0 1920x1080x24 &
DISPLAY=:99 python ui/ui/launch.py
```
