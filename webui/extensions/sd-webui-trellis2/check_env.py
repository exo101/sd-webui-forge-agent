"""
Check if TRELLIS.2 environment is properly set up
"""

import os
import sys

print("=" * 60)
print("TRELLIS.2 Extension - Environment Check")
print("=" * 60)
print()

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Check required packages
required_packages = [
    'torch',
    'gradio',
    'numpy',
    'Pillow',
    'opencv-python',
    'huggingface-hub',
]

print("Checking required packages:")
for package in required_packages:
    try:
        mod = __import__(package.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {package}: {version}")
    except ImportError:
        print(f"  ✗ {package}: NOT INSTALLED")
print()

# Check TRELLIS.2 modules (separate compiled wheels from source modules)
compiled_modules = [
    ('nvdiffrast', 'nvdiffrast'),
    ('nvdiffrec', 'nvdiffrec'),
    ('cumesh', 'CuMesh'),
    ('flexgemm', 'FlexGEMM'),
    ('utils3d', 'utils3d'),
]

source_modules = [
    ('trellis2', 'trellis2'),
    ('o_voxel', 'o-voxel'),
]

print("Checking compiled wheel modules:")
for import_name, display_name in compiled_modules:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'installed')
        print(f"  ✓ {display_name}: {version}")
    except ImportError:
        print(f"  ✗ {display_name}: NOT INSTALLED")
print()

print("Checking source modules:")
for import_name, display_name in source_modules:
    try:
        __import__(import_name)
        print(f"  ✓ {display_name}")
    except ImportError:
        print(f"  ✗ {display_name}: NOT FOUND")
print()

# Check model directory
from modules.paths import models_path
model_dir = os.path.join(models_path, 'trellis2')
print(f"Model directory: {model_dir}")
if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    if files:
        print(f"  ✓ Model directory exists ({len(files)} files)")
    else:
        print(f"  ⚠ Model directory exists but is empty")
else:
    print(f"  ✗ Model directory does not exist")
print()

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: ✓")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"CUDA available: ✗")
except ImportError:
    print(f"PyTorch not installed")
print()

print("=" * 60)
print("Environment check complete")
print("=" * 60)
