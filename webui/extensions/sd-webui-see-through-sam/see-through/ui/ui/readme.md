
# Installation

## Run the source code
### Windows
Python > 3.11 might break mmcv, Python==3.10 works well
``` shell
pip install -r requirements.txt
pip install -U openmim
# torch > 2.1 might not work for mmcv
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 
pip install numpy==1.26.4
mim install mmcv==2.1.0
mim install mmdet==3.3.0
python launch.py
```

### MacOS
Tested with Python==3.11
``` shell
pip install -r requirements.txt
pip install -U openmim
# torch != 2.1.0 might not work for mmcv
pip install torch==2.1.0 torchvision torchaudio
pip install numpy==1.26.4
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
mim install mmdet==3.3.0
python launch.py
```

# Usage
Open (or drag & drop) the folder contain the images you want to process, and click run


## Tips & Shortcuts
* ``` Ctrl ``` (Pressed) + Mouse Wheel to scale the canvas 
* ```A```/```D``` or ```pageUp```/```Down``` to turn the page
* Instances can be selected, moved, and deleted
* ```Ctrl+Z```, ```Ctrl+Shift+Z```, ```Ctrl+Y```, ```Ctrl+Shift+Y``` can undo/redo most operations. (note the undo stack will be cleared after you turn the page)
* ```W``` to enter / quit the box prompting mode. On the canvas, drag the mouse with the right button pressed to create a box. Note if **something is selected**, it will pop a context menu instead of creating a box. Box mode and batch processing share the same run button, quit it to run batch processing.
