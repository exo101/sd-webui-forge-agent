@echo off
call "D:\AI\sd-webui-forge-neo-v3.6.4\system\environment.bat"
cd /d "D:\AI\sd-webui-forge-neo-v3.6.4\webui"
set PYTHON=D:\AI\sd-webui-forge-neo-v3.6.4\system\python\python.exe
set SKIP_VENV=1
set PYTHONIOENCODING=utf-8
set COMMANDLINE_ARGS=--theme dark --port 7869 --listen --autolaunch --cuda-malloc --cuda-stream --pin-shared-memory --api --skip-install --skip-version-check --skip-torch-cuda-test --disable-sage --reserve-vram 1 --neveroom --ckpt-dirs "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\Stable-diffusion" --ckpt-dirs "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\diffusion_models" --text-encoder-dirs "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\text_encoder" --lora-dirs "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\Lora" --vae-dirs "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\VAE" --esrgan-models-path "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\ESRGAN" --controlnet-dir "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\ControlNet" --controlnet-preprocessor-models-dir "D:\AI\sd-webui-forge-neo-v3.6.4\webui\models\ControlNet\preprocessor"
call webui.bat
