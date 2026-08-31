from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Map HuggingFace repo ID to local path for offline mode
        import os
        
        # Navigate from: .../TRELLIS.2/trellis2/pipelines/rembg/BiRefNet.py
        # To: .../models/trellis2/BiRefNet/RMBG-2.0
        # Need 6 levels up: rembg -> pipelines -> trellis2 -> TRELLIS.2 -> sd-webui-trellis2 -> extensions -> project_root
        hf_to_local_map = {
            'briaai/RMBG-2.0': 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..', '..', 'models', 'trellis2', 'BiRefNet', 'RMBG-2.0'),
            'ZhengPeng7/BiRefNet': 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..', '..', 'models', 'trellis2', 'BiRefNet', 'RMBG-2.0'),
        }
        
        # Check if model_name is a HF repo ID and map to local path
        local_model_path = model_name
        for hf_repo_id, local_path in hf_to_local_map.items():
            if model_name == hf_repo_id or model_name.startswith(hf_repo_id):
                if os.path.exists(local_path):
                    print(f"[TRELLIS.2] ✓ Mapping RMBG HF repo '{hf_repo_id}' to local path: {local_path}")
                    local_model_path = local_path
                else:
                    print(f"[TRELLIS.2] ✗ Local path not found for RMBG: {local_path}")
                break
        
        self.model = AutoModelForImageSegmentation.from_pretrained(
            local_model_path, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    