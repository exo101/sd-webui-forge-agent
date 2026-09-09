from typing import Union, List

from torch import nn
import torch
import numpy as np

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam import DeepFeatureFactorization
import timm

from .wdv3_tagger import MODEL_REPO_MAP

def reshape_transform_swinv2(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_vit(tensor, height=28, width=28):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_vit_large(tensor, height=32, width=32):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_eva(tensor, height=32, width=32):
    result = tensor[:, 1:].reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_convnext(tensor):
    tensor = tensor.transpose(2, 3).transpose(1, 2)
    return tensor


MODEL_RESHAPE_TRANSFORMS = {
    'vit': reshape_transform_vit,
    'vit-large': reshape_transform_vit_large,
    'eva': reshape_transform_eva,
    'swinv2': reshape_transform_swinv2,
    'convnext': reshape_transform_convnext
}

def get_gradcam_layer_transform(model_type, model_gradcam):
    if model_type == 'vit':
        target_layers = [model_gradcam.norm]
    elif model_type == 'vit-large':
        target_layers = [model_gradcam.blocks[-1].mlp.fc2]
    elif model_type == 'eva':
        target_layers = [model_gradcam.norm]
    elif model_type == 'swinv2':
        target_layers = [model_gradcam.layers[-1].blocks[-1].norm2]
    else:
        target_layers = [model_gradcam.stages[-1].blocks[-1].mlp.fc2]
    return MODEL_RESHAPE_TRANSFORMS[model_type], target_layers


METHOD_MAP = \
    {"gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "highrescam": HiResCAM}


CACHED_CAM_MODULE = {}


def apply_gradcam(input_tensor: torch.Tensor, targets=Union[List, int, ClassifierOutputTarget], method='gradcam', model_type='eva', eigen_smooth=False, aug_smooth=False, batch_size=32, device='cuda'):
    global CACHED_CAM_MODULE
    method_w_model = method + '-' + model_type
    if method_w_model not in CACHED_CAM_MODULE:

        if method not in METHOD_MAP:
            raise Exception(f"method should be one of {list(METHOD_MAP.keys())}")

        repo_id = MODEL_REPO_MAP[model_type]
        model_gradcam: nn.Module = timm.create_model("hf-hub:" + repo_id).eval()
        model_gradcam.load_state_dict(timm.models.load_state_dict_from_hf(repo_id))
        model_gradcam = model_gradcam.to(device=device)
        reshape_transform, target_layers = get_gradcam_layer_transform(model_type, model_gradcam)

        if method not in METHOD_MAP:
            raise Exception(f"Method {method} not implemented")

        if method == "ablationcam":
            cam = METHOD_MAP[method](model=model_gradcam,
                                        target_layers=target_layers,
                                        reshape_transform=reshape_transform,
                                        ablation_layer=AblationLayerVit())
        else:
            cam = METHOD_MAP[method](model=model_gradcam,
                                        target_layers=target_layers,
                                        reshape_transform=reshape_transform)
        CACHED_CAM_MODULE[method_w_model] = cam

    cam = CACHED_CAM_MODULE[method_w_model]
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = batch_size

    if isinstance(targets, (int, np.int64, np.int32)):
        targets = [ClassifierOutputTarget(targets)]
    elif isinstance(targets, ClassifierOutputTarget):
        targets = [targets]
    elif isinstance(targets, list):
        targets = [ClassifierOutputTarget(i) if isinstance(i, (int, np.int64, np.int32)) else i for i in targets]

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=eigen_smooth,
                        aug_smooth=aug_smooth)

    return grayscale_cam