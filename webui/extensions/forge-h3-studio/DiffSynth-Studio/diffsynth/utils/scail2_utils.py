import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as TT
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize


def load_image_to_tensor_chw_normalized(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor with shape (1, C, H, W), range [-1, 1]."""
    transform = TT.Compose([TT.ToTensor()])
    image_tensor = transform(image)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)
    return image_tensor


def resize_for_rectangle_crop(arr: torch.Tensor, image_size, reshape_mode="center"):
    """Resize and crop tensor to target size."""
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )
    h, w = arr.shape[2], arr.shape[3]
    delta_h = h - image_size[0]
    delta_w = w - image_size[1]
    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr


def extract_and_compress_mask_to_latent(
    mask_cthw: torch.Tensor,
    additional_spatial_downsample: int = 1,
    temporal_compression_stride: int = 4,
) -> torch.Tensor:
    """
    Convert 3-channel RGB colored segmentation mask to 28-channel binary latent.
    Does NOT go through VAE.
    
    Args:
        mask_cthw: (3, T, H, W), range [-1, 1]
        additional_spatial_downsample: extra spatial downsample factor (1 = no extra)
        temporal_compression_stride: temporal compression factor (4 = 7ch * 4 = 28ch)
    
    Returns:
        (28, T_latent, H_latent, W_latent), values {0, 1}
    """
    C, T, H, W = mask_cthw.shape
    # Threshold: original pixel value >= 225才算"亮"
    _ON_THRESH = (225.0 - 127.5) / 127.5  # ≈ 0.765
    
    mask = mask_cthw.permute(1, 0, 2, 3).float()  # (T, 3, H, W)
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    
    # 7 color channels: white, red, green, blue, yellow, magenta, cyan
    binary_7ch = torch.cat([
        R * G * B,      # white
        R * nG * nB,    # red
        nR * G * nB,    # green
        nR * nG * B,    # blue
        R * G * nB,     # yellow
        R * nG * B,     # magenta
        nR * G * B,     # cyan
    ], dim=1)  # (T, 7, H, W)
    
    # Spatial downsample: VAE stride 8 (3 levels of /2) + additional downsample
    H_lat, W_lat = H, W
    if additional_spatial_downsample > 1:
        H_lat = H_lat // additional_spatial_downsample
        W_lat = W_lat // additional_spatial_downsample
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    
    binary_7ch = F.interpolate(
        binary_7ch, size=(H_lat, W_lat), mode='area'
    )  # area = mean downsample, preserves coverage ratio
    
    # Temporal compression: stride 4
    T_latent = (T - 1) // temporal_compression_stride + 1
    # Wan's causal temporal VAE groups the first latent from the first frame
    # repeated four times, then groups each following four-frame block. Match
    # ComfyUI's _extract_mask_to_28ch exactly; padding at the end misaligns all
    # later pose-mask channels with the encoded video latents.
    padded = torch.cat(
        [binary_7ch[:1].repeat(temporal_compression_stride, 1, 1, 1), binary_7ch[1:]],
        dim=0,
    )
    # Reshape: (T_latent, stride*7, H_lat, W_lat) -> (stride*7, T_latent, H_lat, W_lat)
    out = padded.view(
        T_latent, temporal_compression_stride * 7, H_lat, W_lat
    ).permute(1, 0, 2, 3)
    
    return out  # (28, T_latent, H_latent, W_latent)


def create_animation_ref_mask(h: int, w: int) -> torch.Tensor:
    """
    Create reference mask for animation mode:
    - Reference image mask: white background (background should be visible)
    - This signals the model that the reference image's background should be shown
    """
    # White mask (all channels high)
    mask = torch.ones(3, 1, h, w) * 0.766  # just above threshold
    return mask


def create_animation_pose_mask(t: int, h: int, w: int) -> torch.Tensor:
    """
    Create pose mask for animation mode:
    - Pose video mask: black background (background should NOT be visible)
    - This signals the model to replace the pose video's background
    """
    # Black mask (all channels low)
    mask = torch.zeros(3, t, h, w) - 1.0  # below threshold
    return mask


def create_replacement_ref_mask(h: int, w: int) -> torch.Tensor:
    """
    Create reference mask for replacement mode:
    - Reference image mask: black background (background should NOT be visible)
    - Only the character region should be taken from the reference image
    """
    # Black mask
    mask = torch.zeros(3, 1, h, w) - 1.0
    return mask


def create_replacement_pose_mask(t: int, h: int, w: int) -> torch.Tensor:
    """
    Create pose mask for replacement mode:
    - Pose video mask: white background (background SHOULD be visible)
    - The background from the pose video should be preserved
    """
    # White mask
    mask = torch.ones(3, t, h, w) * 0.766
    return mask
