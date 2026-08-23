from functools import partial
from typing import Optional, Dict, Union

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from PIL import Image
import numpy as np


from .util.transform import Resize, NormalizeImage, PrepareForNet
from .dpt import DepthAnythingV2, DPTHead
from utils.torch_utils import zero_module


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # deprecate("scale", "1.0.0", deprecation_message)
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor



class ControlNeXtModel(nn.Module):

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 3,
        downblock_channels = (128, 256, 512),
        conv_out_channels = (256, 256, 256),
        with_down_res_samples: bool = True,
        with_image_encoder: bool = True,
    ):
        super().__init__()

        if with_image_encoder:
            self.embedding = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.GroupNorm(2, 128),
                nn.ReLU(),
                nn.Conv2d(128, downblock_channels[0], kernel_size=3, padding=1, stride=2)
            )
        else:
            self.embedding = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, downblock_channels[0], kernel_size=3, padding=1, stride=1)
            )
        self.with_down_res_samples = with_down_res_samples

        self.res_block = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        if with_down_res_samples:
            self.out_convs = nn.ModuleList()

        down_in_channels = list(downblock_channels)[:-1]
        down_in_channels = [down_in_channels[0]] + down_in_channels

        for i in range(len(downblock_channels)):
            assert downblock_channels[i] % 32 == 0
            self.res_block.append(
                ResnetBlock2D(
                    in_channels=down_in_channels[i],
                    out_channels=downblock_channels[i],
                    groups=downblock_channels[i] // 32
                ),
            )
            if i < 5:
                self.down_sample.append(
                    Downsample2D(
                        downblock_channels[i],
                        use_conv=True,
                        out_channels=downblock_channels[i],
                        padding=1,
                        name="op",
                    )
                )
            else:
                self.down_sample.append(nn.Identity())
            if with_down_res_samples:
                self.out_convs.append(
                    nn.Conv2d(
                        in_channels=downblock_channels[i],
                        out_channels=conv_out_channels[i],
                        kernel_size=3,
                        padding=1
                    )
                )

        last_down_channels = downblock_channels[-1]
        
        self.mid_convs = nn.ModuleList()
        self.mid_convs = nn.Sequential(
            ResnetBlock2D(
                in_channels=last_down_channels,
                out_channels=last_down_channels,
                groups=last_down_channels // 32
            ),
        )
        self.conv_out = nn.Conv2d(
                            in_channels=last_down_channels,
                            out_channels=conv_out_channels[-1],
                            kernel_size=3,
                            padding=1,
                        )

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> Dict:

        output = []

        sample = self.embedding(sample)

        for ii, (res, downsample) in enumerate(zip(self.res_block, self.down_sample)):
            sample = res(sample)
            if self.with_down_res_samples:
                output.append(self.out_convs[ii](sample))
            sample = downsample(sample)

        for res in self.mid_convs:
            sample = res(sample)
        mid_block_res_sample = self.conv_out(sample)
        output.append(mid_block_res_sample)
        return output

    def init_weights(self):
        if self.with_down_res_samples:
            for module in self.out_convs:
                zero_module(module)
        zero_module(self.conv_out)



def add_signal(src, signal):
    sh, sw = src.shape[-2:]
    if signal.shape[-2] != sh or signal.shape[-1] != sw:
        signal = torch.nn.functional.interpolate(signal, (sh, sw), mode='bilinear')
    return src + signal



class DPTHeadAdapter(nn.Module):
    def __init__(
        self, 
        head_ori: DPTHead,
        init_from_model=False,
        control_nchannels=4,
        add_signal_to_refine=True
    ):
        super(DPTHeadAdapter, self).__init__()
        self.head_ori = head_ori
        self.use_clstoken = self.head_ori.use_clstoken
        self.control_nchannels = control_nchannels        

        self.controlnet = ControlNeXtModel(control_nchannels)
        self.add_signal_to_refine = add_signal_to_refine

    def init_weights(self):
        self.controlnet.init_weights()
    
    def forward(self, out_features, patch_h, patch_w, control_input=None):
        out = []
        out_h, out_w = patch_h * 8, patch_w * 8
        bsz = out_features[0][0].shape[0]
        dtype = out_features[0][0].dtype
        device = out_features[0][0].device
        if control_input is None:
            control_input = torch.zeros((bsz, self.control_nchannels, out_h, out_w), dtype=dtype, device=device)
        else:
            control_input = torch.nn.functional.interpolate(control_input, (out_h, out_w), mode='nearest')

        control_signals = self.controlnet(control_input)

        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.head_ori.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.head_ori.projects[i](x)
            x = self.head_ori.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        # ps*4@256, ps*2@512, ps@1024, ps/2@1024

        layer_1_rn = self.head_ori.scratch.layer1_rn(layer_1)
        if not self.add_signal_to_refine:
            layer_1_rn = add_signal(layer_1_rn, control_signals[0])

        layer_2_rn = self.head_ori.scratch.layer2_rn(layer_2)
        if not self.add_signal_to_refine:
            layer_2_rn = add_signal(layer_2_rn, control_signals[1])

        layer_3_rn = self.head_ori.scratch.layer3_rn(layer_3)
        if not self.add_signal_to_refine:
            layer_3_rn = add_signal(layer_3_rn, control_signals[2])

        layer_4_rn = self.head_ori.scratch.layer4_rn(layer_4)
        layer_4_rn = add_signal(layer_4_rn, control_signals[3])

        path_4 = self.head_ori.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        if self.add_signal_to_refine:
            path_4 = add_signal(path_4, control_signals[2])

        path_3 = self.head_ori.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        if self.add_signal_to_refine:
            path_3 = add_signal(path_3, control_signals[1])

        path_2 = self.head_ori.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        if self.add_signal_to_refine:
            path_2 = add_signal(path_2, control_signals[0])

        path_1 = self.head_ori.scratch.refinenet1(path_2, layer_1_rn)
        # ps*4@256, ps*2@256, ps@256, ps/2@256
        # ps*8@256, ps*4@256, ps*2@256, ps@256
        
        out = self.head_ori.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.head_ori.scratch.output_conv2(out)
        
        return out



model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class ExtendDepthAnythingV2(nn.Module):
    def __init__(
        self, 
        depth_model: DepthAnythingV2=None,
        encoder='vitl', 
    ):
        super(ExtendDepthAnythingV2, self).__init__()
        init_from_model = depth_model is not None

        init_args = model_configs[encoder]
        if depth_model is None:
            depth_model = DepthAnythingV2(init_args)
        else:
            encoder = depth_model.encoder
            init_args = model_configs[encoder]

        # self.depth_model = depth_model
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.backbone_ori = depth_model.pretrained
        del depth_model.pretrained
        self.head_ori = depth_model.depth_head
        del depth_model.depth_head
        del depth_model
        
        self.head_adapter = DPTHeadAdapter(self.head_ori, init_from_model=init_from_model)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def init_adapters(self):
        self.head_adapter.init_weights()

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.backbone_ori.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.head_adapter(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='upper_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        if isinstance(raw_image, Image.Image):
            raw_image = np.array(raw_image)
        
        h, w = raw_image.shape[:2]
        
        image = raw_image / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
