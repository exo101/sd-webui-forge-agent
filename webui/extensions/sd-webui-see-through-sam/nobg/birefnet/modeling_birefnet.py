import logging
import os
import re
from dataclasses import dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import deform_conv2d
from transformers import SwinConfig
from transformers.models.swin.modeling_swin import SwinBackbone

from ..loss import birefnet_loss
from ..mixin import Revised_Mixin
from ..utils import model_card_template

logger = logging.getLogger(__name__)

try:
    NOBG_VERSION = version("nobg")
except PackageNotFoundError:  # running from source without an installed dist
    NOBG_VERSION = "0.0.0+unknown"

BIREFNET_CITATION = """@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume={3},
  pages={9150038},
  year={2024},
  url={https://arxiv.org/abs/2401.03407},
}"""


@dataclass
class BiRefNetConfig:
    """Configuration for BiRefNet (Bilateral Reference Network) with Swin backbone."""

    image_size: int = 1024
    patch_size: int = 4
    embed_dim: int = 192
    num_layers: int = 4
    depths: list = field(default_factory=lambda: [2, 2, 18, 2])
    num_heads: list = field(default_factory=lambda: [6, 12, 24, 48])
    window_size: int = 12
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    dec_channels_inter: int = 64
    use_multi_scale_input: bool = True
    use_gradient_attention: bool = True
    use_image_patch_injection: bool = True
    nobg_version: str = NOBG_VERSION

    def __post_init__(self):
        if len(self.depths) != self.num_layers:
            raise ValueError(
                f"depths has {len(self.depths)} entries but num_layers={self.num_layers}"
            )
        if len(self.num_heads) != self.num_layers:
            raise ValueError(
                f"num_heads has {len(self.num_heads)} entries but num_layers={self.num_layers}"
            )


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        ks = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * ks[0] * ks[1],
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        assert self.offset_conv.bias is not None
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * ks[0] * ks[1],
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        assert self.modulator_conv.bias is not None
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = 256
        parallel_block_sizes = [1, 3, 7]

        self.aspp1 = _ASPPModuleDeformable(in_channels, inter_channels, 1, padding=0)
        self.aspp_deforms = nn.ModuleList(
            [
                _ASPPModuleDeformable(
                    in_channels, inter_channels, conv_size, padding=conv_size // 2
                )
                for conv_size in parallel_block_sizes
            ]
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, inter_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            inter_channels * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


def image2patches(image, patch_ref):
    grid_h = image.shape[-2] // patch_ref.shape[-2]
    grid_w = image.shape[-1] // patch_ref.shape[-1]
    B, C, H, W = image.shape
    h = H // grid_h
    w = W // grid_w
    x = image.view(B, C, grid_h, h, grid_w, w)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(B, C * grid_h * grid_w, h, w)
    return x


class Decoder(nn.Module):
    """Generic BiRefNet decoder over `num_layers` stages.

    `channels` is deep-to-shallow: channels[0] is the (squeezed) deepest
    feature width, channels[-1] the shallowest. Stage i of the loop consumes
    channels[i] (+ optional image-patch injection) and produces channels[i+1],
    except the last stage which produces channels[-1] // 2.
    """

    def __init__(self, channels: list[int], config: BiRefNetConfig):
        super().__init__()
        inter = config.dec_channels_inter
        n = config.num_layers
        self.num_layers = n
        self.use_gradient_attention = config.use_gradient_attention
        self.use_image_patch_injection = config.use_image_patch_injection

        # Injection output widths: stages 0 and 1 use channels[0] // 8; stage
        # j >= 2 uses channels[j - 1] // 8; the final full-resolution injection
        # uses channels[-1] // 8.
        ipt_out = [channels[0] // 8] + [
            channels[max(j - 1, 0)] // 8 for j in range(1, n)
        ]
        ipt_out.append(channels[-1] // 8)

        if config.use_image_patch_injection:
            # image2patches gives 3 * grid^2 channels; the grid at stage j is
            # patch_size * 2**(n - 1 - j), and 1 at full resolution.
            ipt_in = [
                3 * (config.patch_size * 2 ** (n - 1 - j)) ** 2 for j in range(n)
            ] + [3]
            self.ipt_blks = nn.ModuleList(
                [
                    SimpleConvs(ipt_in[j], ipt_out[j], inter_channels=inter)
                    for j in range(n + 1)
                ]
            )

        dec_in = [channels[i] + ipt_out[i] for i in range(n)]
        dec_out = [channels[i + 1] for i in range(n - 1)] + [channels[-1] // 2]
        self.decoder_blocks = nn.ModuleList(
            [BasicDecBlk(dec_in[i], dec_out[i], inter) for i in range(n)]
        )

        self.lateral_blocks = nn.ModuleList(
            [BasicLatBlk(channels[i + 1], channels[i + 1]) for i in range(n - 1)]
        )

        self.conv_ms_spvn = nn.ModuleList(
            [nn.Conv2d(channels[i + 1], 1, 1, 1, 0) for i in range(n - 1)]
        )

        if config.use_gradient_attention:
            _N = 16
            self.gdt_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels[i + 1], _N, 3, 1, 1),
                        nn.BatchNorm2d(_N),
                        nn.ReLU(inplace=True),
                    )
                    for i in range(n - 1)
                ]
            )
            self.gdt_convs_pred = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0)) for _ in range(n - 1)]
            )
            self.gdt_convs_attn = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0)) for _ in range(n - 1)]
            )

        self.conv_out1 = nn.Sequential(
            nn.Conv2d(channels[-1] // 2 + channels[-1] // 8, 1, 1, 1, 0)
        )

    def _inject(self, image, p, blk, size):
        patches = image2patches(image, patch_ref=p)
        patches = F.interpolate(patches, size=size, mode="bilinear", align_corners=True)
        return torch.cat((p, blk(patches)), 1)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        image = features[0]
        feats = features[1:]  # shallow -> deep, length num_layers
        n = self.num_layers
        p = feats[-1]  # deepest (already squeezed)
        outs = []

        for i in range(n - 1):
            if self.use_image_patch_injection:
                p = self._inject(image, p, self.ipt_blks[i], p.shape[2:])
            p = self.decoder_blocks[i](p)
            if self.use_gradient_attention:
                p_gdt = self.gdt_convs[i](p)
                gdt_attn = self.gdt_convs_attn[i](p_gdt).sigmoid()
                p = p * gdt_attn
            outs.append(self.conv_ms_spvn[i](p))
            skip = feats[n - 2 - i]
            p = F.interpolate(
                p, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
            p = p + self.lateral_blocks[i](skip)

        if self.use_image_patch_injection:
            p = self._inject(image, p, self.ipt_blks[n - 1], p.shape[2:])
        p = self.decoder_blocks[n - 1](p)
        p = F.interpolate(p, size=image.shape[2:], mode="bilinear", align_corners=True)
        if self.use_image_patch_injection:
            p = self._inject(image, p, self.ipt_blks[n], image.shape[2:])
        outs.append(self.conv_out1(p))
        return outs


def _remap_legacy_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap the pre-0.2.0 custom-Swin key layout to the current layout.

    Backbone: custom Swin keys -> transformers SwinBackbone keys (the fused
    attn.qkv is split into thirds for q/k/v). Decoder: numbered attributes
    (decoder_block4..1, ipt_blk5..1, ...) -> ModuleList indices.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if "relative_position_index" in k:
            continue  # non-persistent buffer in the new layout
        nk = k
        if k.startswith("bb."):
            rest = k[len("bb.") :]
            if rest.startswith("patch_embed.proj."):
                nk = (
                    "bb.swin.embeddings.patch_embeddings.projection."
                    + rest.rsplit(".", 1)[-1]
                )
            elif rest.startswith("patch_embed.norm."):
                nk = "bb.swin.embeddings.norm." + rest.rsplit(".", 1)[-1]
            elif m := re.match(r"norm(\d+)\.(weight|bias)$", rest):
                nk = f"bb.hidden_states_norms.stage{int(m.group(1)) + 1}.{m.group(2)}"
            elif m := re.match(r"layers\.(\d+)\.blocks\.(\d+)\.(.*)$", rest):
                pre = f"bb.swin.encoder.layers.{m.group(1)}.blocks.{m.group(2)}."
                sub = m.group(3)
                if m2 := re.match(r"attn\.qkv\.(weight|bias)$", sub):
                    q, kk, vv = v.chunk(3, dim=0)
                    out[pre + f"attention.q_proj.{m2.group(1)}"] = q
                    out[pre + f"attention.k_proj.{m2.group(1)}"] = kk
                    out[pre + f"attention.v_proj.{m2.group(1)}"] = vv
                    continue
                elif sub.startswith("attn.proj."):
                    nk = pre + "attention.o_proj." + sub.rsplit(".", 1)[-1]
                elif sub == "attn.relative_position_bias_table":
                    nk = (
                        pre
                        + "attention.relative_position_bias.relative_position_bias_table"
                    )
                elif sub.startswith("norm1."):
                    nk = pre + "layernorm_before." + sub.rsplit(".", 1)[-1]
                elif sub.startswith("norm2."):
                    nk = pre + "layernorm_after." + sub.rsplit(".", 1)[-1]
                else:  # mlp.fc1 / mlp.fc2
                    nk = pre + sub
            elif m := re.match(r"layers\.(\d+)\.downsample\.(.*)$", rest):
                nk = f"bb.swin.encoder.layers.{m.group(1)}.downsample.{m.group(2)}"
        elif k.startswith("decoder."):
            rest = k[len("decoder.") :]
            if m := re.match(r"decoder_block(\d)\.(.*)$", rest):
                nk = f"decoder.decoder_blocks.{4 - int(m.group(1))}.{m.group(2)}"
            elif m := re.match(r"lateral_block(\d)\.(.*)$", rest):
                nk = f"decoder.lateral_blocks.{4 - int(m.group(1))}.{m.group(2)}"
            elif m := re.match(r"ipt_blk(\d)\.(.*)$", rest):
                nk = f"decoder.ipt_blks.{5 - int(m.group(1))}.{m.group(2)}"
            elif m := re.match(r"conv_ms_spvn_(\d)\.(.*)$", rest):
                nk = f"decoder.conv_ms_spvn.{4 - int(m.group(1))}.{m.group(2)}"
            elif m := re.match(r"gdt_convs_pred_(\d)\.(.*)$", rest):
                nk = f"decoder.gdt_convs_pred.{4 - int(m.group(1))}.{m.group(2)}"
            elif m := re.match(r"gdt_convs_attn_(\d)\.(.*)$", rest):
                nk = f"decoder.gdt_convs_attn.{4 - int(m.group(1))}.{m.group(2)}"
            elif m := re.match(r"gdt_convs_(\d)\.(.*)$", rest):
                nk = f"decoder.gdt_convs.{4 - int(m.group(1))}.{m.group(2)}"
        out[nk] = v
    return out


class BiRefNet(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    paper_url="https://arxiv.org/abs/2401.03407",
    license="apache-2.0",
    tags=["nobg", "nobg-birefnet"],
    model_card_template=model_card_template(
        class_name="BiRefNet",
        default_repo="nobg/birefnet",
        citation=BIREFNET_CITATION,
    ),
):
    """Bilateral Reference Network for high-resolution dichotomous image segmentation."""

    def __init__(self, config: BiRefNetConfig | None = None):
        super().__init__()
        self.config = config or BiRefNetConfig()

        swin_config = SwinConfig(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            depths=self.config.depths,
            num_heads=self.config.num_heads,
            window_size=self.config.window_size,
            mlp_ratio=self.config.mlp_ratio,
            drop_path_rate=self.config.drop_path_rate,
            out_features=[f"stage{i + 1}" for i in range(self.config.num_layers)],  # ty: ignore[unknown-argument]
        )
        self.bb = SwinBackbone(swin_config)

        base_channels = [
            self.config.embed_dim * (2**i) for i in range(self.config.num_layers)
        ]

        if self.config.use_multi_scale_input:
            channels = [c * 2 for c in base_channels]
        else:
            channels = base_channels

        # channels = [C1 .. Cn] shallow->deep; the decoder expects deep->shallow
        dec_channels = list(reversed(channels))

        # Squeeze module: deepest feature with all shallower features
        # downsampled and concatenated onto it.
        squeeze_in = channels[-1] + sum(channels[:-1])
        self.squeeze_module = nn.Sequential(
            BasicDecBlk(squeeze_in, dec_channels[0], self.config.dec_channels_inter)
        )

        self.decoder = Decoder(dec_channels, self.config)

        # Loss used when `labels` is passed to forward(). Plain function
        # attribute (not a submodule), so it never enters the state dict and
        # can be hot-swapped: `model.criterion = my_loss`.
        self.criterion = birefnet_loss

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x = pixel_values
        feats = list(self.bb(x).feature_maps)

        if self.config.use_multi_scale_input:
            _, _, H, W = x.shape
            x_half = F.interpolate(
                x, size=(H // 2, W // 2), mode="bilinear", align_corners=True
            )
            feats_half = self.bb(x_half).feature_maps
            feats = [
                torch.cat(
                    [
                        f,
                        F.interpolate(
                            fh, size=f.shape[2:], mode="bilinear", align_corners=True
                        ),
                    ],
                    dim=1,
                )
                for f, fh in zip(feats, feats_half)
            ]

        # Context aggregation: all shallower features downsampled onto the deepest
        deepest = feats[-1]
        context = torch.cat(
            [
                F.interpolate(
                    f, size=deepest.shape[2:], mode="bilinear", align_corners=True
                )
                for f in feats[:-1]
            ]
            + [deepest],
            dim=1,
        )
        feats[-1] = self.squeeze_module(context)

        scaled_preds = self.decoder([pixel_values, *feats])

        logits = scaled_preds[-1]
        if labels is not None:
            loss = self.criterion(scaled_preds, labels)
            return {
                "loss": loss,
                "logits": logits,
                "intermediate_logits": scaled_preds[:-1],
            }
        return {"logits": logits, "intermediate_logits": scaled_preds[:-1]}

    @classmethod
    def from_origin(
        cls,
        origin: str | os.PathLike | nn.Module,
        config: BiRefNetConfig | None = None,
        *,
        token: str | None = None,
        **overrides,
    ) -> "BiRefNet":
        """Build a (possibly re-parameterized) BiRefNet from a previous model.

        `origin` is a Hub repo id, a local directory containing
        `config.json` + `model.safetensors`, or an `nn.Module` instance.
        The origin's config fields are merged with `overrides` (or replaced
        entirely by `config`), the new model is constructed, and every origin
        weight whose (remapped) key and shape match is injected; anything new
        or reshaped keeps its fresh initialization. Understands both the
        current key layout and the pre-0.2.0 custom-Swin layout.
        """
        if isinstance(origin, nn.Module):
            state_dict = origin.state_dict()
            origin_config = getattr(origin, "config", None)
            origin_fields = (
                {f.name: getattr(origin_config, f.name) for f in fields(origin_config)}
                if origin_config is not None
                and hasattr(origin_config, "__dataclass_fields__")
                else {}
            )
        else:
            import json

            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            origin = str(origin)
            if os.path.isdir(origin):
                config_path = os.path.join(origin, "config.json")
                weights_path = os.path.join(origin, "model.safetensors")
            else:
                config_path = hf_hub_download(origin, "config.json", token=token)
                weights_path = hf_hub_download(origin, "model.safetensors", token=token)
            with open(config_path) as f:
                origin_fields = json.load(f)
            state_dict = load_file(weights_path)

        if config is None:
            known = {f.name for f in fields(BiRefNetConfig)}
            merged = {k: v for k, v in origin_fields.items() if k in known}
            merged.update(overrides)
            merged["nobg_version"] = overrides.get("nobg_version", NOBG_VERSION)
            if "depths" in merged and "num_layers" not in overrides:
                merged["num_layers"] = len(merged["depths"])
            config = BiRefNetConfig(**merged)
        elif overrides:
            raise ValueError(
                "pass either an explicit `config` or field overrides, not both"
            )

        model = cls(config)

        is_legacy = any(k.startswith("bb.patch_embed.") for k in state_dict)
        if is_legacy:
            state_dict = _remap_legacy_state_dict(state_dict)

        target = model.state_dict()
        compatible = {}
        skipped_shape = []
        for k, v in state_dict.items():
            if k in target and target[k].shape == v.shape:
                compatible[k] = v
            elif k in target:
                skipped_shape.append(k)
        missing = [k for k in target if k not in compatible]
        model.load_state_dict(compatible, strict=False)

        logger.info(
            "from_origin: injected %d/%d tensors (%d shape-mismatched, "
            "%d fresh-initialized)%s",
            len(compatible),
            len(target),
            len(skipped_shape),
            len(missing),
            " [legacy layout remapped]" if is_legacy else "",
        )
        return model
