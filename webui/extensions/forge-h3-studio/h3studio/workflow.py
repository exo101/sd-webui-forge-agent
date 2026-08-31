from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .errors import WorkflowValidationError

MODE_NAMES = {
    "t2v": "文生视频",
    "i2v": "首帧图生视频",
    "fl2v": "首尾帧视频",
    "ref": "多模态参考",
}


def align_h3_frames(value: int) -> int:
    value = max(5, int(value))
    while value % 17 != 5:
        value += 1
    return value


def _bounded_number(value: Any, minimum: float, maximum: float, label: str, *, integer: bool = False):
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowValidationError(f"{label}不是有效数字") from exc
    if number < minimum or number > maximum:
        raise WorkflowValidationError(f"{label}必须位于 {minimum}–{maximum} 之间")
    return number


def _filename(value: Any, label: str) -> str:
    value = str(value or "").strip().replace("\\", "/")
    if not value or value.startswith("/") or ".." in PurePosixPath(value).parts:
        raise WorkflowValidationError(f"{label}文件路径无效")
    return value


def _model_name(value: Any, label: str) -> str:
    value = str(value or "").strip()
    if not value or "\x00" in value or len(value) > 500:
        raise WorkflowValidationError(f"请选择{label}")
    return value


def _clean_prefix(value: Any) -> str:
    value = str(value or "video/Forge_H3_Studio").replace("\\", "/").strip(" /")
    value = re.sub(r"[^0-9A-Za-z_\-./%:]+", "_", value)
    if not value or ".." in PurePosixPath(value).parts:
        return "video/Forge_H3_Studio"
    return value[:240]


def _seeded_prefix(value: Any, seed: int) -> tuple[str, str]:
    """Return the user prefix and a resolved prefix that always names the actual seed."""
    base = _clean_prefix(value)
    base = re.sub(r"_seed_\d+$", "", base, flags=re.IGNORECASE).rstrip("_-") or "video/Forge_H3_Studio"
    suffix = f"_seed_{seed}"
    base = base[: max(1, 240 - len(suffix))].rstrip("_-") or "video/Forge_H3_Studio"
    return base, f"{base}{suffix}"


def _aligned_dimension(value: int, multiple: int) -> int:
    """Round a dimension to the user-selected multiple without exceeding H3 limits."""
    maximum = (4096 // multiple) * multiple
    return min(maximum, max(multiple, int(round(value / multiple)) * multiple))


def _normalize_crop(
    value: Any,
    label: str,
    target_width: int,
    target_height: int,
) -> dict[str, int] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, dict):
        raise WorkflowValidationError(f"{label}裁剪参数无效")
    source_width = int(_bounded_number(value.get("source_width"), 8, 32768, f"{label}原图宽度", integer=True))
    source_height = int(_bounded_number(value.get("source_height"), 8, 32768, f"{label}原图高度", integer=True))
    x = int(_bounded_number(value.get("x", 0), 0, source_width - 1, f"{label}裁剪 X", integer=True))
    y = int(_bounded_number(value.get("y", 0), 0, source_height - 1, f"{label}裁剪 Y", integer=True))
    width = int(_bounded_number(value.get("width"), 8, source_width, f"{label}裁剪宽度", integer=True))
    height = int(_bounded_number(value.get("height"), 8, source_height, f"{label}裁剪高度", integer=True))
    if x + width > source_width or y + height > source_height:
        raise WorkflowValidationError(f"{label}裁剪框超出原图范围")

    # Keep the selected centre while trimming tiny rounding mismatches to the
    # exact output ratio. This guarantees that ImageScale never stretches it.
    target_ratio = target_width / target_height
    centre_x = x + width / 2
    centre_y = y + height / 2
    if width / height > target_ratio:
        width = max(8, min(width, int(round(height * target_ratio))))
    else:
        height = max(8, min(height, int(round(width / target_ratio))))
    x = max(0, min(source_width - width, int(round(centre_x - width / 2))))
    y = max(0, min(source_height - height, int(round(centre_y - height / 2))))
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
    }


def normalize_request(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise WorkflowValidationError("生成请求格式无效")
    data = deepcopy(raw)
    mode = str(data.get("mode") or "t2v")
    if mode not in MODE_NAMES:
        raise WorkflowValidationError("不支持的 H3 生成模式")
    data["mode"] = mode
    data["model"] = _model_name(data.get("model"), "H3 模型")
    data["text_encoder"] = _model_name(data.get("text_encoder"), "MiniMax 文本编码器")
    data["video_vae"] = _model_name(data.get("video_vae"), "视频 VAE")
    data["audio_vae"] = _model_name(data.get("audio_vae"), "音频 VAE")
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise WorkflowValidationError("提示词不能为空")
    if len(prompt) > 40000:
        raise WorkflowValidationError("提示词过长")
    data["prompt"] = prompt

    width = int(_bounded_number(data.get("width", 1344), 32, 4096, "宽度", integer=True))
    height = int(_bounded_number(data.get("height", 768), 32, 4096, "高度", integer=True))
    multiple = int(_bounded_number(data.get("rounding_multiple", 32), 1, 512, "分辨率倍数", integer=True))
    data["rounding_multiple"] = multiple
    data["width"] = _aligned_dimension(width, multiple)
    data["height"] = _aligned_dimension(height, multiple)
    data["aspect_ratio"] = str(data.get("aspect_ratio") or "custom")[:32]
    data["megapixels"] = _bounded_number(data.get("megapixels", data["width"] * data["height"] / 1_000_000), 0.01, 32, "百万像素")
    requested_frames = int(_bounded_number(data.get("frames", 124), 5, 3600, "帧数", integer=True))
    data["requested_frames"] = requested_frames
    data["frames"] = align_h3_frames(requested_frames)
    data["steps"] = int(_bounded_number(data.get("steps", 30), 1, 200, "采样步数", integer=True))
    data["seed"] = int(_bounded_number(data.get("seed", 0), 0, 2**64 - 1, "随机种子", integer=True))
    data["shift_video"] = _bounded_number(data.get("shift_video", 12), 0.01, 100, "视频 Shift")
    data["shift_audio"] = _bounded_number(data.get("shift_audio", 3), 0.01, 100, "音频 Shift")
    data["denoise"] = _bounded_number(data.get("denoise", 1), 0.01, 1, "Denoise")
    data["sampler"] = str(data.get("sampler") or "euler")[:100]
    data["scheduler"] = str(data.get("scheduler") or "simple")[:100]
    data["weight_dtype"] = str(data.get("weight_dtype") or "default")
    if data["weight_dtype"] not in {"default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"}:
        data["weight_dtype"] = "default"
    data["clip_device"] = "cpu" if data.get("clip_device") == "cpu" else "default"

    first = data.get("first_frame")
    last = data.get("last_frame")
    if mode in {"i2v", "fl2v"} and not first:
        raise WorkflowValidationError("当前模式需要首帧图片")
    if mode == "fl2v" and not last:
        raise WorkflowValidationError("首尾帧模式需要尾帧图片")
    if first:
        data["first_frame"] = _filename(first, "首帧")
        data["first_frame_crop"] = _normalize_crop(
            data.get("first_frame_crop"), "首帧", data["width"], data["height"]
        )
    if last:
        data["last_frame"] = _filename(last, "尾帧")
        data["last_frame_crop"] = _normalize_crop(
            data.get("last_frame_crop"), "尾帧", data["width"], data["height"]
        )
    if mode == "t2v":
        data.pop("first_frame", None)
        data.pop("last_frame", None)
        data.pop("first_frame_crop", None)
        data.pop("last_frame_crop", None)
    elif mode == "i2v":
        data.pop("last_frame", None)
        data.pop("last_frame_crop", None)
    elif mode == "ref":
        data.pop("first_frame", None)
        data.pop("last_frame", None)
        data.pop("first_frame_crop", None)
        data.pop("last_frame_crop", None)

    refs = data.get("references") or []
    if not isinstance(refs, list):
        raise WorkflowValidationError("参考素材格式无效")
    counts = {"image": 0, "video": 0, "audio": 0}
    normalized_refs = []
    for item in refs:
        if not isinstance(item, dict) or item.get("kind") not in counts:
            continue
        kind = item["kind"]
        counts[kind] += 1
        limit = 9 if kind == "image" else 3
        if counts[kind] > limit:
            raise WorkflowValidationError(f"{kind} 参考素材超过 H3 限制")
        normalized_refs.append({
            "kind": kind,
            "file": _filename(item.get("file"), "参考素材"),
            "include_audio": bool(item.get("include_audio", True)),
        })
    if mode == "ref" and not normalized_refs:
        raise WorkflowValidationError("多参考模式至少需要一个参考素材")
    data["references"] = normalized_refs if mode == "ref" else []
    data["ref_image_size"] = "max" if data.get("ref_image_size") == "max" else "match"

    loras = data.get("loras") or []
    if not isinstance(loras, list):
        raise WorkflowValidationError("LoRA 配置格式无效")
    normalized_loras = []
    for item in loras[:16]:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        normalized_loras.append({
            "name": _model_name(item.get("name"), "LoRA"),
            "model_strength": _bounded_number(item.get("model_strength", 1), -4, 4, "LoRA 模型权重"),
            "clip_strength": _bounded_number(item.get("clip_strength", 0), -4, 4, "LoRA 文本权重"),
            "apply_to_clip": bool(item.get("apply_to_clip", False)),
        })
    data["loras"] = normalized_loras

    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    base_prefix, resolved_prefix = _seeded_prefix(output.get("filename_prefix"), data["seed"])
    output["base_filename_prefix"] = base_prefix
    output["filename_prefix"] = resolved_prefix
    output["format"] = str(output.get("format") or "auto")
    if output["format"] not in {"auto", "mp4", "webm", "mkv"}:
        output["format"] = "auto"
    output["codec"] = str(output.get("codec") or "auto")
    if output["codec"] not in {"auto", "h264"}:
        output["codec"] = "auto"
    output["crf"] = int(_bounded_number(output.get("crf", 23), 0, 51, "CRF", integer=True))
    output["bit_depth"] = 10 if int(output.get("bit_depth", 8)) == 10 else 8
    output["fps"] = _bounded_number(output.get("fps", 24), 1, 120, "导出 FPS")
    data["output"] = output
    return data


@dataclass
class WorkflowGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    next_id: int = 1

    def add(self, class_type: str, inputs: dict[str, Any], title: str) -> str:
        node_id = str(self.next_id)
        self.next_id += 1
        self.nodes[node_id] = {
            "inputs": inputs,
            "class_type": class_type,
            "_meta": {"title": title},
        }
        return node_id


def build_h3_workflow(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = normalize_request(request)
    graph = WorkflowGraph()

    def load_prepared_frame(filename: str, crop: dict[str, int] | None, title: str) -> list[Any]:
        loaded = graph.add("LoadImage", {"image": filename}, title)
        image_link: list[Any] = [loaded, 0]
        if crop:
            cropped = graph.add(
                "ImageCrop",
                {
                    "image": image_link,
                    "width": crop["width"],
                    "height": crop["height"],
                    "x": crop["x"],
                    "y": crop["y"],
                },
                f"{title} · 联动裁剪",
            )
            scaled = graph.add(
                "ImageScale",
                {
                    "image": [cropped, 0],
                    "upscale_method": "lanczos",
                    "width": data["width"],
                    "height": data["height"],
                    "crop": "disabled",
                },
                f"{title} · 适配输出尺寸",
            )
            image_link = [scaled, 0]
        return image_link

    model = graph.add(
        "UNETLoader",
        {"unet_name": data["model"], "weight_dtype": data["weight_dtype"]},
        "H3 扩散模型",
    )
    clip = graph.add(
        "CLIPLoader",
        {"clip_name": data["text_encoder"], "type": "minimax", "device": data["clip_device"]},
        "MiniMax H3 文本编码器",
    )
    video_vae = graph.add("VAELoader", {"vae_name": data["video_vae"]}, "H3 视频 VAE")
    audio_vae = graph.add("VAELoader", {"vae_name": data["audio_vae"]}, "H3 音频 VAE")

    model_link: list[Any] = [model, 0]
    clip_link: list[Any] = [clip, 0]
    for index, lora in enumerate(data["loras"], start=1):
        if lora["apply_to_clip"]:
            node = graph.add(
                "LoraLoader",
                {
                    "model": model_link,
                    "clip": clip_link,
                    "lora_name": lora["name"],
                    "strength_model": lora["model_strength"],
                    "strength_clip": lora["clip_strength"],
                },
                f"LoRA {index}: {lora['name']}",
            )
            model_link, clip_link = [node, 0], [node, 1]
        else:
            node = graph.add(
                "LoraLoaderModelOnly",
                {"model": model_link, "lora_name": lora["name"], "strength_model": lora["model_strength"]},
                f"H3 LoRA {index}: {lora['name']}",
            )
            model_link = [node, 0]

    shifted = graph.add(
        "MiniMaxH3SigmaShift",
        {"model": model_link, "shift_video": data["shift_video"], "shift_audio": data["shift_audio"]},
        "H3 视频/音频 Sigma Shift",
    )

    if data["mode"] == "ref":
        cond_inputs: dict[str, Any] = {
            "clip": clip_link,
            "vae": [video_vae, 0],
            "audio_vae": [audio_vae, 0],
            "prompt": data["prompt"],
            "width": data["width"],
            "height": data["height"],
            "length": data["frames"],
            "ref_image_size": data["ref_image_size"],
        }
        counters = {"image": 0, "video": 0, "audio": 0}
        for ref in data["references"]:
            kind = ref["kind"]
            slot = counters[kind]
            counters[kind] += 1
            if kind == "image":
                load = graph.add("LoadImage", {"image": ref["file"]}, f"参考图片 {slot + 1}")
                cond_inputs[f"ref_images.ref_image_{slot}"] = [load, 0]
            elif kind == "video":
                load = graph.add("LoadVideo", {"file": ref["file"]}, f"参考视频 {slot + 1}")
                components = graph.add("GetVideoComponents", {"video": [load, 0]}, f"拆分参考视频 {slot + 1}")
                cond_inputs[f"ref_videos.ref_video_{slot}"] = [components, 0]
                if ref["include_audio"]:
                    cond_inputs[f"ref_video_audios.ref_video_audio_{slot}"] = [components, 1]
            else:
                load = graph.add("LoadAudio", {"audio": ref["file"]}, f"参考音频 {slot + 1}")
                cond_inputs[f"ref_audios.ref_audio_{slot}"] = [load, 0]
        conditioning = graph.add("MiniMaxH3ReferenceToVideo", cond_inputs, "MiniMax H3 多模态参考")
    else:
        cond_inputs = {
            "clip": clip_link,
            "vae": [video_vae, 0],
            "prompt": data["prompt"],
            "width": data["width"],
            "height": data["height"],
            "length": data["frames"],
        }
        if data.get("first_frame"):
            cond_inputs["first_frame"] = load_prepared_frame(
                data["first_frame"], data.get("first_frame_crop"), "首帧"
            )
        if data.get("last_frame"):
            cond_inputs["last_frame"] = load_prepared_frame(
                data["last_frame"], data.get("last_frame_crop"), "尾帧"
            )
        conditioning = graph.add("MiniMaxH3ImageToVideo", cond_inputs, "MiniMax H3 文生/图生视频")

    noise = graph.add("RandomNoise", {"noise_seed": data["seed"]}, "随机噪声")
    guider = graph.add("BasicGuider", {"model": [shifted, 0], "conditioning": [conditioning, 0]}, "H3 Basic Guider")
    sampler = graph.add("KSamplerSelect", {"sampler_name": data["sampler"]}, "采样器")
    sigmas = graph.add(
        "BasicScheduler",
        {"model": [shifted, 0], "scheduler": data["scheduler"], "steps": data["steps"], "denoise": data["denoise"]},
        "调度器",
    )
    sampled = graph.add(
        "SamplerCustomAdvanced",
        {
            "noise": [noise, 0],
            "guider": [guider, 0],
            "sampler": [sampler, 0],
            "sigmas": [sigmas, 0],
            "latent_image": [conditioning, 1],
        },
        "MiniMax H3 采样",
    )
    decoded_video = graph.add("VAEDecode", {"samples": [sampled, 0], "vae": [video_vae, 0]}, "解码视频")
    decoded_audio = graph.add("VAEDecodeAudio", {"samples": [sampled, 0], "vae": [audio_vae, 0]}, "解码音频")
    thumbnail_frame = graph.add(
        "ImageFromBatch",
        {"image": [decoded_video, 0], "batch_index": 0, "length": 1},
        "提取任务缩略图",
    )
    graph.add("PreviewImage", {"images": [thumbnail_frame, 0]}, "ComfyUI 任务缩略图")
    created_video = graph.add(
        "CreateVideo",
        {
            "images": [decoded_video, 0],
            "audio": [decoded_audio, 0],
            "fps": data["output"]["fps"],
            "bit_depth": data["output"]["bit_depth"],
        },
        "合成带音频视频",
    )
    # SaveVideo is a v3 node. DynamicCombo inputs are flattened in API prompts
    # and rebuilt by ComfyUI immediately before execution.
    codec_inputs: dict[str, Any] = {"codec": "auto"}
    if data["output"]["codec"] == "h264":
        codec_inputs = {
            "codec": "h264",
            "codec.encoding": "re-encode",
            "codec.encoding.crf": data["output"]["crf"],
        }
    save = graph.add(
        "SaveVideo",
        {
            "video": [created_video, 0],
            "filename_prefix": data["output"]["filename_prefix"],
            "format": data["output"]["format"],
            **codec_inputs,
        },
        "保存 H3 视频",
    )

    reproducible = deepcopy(data)
    reproducible["output"] = {
        **data["output"],
        "filename_prefix": data["output"]["base_filename_prefix"],
    }
    reproducible["output"].pop("base_filename_prefix", None)

    summary = {
        "mode": data["mode"],
        "mode_name": MODE_NAMES[data["mode"]],
        "frames": data["frames"],
        "requested_frames": data["requested_frames"],
        "duration_seconds": round(data["frames"] / 24, 3),
        "resolution": f"{data['width']}x{data['height']}",
        "width": data["width"],
        "height": data["height"],
        "aspect_ratio": data["aspect_ratio"],
        "megapixels": data["megapixels"],
        "rounding_multiple": data["rounding_multiple"],
        "steps": data["steps"],
        "seed": data["seed"],
        "filename_prefix": data["output"]["filename_prefix"],
        "model": data["model"],
        "text_encoder": data["text_encoder"],
        "video_vae": data["video_vae"],
        "audio_vae": data["audio_vae"],
        "loras": data["loras"],
        "output_node": save,
        "node_count": len(graph.nodes),
        "node_titles": {
            node_id: node.get("_meta", {}).get("title", node.get("class_type", node_id))
            for node_id, node in graph.nodes.items()
        },
        "reproducible": reproducible,
        "warnings": [],
    }
    if data["frames"] != data["requested_frames"]:
        summary["warnings"].append(f"帧数已从 {data['requested_frames']} 自动对齐为 {data['frames']}（17k+5）")
    if not 124 <= data["frames"] <= 362:
        summary["warnings"].append("当前时长超出 H3 官方节点标注的主要训练范围 124–362 帧")
    if data["output"]["fps"] != 24:
        summary["warnings"].append("H3 按 24 FPS 生成；其他导出 FPS 会改变播放速度")
    if data["width"] * data["height"] > 1344 * 768:
        summary["warnings"].append("当前像素面积高于 1344×768，显存占用和生成时间会明显增加")
    if data["mode"] == "ref":
        expected_tags: list[str] = []
        image_count = sum(ref["kind"] == "image" for ref in data["references"])
        video_refs = [ref for ref in data["references"] if ref["kind"] == "video"]
        standalone_audio_count = sum(ref["kind"] == "audio" for ref in data["references"])
        expected_tags.extend(f"<Picture {index}>" for index in range(1, image_count + 1))
        audio_ordinal = 0
        for index, ref in enumerate(video_refs, start=1):
            if ref["include_audio"]:
                audio_ordinal += 1
                expected_tags.append(f"<Audio {audio_ordinal}>")
            expected_tags.append(f"<Video {index}>")
        for _ in range(standalone_audio_count):
            audio_ordinal += 1
            expected_tags.append(f"<Audio {audio_ordinal}>")
        missing_tags = [tag for tag in expected_tags if tag not in data["prompt"]]
        if missing_tags:
            summary["warnings"].append("这些参考锚点尚未出现在提示词中：" + "、".join(missing_tags))
    return graph.nodes, summary
