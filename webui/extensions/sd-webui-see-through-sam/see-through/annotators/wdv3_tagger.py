from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F


MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "eva": "SmilingWolf/wd-eva02-large-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3"
}

tagger_and_transform = {}


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tagger_and_transform(model_type: str, device='cuda'):
    global tagger_and_transform
    assert model_type in MODEL_REPO_MAP
    if model_type in tagger_and_transform:
        return tagger_and_transform[model_type]['model'], tagger_and_transform[model_type]['transform'], tagger_and_transform[model_type]['labels']

    repo_id = MODEL_REPO_MAP.get(model_type)
    model: nn.Module = timm.create_model("hf-hub:" + repo_id).eval().to(device=device)
    state_dict = timm.models.load_state_dict_from_hf(repo_id)
    model.load_state_dict(state_dict)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    labels: LabelData = load_labels_hf(repo_id=repo_id)
    tagger_and_transform[model_type] = {
        'model': model,
        'transform': transform,
        'labels': labels

    }
    return model, transform, labels


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
    exclude_cls = None
):
    if exclude_cls is None:
        exclude_cls = {}
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [(probs[i][0], probs[i][1], i) for i in labels.general]
    _gen_labels = {}
    for l in gen_labels:
        if l[1] > gen_threshold and l[0] not in exclude_cls:
            _gen_labels[l[0]] = (l[1], l[2])
    # gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    # gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))
    gen_labels = _gen_labels
    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ")

    return caption, taglist, rating_labels, char_labels, gen_labels


def apply_wdv3_tagger(
    img_input,
    model_type='eva',
    device='cuda',
    gen_threshold = 0.35,
    char_threshold = 0.75,
    exclude_cls = None
):
    model, transform, labels = get_tagger_and_transform(model_type, device=device)
    inputs: Tensor = transform(img_input).unsqueeze(0)
    # NCHW image RGB to BGR
    inputs = inputs[:, [2, 1, 0]]

    with torch.inference_mode():
        inputs = inputs.to(device=device)
        # run the model
        outputs = model.forward(inputs)
        # apply the final activation function (timm doesn't support doing this internally)
        outputs = F.sigmoid(outputs)
        # move inputs, outputs, and model back to to cpu if we were on GPU

        # inputs = inputs.to("cpu")
        outputs = outputs.to("cpu")
            # model = model.to("cpu")

    caption, taglist, ratings, character, general = get_tags(
        probs=outputs.squeeze(0),
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
        exclude_cls=exclude_cls
    )
    return caption, taglist, ratings, character, general
