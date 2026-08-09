# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tqdm
import torch
import torchvision
import shutil
from multiprocessing import Process
import numpy as np
from decord import VideoReader
from einops import rearrange
from eval.hyper_iqa import HyperNet, TargetNet


paths = []


def gather_paths(input_dir, output_dir):
    # os.makedirs(output_dir, exist_ok=True)

    for video in tqdm.tqdm(sorted(os.listdir(input_dir))):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append((video_input, video_output))
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def read_video(video_path: str):
    vr = VideoReader(video_path)
    first_frame = vr[0].asnumpy()
    middle_frame = vr[len(vr) // 2].asnumpy()
    last_frame = vr[-1].asnumpy()
    vr.seek(0)
    video_frames = np.stack([first_frame, middle_frame, last_frame], axis=0)
    video_frames = torch.from_numpy(rearrange(video_frames, "b h w c -> b c h w"))
    video_frames = video_frames / 255.0
    return video_frames


def func(paths, device_id):
    device = f"cuda:{device_id}"

    model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)

    # load the pre-trained model on the koniq-10k dataset
    model_hyper.load_state_dict(
        (torch.load("checkpoints/auxiliary/koniq_pretrained.pkl", map_location=device, weights_only=True))
    )

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for video_input, video_output in paths:
        try:
            video_frames = read_video(video_input)
            video_frames = transforms(video_frames)
            video_frames = video_frames.clone().detach().to(device)
            paras = model_hyper(video_frames)  # 'paras' contains the network weights conveyed to target network

            # Building target network
            model_target = TargetNet(paras).to(device)
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            pred = model_target(paras["target_in_vec"])  # 'paras['target_in_vec']' is the input to target net

            # quality score ranges from 0-100, a higher score indicates a better quality
            quality_score = pred.mean().item()
            print(f"Input video: {video_input}\nVisual quality score: {quality_score:.2f}")

            if quality_score >= 40:
                os.makedirs(os.path.dirname(video_output), exist_ok=True)
                shutil.copy(video_input, video_output)
        except Exception as e:
            print(e)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def filter_visual_quality_multi_gpus(input_dir, output_dir, num_workers):
    gather_paths(input_dir, output_dir)
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found")
    split_paths = list(split(paths, num_workers * num_devices))
    processes = []

    for i in range(num_devices):
        for j in range(num_workers):
            process_index = i * num_workers + j
            process = Process(target=func, args=(split_paths[process_index], i))
            process.start()
            processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/av_synced"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/high_visual_quality"
    num_workers = 20  # How many processes per device

    filter_visual_quality_multi_gpus(input_dir, output_dir, num_workers)
