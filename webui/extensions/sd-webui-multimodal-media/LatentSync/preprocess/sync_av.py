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
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from eval.eval_sync_conf import syncnet_eval
import torch
import subprocess
import shutil
from multiprocessing import Process

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


def adjust_offset(video_input: str, video_output: str, av_offset: int, fps: int = 25):
    command = f"ffmpeg -loglevel error -y -i {video_input} -itsoffset {av_offset/fps} -i {video_input} -map 0:v -map 1:a -c copy -q:v 0 -q:a 0 {video_output}"
    subprocess.run(command, shell=True)


def func(sync_conf_threshold, paths, device_id, process_temp_dir):
    os.makedirs(process_temp_dir, exist_ok=True)
    device = f"cuda:{device_id}"

    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters("checkpoints/auxiliary/syncnet_v2.model")

    detect_results_dir = os.path.join(process_temp_dir, "detect_results")
    syncnet_eval_results_dir = os.path.join(process_temp_dir, "syncnet_eval_results")

    syncnet_detector = SyncNetDetector(device=device, detect_results_dir=detect_results_dir)

    for video_input, video_output in paths:
        try:
            av_offset, conf = syncnet_eval(
                syncnet, syncnet_detector, video_input, syncnet_eval_results_dir, detect_results_dir
            )

            if conf >= sync_conf_threshold and abs(av_offset) <= 6:
                os.makedirs(os.path.dirname(video_output), exist_ok=True)
                if av_offset == 0:
                    shutil.copy(video_input, video_output)
                else:
                    adjust_offset(video_input, video_output, av_offset)
        except Exception as e:
            print(e)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def sync_av_multi_gpus(input_dir, output_dir, temp_dir, num_workers, sync_conf_threshold):
    gather_paths(input_dir, output_dir)
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found")
    split_paths = list(split(paths, num_workers * num_devices))
    processes = []

    for i in range(num_devices):
        for j in range(num_workers):
            process_index = i * num_workers + j
            process = Process(
                target=func,
                args=(
                    sync_conf_threshold,
                    split_paths[process_index],
                    i,
                    os.path.join(temp_dir, f"process_{process_index}"),
                ),
            )
            process.start()
            processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/affine_transformed"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/av_synced"
    temp_dir = "temp"
    num_workers = 20  # How many processes per device
    sync_conf_threshold = 3

    sync_av_multi_gpus(input_dir, output_dir, temp_dir, num_workers, sync_conf_threshold)
