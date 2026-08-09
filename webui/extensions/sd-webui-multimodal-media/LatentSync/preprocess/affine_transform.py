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

from latentsync.utils.util import write_video
from latentsync.utils.image_processor import VideoProcessor
import torch
import os
import subprocess
from multiprocessing import Process
import shutil

paths = []


def gather_video_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append((video_input, video_output))
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_video_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def combine_video_audio(video_frames, video_input_path, video_output_path, process_temp_dir):
    video_name = os.path.basename(video_input_path)[:-4]
    audio_temp = os.path.join(process_temp_dir, f"{video_name}_temp.wav")
    video_temp = os.path.join(process_temp_dir, f"{video_name}_temp.mp4")

    write_video(video_temp, video_frames, fps=25)

    command = f"ffmpeg -y -loglevel error -i {video_input_path} -q:a 0 -map a {audio_temp}"
    subprocess.run(command, shell=True)

    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
    command = f"ffmpeg -y -loglevel error -i {video_temp} -i {audio_temp} -c:v libx264 -c:a aac -map 0:v -map 1:a -q:v 0 -q:a 0 {video_output_path}"
    subprocess.run(command, shell=True)

    os.remove(audio_temp)
    os.remove(video_temp)


def func(paths, process_temp_dir, device_id, resolution):
    os.makedirs(process_temp_dir, exist_ok=True)
    video_processor = VideoProcessor(resolution, f"cuda:{device_id}")

    for video_input, video_output in paths:
        if os.path.isfile(video_output):
            continue
        try:
            video_frames = video_processor.affine_transform_video(video_input)
        except Exception as e:  # Handle the exception of face not detcted
            print(f"Exception: {e} - {video_input}")
            continue

        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        combine_video_audio(video_frames, video_input, video_output, process_temp_dir)
        print(f"Saved: {video_output}")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def affine_transform_multi_gpus(input_dir, output_dir, temp_dir, resolution, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_video_paths(input_dir, output_dir)
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    split_paths = list(split(paths, num_workers * num_devices))

    processes = []

    for i in range(num_devices):
        for j in range(num_workers):
            process_index = i * num_workers + j
            process = Process(
                target=func, args=(split_paths[process_index], os.path.join(temp_dir, f"process_{i}"), i, resolution)
            )
            process.start()
            processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/segmented"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/affine_transformed"
    temp_dir = "temp"
    resolution = 256
    num_workers = 10  # How many processes per device

    affine_transform_multi_gpus(input_dir, output_dir, temp_dir, resolution, num_workers)
