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
import subprocess
import tqdm
from multiprocessing import Pool

paths = []


def gather_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, output_dir])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def detect_shot(video_input, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video = os.path.basename(video_input)[:-4]
    command = f"scenedetect --quiet -i {video_input} detect-adaptive --threshold 2 split-video --filename '{video}_shot_$SCENE_NUMBER' --output {output_dir}"
    # command = f"scenedetect --quiet -i {video_input} detect-adaptive --threshold 2 split-video --high-quality --filename '{video}_shot_$SCENE_NUMBER' --output {output_dir}"
    subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return detect_shot(*args)


def detect_shot_multiprocessing(input_dir, output_dir, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)

    print(f"Detecting shot of {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/high_resolution"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/shot"
    num_workers = 50

    detect_shot_multiprocessing(input_dir, output_dir, num_workers)
