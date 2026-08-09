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
import shutil
from tqdm import tqdm

paths = []


def gather_paths(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, output_dir])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def main(input_dir, output_dir):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)

    for video_input, output_dir in tqdm(paths):
        shutil.move(video_input, output_dir)


if __name__ == "__main__":
    # from input_dir to output_dir
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/willdata2"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/willdata"

    main(input_dir, output_dir)
