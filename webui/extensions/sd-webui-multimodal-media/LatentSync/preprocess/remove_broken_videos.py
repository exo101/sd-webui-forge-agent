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
from multiprocessing import Pool
import tqdm

from latentsync.utils.av_reader import AVReader
from latentsync.utils.util import gather_video_paths_recursively


def remove_broken_video(video_path):
    try:
        AVReader(video_path)
    except Exception:
        os.remove(video_path)


def remove_broken_videos_multiprocessing(input_dir, num_workers):
    video_paths = gather_video_paths_recursively(input_dir)

    print("Removing broken videos...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(remove_broken_video, video_paths), total=len(video_paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/raw"
    num_workers = 50

    remove_broken_videos_multiprocessing(input_dir, num_workers)
