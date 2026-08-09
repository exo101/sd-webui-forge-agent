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

from tqdm import tqdm
from latentsync.utils.util import gather_video_paths_recursively


class FileslistWriter:
    def __init__(self, fileslist_path: str):
        self.fileslist_path = fileslist_path
        with open(fileslist_path, "w") as _:
            pass

    def append_dataset(self, dataset_dir: str):
        print(f"Dataset dir: {dataset_dir}")
        video_paths = gather_video_paths_recursively(dataset_dir)
        with open(self.fileslist_path, "a") as f:
            for video_path in tqdm(video_paths):
                f.write(f"{video_path}\n")


if __name__ == "__main__":
    fileslist_path = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/data_v10_core.txt"

    writer = FileslistWriter(fileslist_path)
    writer.append_dataset("/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/high_visual_quality/train")
    writer.append_dataset("/mnt/bn/maliva-gen-ai-v2/chunyu.li/HDTF/high_visual_quality/train")
