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


def remove_outdated_files(input_dir, begin_date, end_date):
    # Remove files from a specific time period
    for subdir in os.listdir(input_dir):
        if subdir >= begin_date and subdir <= end_date:
            subdir_path = os.path.join(input_dir, subdir)
            command = f"rm -rf {subdir_path}"
            subprocess.run(command, shell=True)
            print(f"Deleted: {subdir_path}")


if __name__ == "__main__":
    input_dir = "/mnt/bn/video-datasets/output/unet"
    begin_date = "train-2024_05_29-12:22:35"
    end_date = "train-2024_09_26-00:10:46"

    remove_outdated_files(input_dir, begin_date, end_date)
