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

import torch
import os
import torch.multiprocessing as mp
import time


def check_mem(cuda_device):
    devices_info = (
        os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader')
        .read()
        .strip()
        .split("\n")
    )
    total, used = devices_info[int(cuda_device)].split(",")
    return total, used


def loop(cuda_device):
    cuda_i = torch.device(f"cuda:{cuda_device}")
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    while True:
        x = torch.rand(50, 512, 512, dtype=torch.float, device=cuda_i)
        y = torch.rand(50, 512, 512, dtype=torch.float, device=cuda_i)
        time.sleep(0.001)
        x = torch.matmul(x, y)


def main():
    if torch.cuda.is_available():
        num_processes = torch.cuda.device_count()
        processes = list()
        for i in range(num_processes):
            p = mp.Process(target=loop, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
