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
import matplotlib.pyplot as plt


class Chart:
    def __init__(self):
        self.loss_list = []

    def add_ckpt(self, ckpt_path, line_name):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        train_step_list = ckpt["train_step_list"]
        train_loss_list = ckpt["train_loss_list"]
        val_step_list = ckpt["val_step_list"]
        val_loss_list = ckpt["val_loss_list"]
        self.loss_list.append((line_name, train_step_list, train_loss_list, val_step_list, val_loss_list))

    def draw(self, save_path, plot_val=True):
        # Global settings
        plt.rcParams["font.size"] = 14
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Lucida Grande"]
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

        # Creating the plot
        plt.figure(figsize=(7.766, 4.8))  # Golden ratio
        for loss in self.loss_list:
            if plot_val:
                (line,) = plt.plot(loss[1], loss[2], label=loss[0], linewidth=0.5, alpha=0.5)
                line_color = line.get_color()
                plt.plot(loss[3], loss[4], linewidth=1.5, color=line_color)
            else:
                plt.plot(loss[1], loss[2], label=loss[0], linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        legend = plt.legend()
        # legend = plt.legend(loc='upper right', bbox_to_anchor=(1, 0.82))

        # Adjust the linewidth of legend
        for line in legend.get_lines():
            line.set_linewidth(2)

        plt.savefig(save_path, transparent=True)
        plt.close()


if __name__ == "__main__":
    chart = Chart()
    chart.add_ckpt("output/syncnet/train-2024_10_28-23:16:40/checkpoints/checkpoint-20000.pt", "Wav2Lip SyncNet")
    chart.add_ckpt("output/syncnet/train-2024_10_29-20:13:43/checkpoints/checkpoint-20000.pt", "StableSyncNet")
    chart.draw("ablation.pdf", plot_val=True)
