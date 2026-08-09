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

import matplotlib.pyplot as plt
from latentsync.utils.util import count_video_time, gather_video_paths_recursively
from tqdm import tqdm


def plot_histogram(data, fig_path):
    # Create histogram
    plt.hist(data, bins=30, edgecolor="black")

    # Add titles and labels
    plt.title("Histogram of Data Distribution")
    plt.xlabel("Video time")
    plt.ylabel("Frequency")

    # Save plot as an image file
    plt.savefig(fig_path)  # Save as PNG file. You can also use 'histogram.jpg', 'histogram.pdf', etc.


def main(input_dir, fig_path):
    video_paths = gather_video_paths_recursively(input_dir)
    video_times = []
    for video_path in tqdm(video_paths):
        video_times.append(count_video_time(video_path))
    plot_histogram(video_times, fig_path)


if __name__ == "__main__":
    input_dir = "validation"
    fig_path = "histogram.png"

    main(input_dir, fig_path)
