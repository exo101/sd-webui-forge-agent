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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

"""
To use this python script, first install yt-dlp by:

pip install -U yt-dlp
"""


def download_video(video_url, video_path):
    download_video_command = f"yt-dlp -f bestvideo+bestaudio --skip-unavailable-fragments --merge-output-format mp4 '{video_url}' --output '{video_path}' --external-downloader aria2c --external-downloader-args '-x 16 -k 1M'"
    try:
        subprocess.run(download_video_command, shell=True)  # ignore_security_alert_wait_for_fix RCE
    except KeyboardInterrupt:
        print("Stopped")
        exit()
    except:
        print(f"Error downloading video {video_url}")


def download_videos(num_workers, video_urls, video_paths):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(download_video, video_urls, video_paths)


def extract_vid(video_url):
    if "clip" in video_url:
        print(f"Cannot download youtube clip video: {video_url}")
        return None
    elif "watch?v=" in video_url:  # ignore_security_alert_wait_for_fix RCE
        return video_url.split("watch?v=")[1][:11]
    elif "shorts/" in video_url:
        return video_url.split("shorts/")[1][:11]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1][:11]
    elif "&v=" in video_url:
        return video_url.split("&v=")[1][:11]
    elif "bilibili.com/video/" in video_url:
        return video_url.split("bilibili.com/video/")[1][:12]
    elif "douyin.com/video/" in video_url:
        return video_url.split("douyin.com/video/")[1][:19]
    elif "douyin.com/user/self?modal_id=" in video_url:
        return video_url.split("douyin.com/user/self?modal_id=")[1][:19]
    else:
        print(f"Invalid video url: {video_url}")
        return None


def main(urls_txt_path, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)

    with open(urls_txt_path, "r") as file:
        # Read lines into a list and strip newline characters
        all_video_urls = [line.strip() for line in file]

    video_paths = []
    video_urls = []

    print("Extracting vid...")
    for video_url in tqdm(all_video_urls):
        vid = extract_vid(video_url)
        if vid is None:
            continue
        video_path = os.path.join(output_dir, f"vid_{vid}.mp4")
        if os.path.isfile(video_path):
            continue
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_paths.append(video_path)
        video_urls.append(video_url)

    if len(video_paths) == 0:
        print("All videos have been downloaded")
        exit()
    else:
        print(f"Downloading {len(video_paths)} videos")

    download_videos(num_workers, video_urls, video_paths)


if __name__ == "__main__":
    urls_txt_path = "video_urls.txt"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/youtube/raw"
    num_workers = 50

    maximum_duration = 60 * 30  # set video maximum duration as 30 minutes

    main(urls_txt_path, output_dir, num_workers)
