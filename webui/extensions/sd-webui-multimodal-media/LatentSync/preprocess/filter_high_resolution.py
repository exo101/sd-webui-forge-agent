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

import mediapipe as mp
from latentsync.utils.util import read_video
import os
import tqdm
import shutil
from multiprocessing import Pool

paths = []


def gather_video_paths(input_dir, output_dir, resolution):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output, resolution])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_video_paths(os.path.join(input_dir, video), os.path.join(output_dir, video), resolution)


class FaceDetector:
    def __init__(self, resolution=256):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.resolution = resolution

    def detect_face(self, image):
        height, width = image.shape[:2]
        # Process the image and detect faces.
        results = self.face_detection.process(image)

        if not results.detections:  # Face not detected
            raise Exception("Face not detected")

        if len(results.detections) != 1:
            return False
        detection = results.detections[0]  # Only use the first face in the image

        bounding_box = detection.location_data.relative_bounding_box
        face_width = int(bounding_box.width * width)
        face_height = int(bounding_box.height * height)
        if face_width < self.resolution or face_height < self.resolution:
            return False
        return True

    def detect_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        if len(video_frames) == 0:
            return False
        for frame in video_frames:
            if not self.detect_face(frame):
                return False
        return True

    def close(self):
        self.face_detection.close()


def filter_video(video_input, video_out, resolution):
    if os.path.isfile(video_out):
        return
    face_detector = FaceDetector(resolution)
    try:
        save = face_detector.detect_video(video_input)
    except Exception as e:
        # print(f"Exception: {e} Input video: {video_input}")
        face_detector.close()
        return
    if save:
        os.makedirs(os.path.dirname(video_out), exist_ok=True)
        shutil.copy(video_input, video_out)
    face_detector.close()


def multi_run_wrapper(args):
    return filter_video(*args)


def filter_high_resolution_multiprocessing(input_dir, output_dir, resolution, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_video_paths(input_dir, output_dir, resolution)

    print(f"Filtering high resolution videos in {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/raw"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li//VoxCeleb2/high_resolution"
    resolution = 256
    num_workers = 50

    filter_high_resolution_multiprocessing(input_dir, output_dir, resolution, num_workers)
