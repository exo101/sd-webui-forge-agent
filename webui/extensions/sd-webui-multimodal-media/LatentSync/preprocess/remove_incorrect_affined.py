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
from latentsync.utils.util import read_video, gather_video_paths_recursively
import os
import tqdm
from multiprocessing import Pool


class FaceDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def detect_face(self, image):
        # Process the image and detect faces.
        results = self.face_detection.process(image)

        if not results.detections:  # Face not detected
            return False

        if len(results.detections) != 1:
            return False
        return True

    def detect_video(self, video_path):
        try:
            video_frames = read_video(video_path, change_fps=False)
        except Exception as e:
            print(f"Exception: {e} - {video_path}")
            return False
        if len(video_frames) == 0:
            return False
        for frame in video_frames:
            if not self.detect_face(frame):
                return False
        return True

    def close(self):
        self.face_detection.close()


def remove_incorrect_affined(video_path):
    if not os.path.isfile(video_path):
        return
    face_detector = FaceDetector()
    has_face = face_detector.detect_video(video_path)
    if not has_face:
        os.remove(video_path)
        print(f"Removed: {video_path}")
    face_detector.close()


def remove_incorrect_affined_multiprocessing(input_dir, num_workers):
    video_paths = gather_video_paths_recursively(input_dir)
    print(f"Total videos: {len(video_paths)}")

    print(f"Removing incorrect affined videos in {input_dir} ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(remove_incorrect_affined, video_paths), total=len(video_paths)):
            pass


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/affine_transformed"
    num_workers = 50

    remove_incorrect_affined_multiprocessing(input_dir, num_workers)
