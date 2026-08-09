from latentsync.utils.util import count_video_time
from tqdm import tqdm


def count_total_videos_time(fileslist_path: str):
    with open(fileslist_path, "r") as f:
        filepaths = f.readlines()

    # Remove trailing newline characters
    filepaths = [filepath.strip() for filepath in filepaths]

    total_videos_time = 0
    for filepath in tqdm(filepaths):
        total_videos_time += count_video_time(filepath)

    print(f"Fileslist path: {fileslist_path}")
    print(f"Total videos time: {round(total_videos_time/3600)} hours")


if __name__ == "__main__":
    fileslist_path = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/data_v10_core.txt"
    count_total_videos_time(fileslist_path)
