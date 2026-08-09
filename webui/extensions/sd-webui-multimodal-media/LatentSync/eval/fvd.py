# Adapted from https://github.com/universome/fvd-comparison/blob/master/our_fvd.py

from typing import Tuple
import scipy
import numpy as np
import torch
from latentsync.utils.util import check_model_and_download


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


@torch.no_grad()
def compute_our_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str = "cuda") -> float:
    i3d_path = "checkpoints/auxiliary/i3d_torchscript.pt"
    check_model_and_download(i3d_path)
    i3d_kwargs = dict(
        rescale=False, resize=False, return_features=True
    )  # Return raw features before the softmax layer.

    with open(i3d_path, "rb") as f:
        i3d_model = torch.jit.load(f).eval().to(device)

    videos_fake = videos_fake.permute(0, 4, 1, 2, 3).to(device)
    videos_real = videos_real.permute(0, 4, 1, 2, 3).to(device)

    feats_fake = i3d_model(videos_fake, **i3d_kwargs).cpu().numpy()
    feats_real = i3d_model(videos_real, **i3d_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


def main():
    # input shape: (b, f, h, w, c)
    videos_fake = torch.rand(10, 16, 224, 224, 3)
    videos_real = torch.rand(10, 16, 224, 224, 3)

    our_fvd_result = compute_our_fvd(videos_fake, videos_real)
    print(f"[FVD scores] Ours: {our_fvd_result}")


if __name__ == "__main__":
    main()
