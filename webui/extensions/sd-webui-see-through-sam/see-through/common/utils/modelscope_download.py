"""Download See-Through models from ModelScope into a stable local directory."""
import os


MODELSCOPE_MODELS = {
    "layerdiff": "ljsabc/seethroughv0.0.2_layerdiff3d_nf4",
    "depth": "ljsabc/seethroughv0.0.1_marigold_nf4",
}


def ensure_modelscope_model(repo_id: str, local_root: str, kind: str) -> str:
    """Return a local model path, downloading from ModelScope when necessary."""
    if os.path.isdir(repo_id):
        return repo_id
    model_name = repo_id.split("/", 1)[-1]
    target = os.path.join(local_root, model_name)

    def is_model_dir(path: str) -> bool:
        """识别 ModelScope 直接目录和旧版 models--*/snapshots 目录。"""
        if not os.path.isdir(path):
            return False
        candidates = [path]
        snapshots = os.path.join(path, "snapshots")
        if os.path.isdir(snapshots):
            candidates.extend(
                os.path.join(snapshots, name)
                for name in os.listdir(snapshots)
            )
        for candidate in candidates:
            if os.path.isdir(os.path.join(candidate, "unet")) and (
                os.path.isdir(os.path.join(candidate, "trans_vae"))
                or "Marigold" in kind
                or "marigold" in model_name.lower()
            ):
                return candidate
        return None

    # 兼容下载器旧版本错误使用 models--24yearsold-- 前缀的目录。
    legacy = os.path.join(local_root, f"models--24yearsold--{model_name}")
    for candidate in (target, legacy):
        if is_model_dir(candidate):
            print(f"[See-Through] 使用已有模型: {candidate}", flush=True)
            return candidate

    os.makedirs(target, exist_ok=True)
    print(f"[See-Through] 正在从 ModelScope 下载 {kind} 模型: {repo_id}")
    try:
        # ModelScope 包没有 __main__.py，不能使用 python -m modelscope。
        # 直接调用官方 Python API，兼容 WebUI 内置 Python 环境。
        from modelscope import snapshot_download
        snapshot_download(model_id=repo_id, local_dir=target)
    except Exception as e:
        raise RuntimeError(f"ModelScope 下载失败: {repo_id}: {e}") from e
    return target
