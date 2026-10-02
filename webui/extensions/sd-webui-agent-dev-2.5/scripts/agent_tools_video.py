# =============================================================================
# Agent Tools — 视频生成工具（MiniMax H3）
# =============================================================================

import os
import tempfile
import time

from PIL import Image


def _get_webui_base_url():
    """自动检测 WebUI 基础 URL，验证 /h3studio/api/bootstrap 端点可用。

    优先使用 cmd_args.cmd_opts.port，然后扫描 7860-7870 端口。
    返回 base url 字符串或 None。
    """
    import httpx

    def _check(base):
        try:
            resp = httpx.get(f"{base}/h3studio/api/bootstrap", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    # 优先：cmd_args.cmd_opts.port
    try:
        from modules import shared
        port = getattr(shared.cmd_opts, "port", None)
        if port:
            base = f"http://127.0.0.1:{port}"
            if _check(base):
                return base
    except Exception:
        pass

    # 回退：扫描 7860-7870
    for port in range(7860, 7871):
        base = f"http://127.0.0.1:{port}"
        if _check(base):
            return base

    return None


def h3_video_generate_tool(prompt, duration=5, aspect_ratio="16:9",
                           first_frame=None, last_frame=None, reference_image=None):
    """调用 forge-h3-studio 插件生成视频（MiniMax H3）。

    支持两种后端模式：
    - api（云端）：直接使用 CloudClient 调用 MiniMax 云端 API
    - managed/external（本地 ComfyUI）：通过 forge-h3-studio HTTP API 提交

    参数:
        prompt: 视频描述提示词（中英文均可）
        duration: 视频时长（秒），4-15，默认5
        aspect_ratio: 宽高比，16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9，默认16:9
        first_frame: 首帧图片（PIL Image 或文件路径，可选）
        last_frame: 尾帧图片（PIL Image 或文件路径，可选）
        reference_image: 参考图片（PIL Image 或文件路径，可选，多模态参考模式）
    返回: dict，包含 status 和 video_path
    """
    import httpx
    import hashlib

    max_wait = 600  # 10 分钟
    poll_interval = 8

    # 1. bootstrap 检查
    base = _get_webui_base_url()
    if not base:
        return {"status": "error", "error": "forge-h3-studio 插件未检测到，请确认已安装并启用"}

    try:
        resp = httpx.get(f"{base}/h3studio/api/bootstrap", timeout=10)
        resp.raise_for_status()
        bootstrap = resp.json()
    except Exception as e:
        return {"status": "error", "error": f"forge-h3-studio bootstrap 检查失败: {e}"}

    cfg = bootstrap.get("config", {})
    backend = bootstrap.get("backend", {})
    mode = str(cfg.get("backend_mode", "managed")).lower()

    # 输出目录（webui 标准 outputs/h3_video）
    webui_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.path.join(webui_dir, "outputs", "h3_video")
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, f"h3_{int(time.time())}.mp4")

    # 参数校验
    try:
        duration = max(4, min(15, int(duration)))
    except (TypeError, ValueError):
        duration = 5
    valid_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"}
    if aspect_ratio not in valid_ratios:
        aspect_ratio = "16:9"

    # 保存参考图的辅助函数
    def _save_asset(img):
        if img is None:
            return None
        if isinstance(img, str) and os.path.isfile(img):
            return img
        # PIL Image 或 bytes → 保存到临时文件
        name = f"h3_ref_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}.png"
        tmp_dir = os.path.join(tempfile.gettempdir(), "h3_agent_refs")
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, name)
        if isinstance(img, Image.Image):
            img.save(path, "PNG")
        elif isinstance(img, (bytes, bytearray)):
            with open(path, "wb") as f:
                f.write(img)
        else:
            return None
        return path

    first_frame_path = _save_asset(first_frame)
    last_frame_path = _save_asset(last_frame)
    ref_path = _save_asset(reference_image)

    # ============================================================
    # 云端模式（api）：直接使用 CloudClient
    # ============================================================
    if mode == "api":
        try:
            from h3studio.cloud_client import CloudClient, CLOUD_UPLOAD_DIR, H3_FPS
        except ImportError as e:
            return {"status": "error", "error": f"无法导入 forge-h3-studio CloudClient: {e}"}

        # 注意：bootstrap 返回的公开 config 中 minimax_api_key 被清空（安全考虑，
        # 见 forge-h3-studio/h3studio/api.py 的 _public_config）。因此不能把 cfg
        # 传给 CloudClient，否则 api_key 为空、enabled() 返回 False。
        # 不传 config，让 CloudClient 内部调用 load_config() 从配置文件读取真实 Key。
        client = CloudClient()
        if not client.enabled():
            return {"status": "error", "error": "MiniMax API Key 未配置，请在 forge-h3-studio 设置或智能体「图像/视频生成 API 设置」中填写"}

        # 将参考图复制到 CLOUD_UPLOAD_DIR（CloudClient 从这里读取并转为 data URL）
        import shutil

        def _copy_to_cloud_uploads(src_path):
            if not src_path:
                return None
            dst = CLOUD_UPLOAD_DIR / os.path.basename(src_path)
            if not dst.is_file():
                try:
                    shutil.copy2(src_path, str(dst))
                except Exception:
                    # 如果同名冲突，用唯一文件名
                    dst = CLOUD_UPLOAD_DIR / f"ref_{int(time.time()*1000)}.png"
                    shutil.copy2(src_path, str(dst))
            return dst.name

        cloud_first = _copy_to_cloud_uploads(first_frame_path)
        cloud_last = _copy_to_cloud_uploads(last_frame_path)
        cloud_ref = _copy_to_cloud_uploads(ref_path)

        frames = duration * H3_FPS
        # 根据宽高比选尺寸
        if aspect_ratio in ("9:16", "3:4"):
            w, h = 768, 1344
        else:
            w, h = 1344, 768

        request = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "frames": frames,
            "duration": duration,
            "first_frame": cloud_first or "",
            "last_frame": cloud_last or "",
            "references": [{"kind": "image", "file": cloud_ref}] if cloud_ref else [],
        }

        print(f"[Agent] H3 云端提交: prompt='{prompt[:50]}...' duration={duration}s ratio={aspect_ratio}")

        try:
            task_id = client.submit(request)
        except Exception as e:
            return {"status": "error", "error": f"MiniMax 云端提交失败: {e}"}

        # 轮询
        start = time.time()
        last_status = ""
        while time.time() - start < max_wait:
            try:
                result = client.query(task_id)
            except Exception as e:
                return {"status": "error", "error": f"MiniMax 云端查询失败: {e}"}

            status = str(result.get("status", "")).lower()
            if status != last_status:
                print(f"[Agent] H3 任务状态: {status}")
                last_status = status

            if status in ("succeeded", "success"):
                video_url = result.get("video_url", "")
                if not video_url:
                    return {"status": "error", "error": "任务成功但未返回视频 URL"}
                # 关键修复：处理相对路径 URL
                if video_url.startswith("/"):
                    video_url = f"{base}{video_url}"
                try:
                    content, _ = client.download(video_url)
                    with open(video_path, "wb") as f:
                        f.write(content)
                except Exception as e:
                    return {"status": "error", "error": f"下载视频失败: {e}"}
                print(f"[Agent] H3 视频已保存: {video_path}")
                return {
                    "status": "success",
                    "video_path": video_path,
                    "duration": duration,
                    "prompt": prompt,
                    "backend": "minimax-cloud",
                }
            elif status in ("failed", "error"):
                return {"status": "error", "error": result.get("error") or "MiniMax 云端生成失败"}

            time.sleep(poll_interval)

        return {"status": "error", "error": f"视频生成超时（超过 {max_wait} 秒）"}

    # ============================================================
    # 本地模式（local / managed / external）：通过 forge-h3-studio HTTP API
    # ============================================================
    else:
        # local 模式（Forge 进程内 DiffSynth）不需要 ComfyUI，即使 backend.ready
        # 为 False（模型未下载）也可以提交——后端会在首次运行时自动下载缺失组件。
        # 只有 managed/external（ComfyUI）模式才需要强制检查后端就绪。
        if mode != "local" and not backend.get("ready"):
            return {"status": "error", "error": f"H3 后端未就绪: state={backend.get('state')}。请先启动 ComfyUI 后端。"}

        # 从 catalog 获取实际可用的模型文件名，避免硬编码文件名与云端 GPU 上
        # 实际下载的文件名不匹配（例如 pruned 版本、不同量化版本等）。
        try:
            catalog_resp = httpx.get(f"{base}/h3studio/api/catalog", timeout=15)
            catalog_resp.raise_for_status()
            catalog = catalog_resp.json()
        except Exception as e:
            catalog = {}
            print(f"[Agent] 获取 H3 catalog 失败，使用默认文件名: {e}")

        dit_models = catalog.get("models") or []
        te_models = catalog.get("text_encoders") or []
        vae_models = catalog.get("vaes") or []

        # 如果 catalog 显示模型未就绪，给出明确提示（不自动跳过，让用户知道缺什么）
        if mode == "local" and not catalog.get("h3_ready", True):
            missing = catalog.get("missing_nodes") or []
            missing_text = "；".join(missing) if missing else "本地模型目录中未找到 H3 权重"
            return {
                "status": "error",
                "error": f"H3 本地模型未就绪：{missing_text}。请在 forge-h3-studio 的模型下载器中下载 H3 模型（DiT + 文本编码器），VAE 可由后端自动下载。",
            }

        # 自动选择可用的模型文件名（取列表第一个；catalog 已按优先级排序）
        model_name = dit_models[0] if dit_models else "minimax_h3_fl2va_fp8.safetensors"
        te_name = te_models[0] if te_models else "qwen3vl_32b_minimax_h3_fp8.safetensors"
        # VAE：优先从 catalog 中选，找不到就用默认名（后端会自动下载）
        video_vae_name = next((v for v in vae_models if "video" in v.lower()), None) or "minimax_h3_video_vae_fp16.safetensors"
        audio_vae_name = next((v for v in vae_models if "audio" in v.lower()), None) or "minimax_h3_audio_vae_fp32.safetensors"

        print(f"[Agent] H3 模型选择: model={model_name}, te={te_name}, video_vae={video_vae_name}, audio_vae={audio_vae_name}")

        # 上传参考图
        def _upload_asset(img_path):
            if not img_path:
                return None
            try:
                with open(img_path, "rb") as f:
                    files = {"file": (os.path.basename(img_path), f, "image/png")}
                    up_resp = httpx.post(f"{base}/h3studio/api/assets/upload", files=files, timeout=60)
                    up_resp.raise_for_status()
                    return up_resp.json()
            except Exception as e:
                print(f"[Agent] 上传参考图失败: {e}")
                return None

        first_asset = _upload_asset(first_frame_path)
        last_asset = _upload_asset(last_frame_path)
        ref_asset = _upload_asset(ref_path)

        # 根据宽高比选尺寸
        if aspect_ratio in ("9:16", "3:4"):
            w, h = 768, 1344
        else:
            w, h = 1344, 768

        request = {
            "mode": "t2v",
            "model": model_name,
            "text_encoder": te_name,
            "video_vae": video_vae_name,
            "audio_vae": audio_vae_name,
            "prompt": prompt,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "frames": duration * 24,
            "steps": 30,
            "sampler": "euler",
            "scheduler": "simple",
            "shift_video": 12,
            "shift_audio": 3,
            "denoise": 1,
        }
        if first_asset:
            request["first_frame"] = first_asset.get("file", "")
        if last_asset:
            request["last_frame"] = last_asset.get("file", "")
        if ref_asset:
            request["references"] = [{"kind": "image", "file": ref_asset.get("file", "")}]

        print(f"[Agent] H3 本地提交: prompt='{prompt[:50]}...' duration={duration}s")

        try:
            submit_resp = httpx.post(f"{base}/h3studio/api/jobs", json=request, timeout=30)
            submit_resp.raise_for_status()
            job = submit_resp.json()
            job_id = job.get("id")
        except Exception as e:
            return {"status": "error", "error": f"提交 H3 任务失败: {e}"}

        if not job_id:
            return {"status": "error", "error": "提交成功但未返回 job_id"}

        # 轮询任务状态
        start = time.time()
        last_state = ""
        while time.time() - start < max_wait:
            try:
                poll_resp = httpx.get(f"{base}/h3studio/api/jobs/{job_id}", timeout=15)
                poll_resp.raise_for_status()
                job = poll_resp.json()
            except Exception as e:
                return {"status": "error", "error": f"轮询任务失败: {e}"}

            state = str(job.get("state", "")).lower()
            if state != last_state:
                print(f"[Agent] H3 任务状态: {state}")
                last_state = state

            if state == "completed":
                outputs = job.get("outputs", [])
                if not outputs:
                    return {"status": "error", "error": "任务完成但无输出文件"}
                # 找到视频输出
                video_url = ""
                for out in outputs:
                    fname = str(out.get("filename", "")).lower()
                    if fname.endswith((".mp4", ".webm", ".mov")):
                        video_url = out.get("url", "")
                        break
                if not video_url and outputs:
                    video_url = outputs[0].get("url", "")
                if not video_url:
                    return {"status": "error", "error": "未找到视频输出 URL"}

                # 关键修复：处理相对路径 URL
                if video_url.startswith("/"):
                    video_url = f"{base}{video_url}"

                try:
                    vid_resp = httpx.get(video_url, timeout=120, follow_redirects=True)
                    vid_resp.raise_for_status()
                    with open(video_path, "wb") as f:
                        f.write(vid_resp.content)
                except Exception as e:
                    return {"status": "error", "error": f"下载视频失败: {e}"}

                print(f"[Agent] H3 视频已保存: {video_path}")
                return {
                    "status": "success",
                    "video_path": video_path,
                    "duration": duration,
                    "prompt": prompt,
                    "backend": "forge-h3-studio",
                }
            elif state in ("failed", "cancelled"):
                return {"status": "error", "error": job.get("error") or f"任务{state}"}

            time.sleep(poll_interval)

        return {"status": "error", "error": f"视频生成超时（超过 {max_wait} 秒）"}
