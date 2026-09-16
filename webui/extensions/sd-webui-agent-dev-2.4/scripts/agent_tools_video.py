# =============================================================================
# Agent Tools — 视频生成工具（MiniMax H3 / Dreamina SeaDance）
# =============================================================================

import base64
import json
import os
import tempfile
import time
from io import BytesIO

from PIL import Image

from scripts.agent_config import load_config


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

    # 本地/托管工作台使用 forge-h3-studio 的 ComfyUI 工作流；只有工作台
    # 明确设为 api 时，才使用 Agent 的 YoboxAI 视频 API 分支。
    h3_backend_mode = "managed"
    try:
        h3_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "forge-h3-studio", "data", "config.json",
        )
        with open(h3_config_path, "r", encoding="utf-8") as h3_config_file:
            h3_backend_mode = str(json.load(h3_config_file).get("backend_mode") or "managed").strip().lower()
    except Exception:
        pass

    # YoboxAI H3 使用独立的视频 API，不依赖 forge-h3-studio 的 MiniMax Key。
    # 该分支必须在本地插件 bootstrap 检查之前执行，避免 API 模式被误判为未配置。
    try:
        agent_cfg = load_config()
        agent_provider = str(agent_cfg.get("api_provider") or "").strip().lower()
        agent_base_url = str(agent_cfg.get("base_url") or "").rstrip("/")
        agent_api_key = str(agent_cfg.get("api_key") or "").strip()
        if h3_backend_mode == "api" and (agent_provider == "yoboxai" or "api.yoboxai.com" in agent_base_url.lower()):
            if not agent_api_key:
                return {"status": "error", "error": "YoboxAI API Key 未配置，请在 Agent API 设置中填写"}

            yobox_base = agent_base_url or "https://api.yoboxai.com/v1"
            if not yobox_base.endswith("/v1"):
                yobox_base = yobox_base.rstrip("/") + "/v1"
            video_url = f"{yobox_base}/videos"
            timestamp = int(time.time() * 1000)

            def _data_url(asset):
                if asset is None:
                    return None
                if isinstance(asset, str) and os.path.isfile(asset):
                    with open(asset, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("ascii")
                    return f"data:image/png;base64,{encoded}"
                if isinstance(asset, Image.Image):
                    buf = BytesIO()
                    asset.convert("RGB").save(buf, format="PNG")
                    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
                return asset if isinstance(asset, str) and asset.startswith(("http://", "https://", "data:")) else None

            start_url = _data_url(first_frame)
            end_url = _data_url(last_frame)
            ref_url = _data_url(reference_image)
            input_data = {
                "prompt": str(prompt),
                "aspect_ratio": aspect_ratio,
                "resolution": "2K",
                "duration": duration,
                "audio": True,
                "n": 1,
            }
            if start_url:
                input_data["start_frames"] = [{"url": start_url}]
            if end_url:
                if not start_url:
                    return {"status": "error", "error": "首尾帧模式必须同时提供首帧和尾帧"}
                input_data["end_frames"] = [{"url": end_url}]
            elif ref_url:
                input_data["image_references"] = [{"url": ref_url, "strength": "MID"}]

            request = {"model": "MiniMax-H3", "input": input_data}
            headers = {
                "Authorization": f"Bearer {agent_api_key}",
                "Content-Type": "application/json",
                "Idempotency-Key": f"forge-agent-{timestamp}",
            }
            try:
                response = httpx.post(video_url, json=request, headers=headers, timeout=60)
                response.raise_for_status()
                submitted = response.json()
            except Exception as e:
                detail = response.text[:1200] if "response" in locals() else ""
                return {"status": "error", "error": f"YoboxAI H3 提交失败: {e}", "detail": detail}

            task_id = ((submitted.get("data") or {}).get("task_id") or submitted.get("task_id"))
            if not task_id:
                return {"status": "error", "error": "YoboxAI H3 未返回 task_id", "raw": submitted}

            deadline = time.time() + 900
            while time.time() < deadline:
                try:
                    poll = httpx.get(f"{video_url}/{task_id}", headers={"Authorization": f"Bearer {agent_api_key}"}, timeout=30)
                    poll.raise_for_status()
                    result = poll.json()
                except Exception as e:
                    return {"status": "error", "error": f"YoboxAI H3 查询失败: {e}"}
                data = result.get("data") or result
                status = str(data.get("status") or "").upper()
                if status in ("SUCCESS", "SUCCEEDED", "COMPLETED"):
                    output_url = data.get("video_url") or data.get("url") or data.get("download_url")
                    if not output_url:
                        return {"status": "error", "error": "YoboxAI H3 已完成但未返回视频 URL", "raw": result}
                    yobox_out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "h3_video")
                    os.makedirs(yobox_out_dir, exist_ok=True)
                    video_path = os.path.join(yobox_out_dir, f"h3_yobox_{timestamp}.mp4")
                    content = httpx.get(output_url, timeout=180).content
                    with open(video_path, "wb") as f:
                        f.write(content)
                    return {"status": "success", "video_path": video_path, "duration": duration,
                            "prompt": prompt, "backend": "yoboxai", "model": "MiniMax-H3", "task_id": task_id}
                if status in ("FAILED", "ERROR", "CANCELED", "CANCELLED"):
                    return {"status": "error", "error": data.get("error") or "YoboxAI H3 视频生成失败", "raw": result}
                time.sleep(8)
            return {"status": "error", "error": "YoboxAI H3 视频生成超时（超过 15 分钟）", "task_id": task_id}
    except Exception as e:
        return {"status": "error", "error": f"YoboxAI H3 参数处理失败: {e}"}

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

        client = CloudClient(cfg)
        if not client.enabled():
            return {"status": "error", "error": "MiniMax API Key 未配置，请在 forge-h3-studio 设置中填写"}

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
    # 本地模式（managed/external）：通过 forge-h3-studio HTTP API
    # ============================================================
    else:
        # 检查后端是否就绪
        if not backend.get("ready"):
            return {"status": "error", "error": f"H3 后端未就绪: state={backend.get('state')}。请先启动 ComfyUI 后端。"}

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
            "model": "minimax_h3_fl2va_fp8.safetensors",
            "text_encoder": "qwen3vl_32b_minimax_h3_fp8.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
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


def dreamina_video_generate_tool(prompt, duration=5, aspect_ratio="16:9", model=None,
                                 first_frame=None, last_frame=None, reference_image=None):
    """使用 YoboxAI Dreamina SeaDance 生成视频，独立于 MiniMax H3 本地/云端设置。"""
    import httpx
    import hashlib

    try:
        cfg = load_config()
        provider = str(cfg.get("video_api_provider") or "").strip().lower()
        base_url = str(cfg.get("video_base_url") or "https://api.yoboxai.com/v1").rstrip("/")
        api_key = str(cfg.get("video_api_key") or "").strip()
        prompt_text = str(prompt)
        model = str(model or cfg.get("video_model") or "dreamina-seedance-2-0-hc").strip()
        if "dreamina-seedance-2-5-hc" in prompt_text.lower():
            model = "dreamina-seedance-2-5-hc"
        elif "dreamina-seedance-2-0-hc" in prompt_text.lower():
            model = "dreamina-seedance-2-0-hc"

        if not api_key:
            return {"status": "error", "error": "视频生成 API Key 未配置，请在 Agent 视频生成设置中填写"}
        if provider and provider != "yoboxai" and "yoboxai.com" not in base_url.lower():
            return {"status": "error", "error": "Dreamina 当前只支持 YoboxAI 视频供应商，请在 Agent 视频生成设置中选择 YoboxAI"}
        if "api.yoboxai.com" in base_url.lower():
            base_url = "https://api.yoboxai.com/api/v3/contents/generations/tasks"
        elif not base_url.endswith("/tasks"):
            base_url = base_url.rstrip("/") + "/api/v3/contents/generations/tasks"

        try:
            duration = max(4, min(15, int(duration)))
        except (TypeError, ValueError):
            duration = 5
        valid_ratios = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "16:9"

        def _asset_url(asset):
            if asset is None:
                return None
            if isinstance(asset, str) and asset.startswith(("http://", "https://", "asset://")):
                return asset
            if isinstance(asset, str) and os.path.isfile(asset):
                with open(asset, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            if isinstance(asset, Image.Image):
                buf = BytesIO()
                asset.convert("RGB").save(buf, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
            return None

        start_url = _asset_url(first_frame)
        end_url = _asset_url(last_frame)
        ref_url = _asset_url(reference_image)
        content = [{"type": "text", "text": prompt_text}]
        if start_url:
            content.append({"type": "image_url", "image_url": {"url": start_url}, "role": "first_frame"})
        if end_url:
            if not start_url:
                return {"status": "error", "error": "首尾帧模式必须同时提供首帧和尾帧"}
            content.append({"type": "image_url", "image_url": {"url": end_url}, "role": "last_frame"})
        elif ref_url:
            content.append({"type": "image_url", "image_url": {"url": ref_url}, "role": "reference_image"})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": f"dreamina-{hashlib.md5(str(time.time()).encode()).hexdigest()}",
        }
        payload = {
            "model": model,
            "content": content,
            "duration": duration,
            "resolution": "720p",
            "ratio": aspect_ratio,
            "generate_audio": False,
            "watermark": False,
        }

        response = httpx.post(base_url, json=payload, headers=headers, timeout=60)
        if response.status_code >= 400:
            return {"status": "error", "error": f"Dreamina 视频提交失败（HTTP {response.status_code}）：{response.text[:800]}"}
        try:
            submitted = response.json()
        except ValueError:
            return {"status": "error", "error": f"Dreamina 视频提交返回的不是 JSON（HTTP {response.status_code}）：{response.text[:800]}"}

        task = submitted.get("task") or submitted.get("data") or {}
        if not isinstance(task, dict):
            task = {}
        task_id = str(
            task.get("id")
            or task.get("task_id")
            or submitted.get("id")
            or submitted.get("task_id")
            or submitted.get("taskId")
            or ""
        ).strip()
        if not task_id:
            return {"status": "error", "error": "Dreamina 未返回 task_id", "raw": submitted}

        deadline = time.time() + 900
        while time.time() < deadline:
            poll = httpx.get(f"{base_url}/{task_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            if poll.status_code >= 400:
                return {"status": "error", "error": f"Dreamina 查询失败（HTTP {poll.status_code}）：{poll.text[:800]}"}
            try:
                result = poll.json()
            except ValueError:
                return {"status": "error", "error": f"Dreamina 查询返回的不是 JSON（HTTP {poll.status_code}）：{poll.text[:800]}"}
            task = result.get("task") or result.get("data") or result
            if not isinstance(task, dict):
                task = result if isinstance(result, dict) else {}
            status = str(task.get("status") or "").lower()
            if status in ("completed", "complete", "succeeded", "success", "done"):
                def _find_video_url(value):
                    if isinstance(value, str):
                        return value if value.startswith(("http://", "https://")) and ".mp4" in value.lower() else None
                    if isinstance(value, list):
                        for item in value:
                            found = _find_video_url(item)
                            if found:
                                return found
                    if isinstance(value, dict):
                        for key in ("video_url", "videoUrl", "url", "download_url", "downloadUrl", "uri", "output"):
                            found = _find_video_url(value.get(key))
                            if found:
                                return found
                        for item in value.values():
                            found = _find_video_url(item)
                            if found:
                                return found
                    return None

                output_url = _find_video_url(result)
                if not output_url:
                    return {"status": "error", "error": "Dreamina 已完成但未返回视频 URL", "raw": result}
                out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "dreamina_video")
                os.makedirs(out_dir, exist_ok=True)
                video_path = os.path.join(out_dir, f"dreamina_{int(time.time())}.mp4")
                vid = httpx.get(output_url, timeout=180, follow_redirects=True)
                vid.raise_for_status()
                with open(video_path, "wb") as f:
                    f.write(vid.content)
                return {"status": "success", "video_path": video_path, "duration": duration,
                        "prompt": prompt, "backend": "dreamina-yoboxai", "model": model, "task_id": task_id}
            if status in ("failed", "error", "cancelled", "canceled"):
                return {"status": "error", "error": task.get("error") or "Dreamina 视频生成失败", "raw": result}
            time.sleep(8)
        return {"status": "error", "error": "Dreamina 视频生成超时（超过 15 分钟）", "task_id": task_id}
    except Exception as e:
        return {"status": "error", "error": f"Dreamina 视频生成失败: {e}"}
