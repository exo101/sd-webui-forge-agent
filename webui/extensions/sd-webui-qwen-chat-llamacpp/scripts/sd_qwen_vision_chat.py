import gradio as gr
from modules import scripts, script_callbacks
from pathlib import Path
import sys, os, json, base64, io, glob, traceback, logging
from modules.llama_port import get_llama_url
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)
_LLAMA_URL = get_llama_url()

scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))
extension_dir = scripts_dir.parent
if str(extension_dir) not in sys.path:
    sys.path.append(str(extension_dir))

# Import llama.cpp API
LLAMACPP_AVAILABLE = False
get_llamacpp_models = None
get_response_lvm_llamacpp_api = None
try:
    ollama_dir = extension_dir / "ollama"
    ollama_dir_str = str(ollama_dir)
    for mod_name in list(sys.modules.keys()):
        if 'llamacpp_api' in mod_name:
            del sys.modules[mod_name]
    if ollama_dir_str in sys.path:
        sys.path.remove(ollama_dir_str)
    sys.path.insert(0, ollama_dir_str)
    from llamacpp_api import get_response_lvm_llamacpp_api, get_llamacpp_models
    LLAMACPP_AVAILABLE = True
    logger.info(f"Vision Chat: llama.cpp API imported from {ollama_dir}")
except ImportError as e:
    logger.warning(f"Vision Chat: Could not import llama.cpp API: {e}")

default_vision_model = "Qwen3.5-2B-Q6_K.gguf"


class VisionChatScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self._injected = False
    def title(self):
        return "图像识别"
    def show(self, is_img2img):
        return False
    def ui(self, is_img2img):
        return []


# ============================================================
#  API Endpoints - Module-level handlers
# ============================================================

def on_app_started(demo, app):
    """Register all API endpoints"""
    logger.info("Vision Chat: registering API routes")

    # ---- Test/Ping Endpoint ----
    @app.get("/api/vision_ping")
    async def vision_ping():
        return JSONResponse({
            "status": "ok",
            "llamacpp_available": LLAMACPP_AVAILABLE,
            "llama_url": _LLAMA_URL,
            "default_model": default_vision_model
        })

    # ---- Vision Chat (POST) ----
    @app.post("/api/vision_chat")
    async def vision_chat(request: Request):
        try:
            data = await request.json()
            message = data.get("message", "")
            image_base64 = data.get("image", None)
            if not message and not image_base64:
                return JSONResponse({"response": "请输入消息或上传图片"})

            temp_image_path = None
            if image_base64:
                try:
                    temp_dir = os.path.join(extension_dir, "tmp", "vision_uploads")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, f"temp_{os.urandom(4).hex()}.png")
                    if "," in image_base64:
                        image_base64 = image_base64.split(",")[1]
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    image.save(temp_path)
                    temp_image_path = temp_path
                except Exception as e:
                    logger.error(f"Failed to save image: {e}")

            ai_response = ""
            if LLAMACPP_AVAILABLE and get_response_lvm_llamacpp_api:
                try:
                    ai_response = get_response_lvm_llamacpp_api(
                        input_model_name=default_vision_model,
                        input_content=message or "请描述这张图片",
                        input_image_path=temp_image_path,
                        llamacpp_host=_LLAMA_URL,
                        timeout=300
                    )
                    if not ai_response:
                        ai_response = "模型返回空结果，请检查模型是否已加载"
                except Exception as e:
                    ai_response = f"请求出错: {str(e)}"
                    logger.error(f"Vision chat API error: {e}")
            else:
                ai_response = "llama.cpp API 模块不可用，请检查安装"

            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass
            return JSONResponse({"response": ai_response})
        except Exception as e:
            logger.error(f"Vision chat error: {e}")
            traceback.print_exc()
            return JSONResponse({"response": f"服务器错误: {str(e)}"}, status_code=500)

    # ---- Tag Management (POST) ----
    @app.post("/api/vision_tag_manage")
    async def vision_tag_manage(request: Request):
        try:
            body_data = await request.json()
            action = body_data.get("action", "")
            folder_path = body_data.get("folder", "")
            keyword = body_data.get("keyword", "")
            position = body_data.get("position", "开头")
            file_name = body_data.get("file", "")

            if action == "list_files":
                if not folder_path or not os.path.isdir(folder_path):
                    return JSONResponse({"files": [], "error": "无效的文件夹路径"})
                txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                return JSONResponse({"files": txt_files})

            if action in ("add", "remove"):
                file_path = os.path.join(folder_path, file_name) if folder_path and file_name else ""
                if not file_path or not os.path.exists(file_path):
                    return JSONResponse({"result": "❌ 文件不存在"})
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if action == "add":
                        if position == "开头":
                            content = keyword + '\n' + content
                        elif position == "结尾":
                            content = content + '\n' + keyword
                        else:
                            import random
                            lines = content.split('\n')
                            idx = random.randint(0, len(lines))
                            lines.insert(idx, keyword)
                            content = '\n'.join(lines)
                    else:
                        content = content.replace(keyword, "")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return JSONResponse({"result": f"✅ 关键词已{'添加' if action == 'add' else '删除'}到 {file_name}"})
                except Exception as e:
                    return JSONResponse({"result": f"❌ 错误: {str(e)}"})

            if action in ("add_all", "remove_all"):
                if not folder_path or not os.path.isdir(folder_path):
                    return JSONResponse({"result": "❌ 无效的文件夹路径"})
                txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                success = 0
                failed = 0
                results = []
                for fname in txt_files:
                    file_path = os.path.join(folder_path, fname)
                    if not os.path.exists(file_path):
                        failed += 1
                        continue
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if action == "add_all":
                            if position == "开头":
                                content = keyword + '\n' + content
                            elif position == "结尾":
                                content = content + '\n' + keyword
                            else:
                                import random
                                lines = content.split('\n')
                                idx = random.randint(0, len(lines))
                                lines.insert(idx, keyword)
                                content = '\n'.join(lines)
                        else:
                            content = content.replace(keyword, "")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        success += 1
                        results.append(f"✅ {fname}")
                    except Exception as e:
                        failed += 1
                        results.append(f"❌ {fname}: {str(e)}")
                return JSONResponse({"result": f"操作完成: 成功 {success} 个, 失败 {failed} 个\n" + "\n".join(results)})

            return JSONResponse({"result": "未知操作"})
        except Exception as e:
            return JSONResponse({"result": f"❌ 服务器错误: {str(e)}"}, status_code=500)

    # ---- Batch Images (POST) ----
    @app.post("/api/vision_batch_images")
    async def vision_batch_images(request: Request):
        try:
            body_data = await request.json()
            dir_path = body_data.get("dir", "")
            if not dir_path or not os.path.isdir(dir_path):
                return JSONResponse({"images": [], "error": "无效的目录路径"})
            supported = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
            images = []
            for ext in supported:
                for f in glob.glob(os.path.join(dir_path, f"*{ext}")):
                    try:
                        with Image.open(f) as img:
                            img.verify()
                        images.append({"name": os.path.basename(f), "path": os.path.normpath(f)})
                    except:
                        pass
            return JSONResponse({"images": images})
        except Exception as e:
            return JSONResponse({"images": [], "error": str(e)}, status_code=500)

    # ---- Batch Recognize (POST) ----
    @app.post("/api/vision_batch_recognize")
    async def vision_batch_recognize(request: Request):
        try:
            body_data = await request.json()
            dir_path = body_data.get("dir", "")
            if not dir_path or not os.path.isdir(dir_path):
                return JSONResponse({"results": [], "error": "无效的目录路径"})
            supported = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
            image_files = []
            for ext in supported:
                image_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
            if not image_files:
                return JSONResponse({"results": [], "error": "没有找到图片文件"})

            import concurrent.futures
            results = []
            max_workers = min(4, len(image_files))

            def process_one(img_path):
                try:
                    if LLAMACPP_AVAILABLE and get_response_lvm_llamacpp_api:
                        resp = get_response_lvm_llamacpp_api(
                            input_model_name=default_vision_model,
                            input_content="简单描述这张图片的内容",
                            input_image_path=img_path,
                            llamacpp_host=_LLAMA_URL,
                            timeout=300
                        )
                        return {"name": os.path.basename(img_path), "result": resp or "(空结果)"}
                    return {"name": os.path.basename(img_path), "result": "API 不可用"}
                except Exception as e:
                    return {"name": os.path.basename(img_path), "result": f"错误: {str(e)}"}

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(process_one, p) for p in image_files]
                for f in concurrent.futures.as_completed(futures):
                    results.append(f.result())

            return JSONResponse({"results": results})
        except Exception as e:
            return JSONResponse({"results": [], "error": str(e)}, status_code=500)


# ============================================================
#  Register callbacks
# ============================================================

script_callbacks.on_app_started(on_app_started)