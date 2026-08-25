from __future__ import annotations
import os
import sys
import time
import json
import ssl
import requests
import gradio as gr
from typing import List, Dict

# ============================================================
# 修复: 不再全局污染 SSL (魔搭创空间容器需要完整证书链才能让
# ModelScope SDK 内部 requests 握手成功). 仅在具体请求的 session
# 上单独关闭验证, 不影响其它模块.
# ============================================================
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
    from huggingface_hub.hf_api import HfApi
    from huggingface_hub.utils._http import hf_raise_for_status
    from huggingface_hub.utils import build_hf_headers
except ImportError as e:
    print(f"[ModelDownloader] 核心依赖缺失，请运行: pip install huggingface_hub")
    raise e

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    from modelscope.hub.file_download import model_file_download
    from modelscope import HubApi
except ImportError as e:
    ms_snapshot_download = None
    model_file_download = None
    HubApi = None
    print(f"[ModelDownloader] ModelScope SDK 导入失败: {e}")


def _get_modelscope_token() -> str:
    """
    多渠道获取 ModelScope token:
      1. 环境变量 MODELSCOPE_API_TOKEN (魔搭创空间通常会自动注入)
      2. 环境变量 MODELSCOPE_TOKEN (兼容)
      3. ModelScope SDK 默认 ~/.modelscope/token 缓存
      4. 本插件 cache 目录手动保存的 token
    返回空字符串表示未配置 (匿名访问 modelscope.cn 上的公开模型仍可用)
    """
    tok = (
        os.environ.get('MODELSCOPE_API_TOKEN')
        or os.environ.get('MODELSCOPE_TOKEN')
        or ''
    )
    if tok:
        return tok
    # 兜底: 读 SDK 默认缓存
    try:
        from modelscope.utils.constant import MODELSCOPE_TOKEN_PATH
        if os.path.exists(MODELSCOPE_TOKEN_PATH):
            with open(MODELSCOPE_TOKEN_PATH, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass
    # 兜底: 本插件缓存
    local_tok = os.path.join(CACHE_DIR, "ms_token.txt")
    if os.path.exists(local_tok):
        try:
            with open(local_tok, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            pass
    return ''

from modules import script_callbacks
from modules.shared import opts, cmd_opts
from modules.paths import models_path

MODEL_DOWNLOADER_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CACHE_DIR = os.path.join(MODEL_DOWNLOADER_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class ModelDownloader:
    def __init__(self):
        self.downloading = False
        self.current_task = None
        self.download_history = []
        self.history_file = os.path.join(CACHE_DIR, "download_history.json")
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.download_history = json.load(f)
            except Exception:
                self.download_history = []

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_history[:100], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def list_model_files(self, model_name: str, source: str) -> List[str]:
        """
        返回模型仓库内的文件列表. 失败时抛异常给调用方,
        让前端能看到真实错误 (而不是静默返回空列表).
        """
        model_name = (model_name or '').strip()
        if not model_name:
            raise ValueError("模型名称不能为空")

        if source == "huggingface":
            # HuggingFace 分支: 在魔搭创空间容器内 huggingface.co 通常不可达,
            # 给 12 秒超时 + 友好报错, 让用户知道该改用 ModelScope 源.
            session = requests.Session()
            session.verify = False  # 仅本 session 不验证, 不影响其它请求
            session.headers.update(build_hf_headers())
            url = f"https://huggingface.co/api/models/{model_name}/tree/main?recursive=True&expand=False"
            try:
                response = session.get(url, timeout=12)
                hf_raise_for_status(response)
                data = response.json()
                files: List[str] = []
                self._extract_files(data, files)
                return files
            except requests.exceptions.ConnectTimeout:
                raise RuntimeError(
                    "无法连接 HuggingFace (huggingface.co). "
                    "魔搭创空间容器可能禁止外网访问, 请改用源 = ModelScope."
                )
            except requests.exceptions.SSLError as e:
                raise RuntimeError(f"SSL 握手失败: {e}. 请检查容器证书链.")
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else "?"
                if code == 401 or code == 403:
                    raise RuntimeError(
                        f"HuggingFace 返回 {code}: 未授权. 请先 `huggingface-cli login` "
                        f"或在魔搭创空间配置 HF_TOKEN 环境变量."
                    )
                if code == 404:
                    raise RuntimeError(f"模型仓库不存在: {model_name}")
                raise RuntimeError(f"HuggingFace HTTP {code}: {e}")
            except Exception as e:
                raise RuntimeError(f"HuggingFace 获取失败: {e}")

        # ============ ModelScope 分支 ============
        if HubApi is None:
            raise RuntimeError(
                "ModelScope SDK 未安装. 请在魔搭创空间终端运行: "
                "pip install modelscope"
            )

        # 带上 token (魔搭创空间的环境变量 / SDK 缓存 / 本插件缓存)
        token = _get_modelscope_token()
        try:
            # 关键: 显式传 token, 让私有模型/受限模型也能列文件
            api = HubApi()
            if token:
                try:
                    api.login_with_token(token)
                except Exception:
                    # 某些 SDK 版本没有 login_with_token, 用 set_token 兜底
                    try:
                        os.environ['MODELSCOPE_API_TOKEN'] = token
                    except Exception:
                        pass
            info = api.model_info(model_name)
            files = []
            if hasattr(info, 'siblings') and info.siblings:
                for item in info.siblings:
                    if hasattr(item, 'rfilename'):
                        files.append(item.rfilename)
            if not files:
                # 兜底: 有些 SDK 版本字段名不一样, 用 dict 探测
                if isinstance(info, dict):
                    for item in info.get('Data', {}).get('Files', []) or []:
                        if isinstance(item, dict) and 'Name' in item:
                            files.append(item['Name'])
            return files
        except Exception as e:
            err_str = str(e)
            # 给用户明确的失败原因 (而不是静默返回空)
            if '401' in err_str or '403' in err_str or 'Unauthorized' in err_str:
                raise RuntimeError(
                    f"ModelScope 未授权 ({err_str[:120]}). "
                    f"魔搭创空间请在环境变量里配置 MODELSCOPE_API_TOKEN."
                )
            if '404' in err_str or 'not exist' in err_str.lower():
                raise RuntimeError(f"ModelScope 模型仓库不存在: {model_name}")
            if 'timeout' in err_str.lower() or 'timed out' in err_str.lower():
                raise RuntimeError(
                    "ModelScope API 超时. 魔搭创空间可能需要配置内网代理."
                )
            raise RuntimeError(f"ModelScope 获取失败: {err_str[:200]}")
    
    def _extract_files(self, data, files):
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('type') == 'file':
                        files.append(item.get('path', ''))
                    elif item.get('type') == 'dir':
                        self._extract_files(item.get('children', []), files)

    def download_model(self, model_name: str, source: str, save_path: str, filename: str = "") -> bool:
        try:
            if self.downloading:
                return False
            self.downloading = True
            
            model_name = model_name.strip()
            if not model_name:
                raise ValueError("模型名称不能为空")
                
            if len(model_name.split('/')) < 2:
                raise ValueError(f"模型 ID 格式错误。正确格式: '用户名/模型名'\n例如: Comfy-Org/Krea-2 或 qwen/Qwen2.5-7B-Instruct")
                
            self.current_task = f"Downloading {model_name}"
            os.makedirs(save_path, exist_ok=True)

            if source == "huggingface":
                print(f"[ModelDownloader] 从 HuggingFace 下载: {model_name}")
                session = requests.Session()
                session.verify = False
                if filename:
                    hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        token=None,
                        session=session,
                    )
                else:
                    ignore_8bit = not getattr(cmd_opts, 'load_in_8bit', False)
                    ignore_patterns = ["*.bin", "*.h5"] if ignore_8bit else []
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        ignore_patterns=ignore_patterns,
                        token=None,
                        session=session,
                    )
            else:
                print(f"[ModelDownloader] 从 ModelScope 下载: {model_name}")
                if filename:
                    if model_file_download:
                        model_file_download(
                            model_id=model_name,
                            file_path=filename,
                            cache_dir=save_path,
                        )
                    else:
                        raise RuntimeError("ModelScope SDK 单文件下载功能不可用，请尝试完整模型下载")
                else:
                    if ms_snapshot_download:
                        ms_snapshot_download(
                            model_id=model_name,
                            cache_dir=save_path,
                        )
                    else:
                        raise RuntimeError("ModelScope SDK 完整模型下载功能不可用")

            self.download_history.append({
                "model_name": model_name, 
                "source": source, 
                "save_path": save_path,
                "filename": filename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), 
                "type": "single" if filename else "full",
            })
            self._save_history()
            return True
        except Exception as e:
            print(f"[ModelDownloader] 下载错误: {str(e)}")
            return False
        finally:
            self.downloading = False
            self.current_task = None

model_downloader = ModelDownloader()

def create_ui():
    with gr.Blocks(title="开源社区模型下载器", analytics_enabled=False) as model_downloader_tab:
        with gr.Tabs():
            with gr.TabItem("模型下载"):
                with gr.Row():
                    with gr.Column(scale=1):
                        source = gr.Radio(
                            choices=["huggingface", "modelscope"],
                            value="modelscope",
                            label="模型源",
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        model_input = gr.Textbox(
                            label="模型名称/仓库ID",
                            placeholder="例如: Comfy-Org/Krea-2 或 qwen/Qwen2.5-7B-Instruct",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        fetch_files_btn = gr.Button("获取文件列表")
                        download_full_btn = gr.Button("下载完整模型", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        file_dropdown = gr.Dropdown(
                            choices=[],
                            label="选择要下载的文件",
                            interactive=True,
                            multiselect=True,
                        )
                    with gr.Column(scale=1):
                        download_single_btn = gr.Button("下载选中文件")
                
                save_path = gr.Textbox(
                    label="保存路径",
                    value=os.path.join(models_path, "Stable-diffusion"),
                    interactive=True,
                )
                
                status = gr.Textbox(label="下载状态", interactive=False)

                def fetch_files(model_name, src):
                    if not model_name:
                        return gr.update(choices=[], value=None), "❌ 请输入模型名称"
                    if len(model_name.split('/')) < 2:
                        return gr.update(choices=[], value=None), "❌ 模型 ID 格式错误，正确格式: 用户名/模型名"
                    try:
                        files = model_downloader.list_model_files(model_name, src)
                    except Exception as e:
                        # 修复: 不再吞异常, 把真实错误返回前端
                        return gr.update(choices=[], value=None), f"❌ 获取失败: {e}"
                    if not files:
                        return gr.update(choices=[], value=None), (
                            f"⚠️ 仓库 {model_name} 返回空文件列表.\n"
                            f"可能原因:\n"
                            f"  - 模型 ID 拼写错误\n"
                            f"  - 该仓库是 LFS 仓库, 文件需用完整下载\n"
                            f"  - 源选择错误 (魔搭创空间内 HuggingFace 不可达, 请切到 ModelScope)\n"
                            f"  - 未配置 MODELSCOPE_API_TOKEN 环境变量"
                        )
                    # 显示前 5 个文件名作为预览
                    preview = "\n".join(files[:5])
                    if len(files) > 5:
                        preview += f"\n... 等共 {len(files)} 个文件"
                    return gr.update(choices=files, value=None), (
                        f"✅ 获取到 {len(files)} 个文件:\n{preview}"
                    )

                def download_full(model_name, src, path):
                    if not model_name: return "请输入模型名称"
                    try:
                        prog = gr.Progress()
                        prog(0, desc="开始下载...")
                        success = model_downloader.download_model(model_name, src, path)
                        prog(1, desc="下载完成" if success else "下载失败")
                        return f"模型 {model_name} 下载完成，保存到: {path}" if success else "下载失败，请查看控制台日志"
                    except Exception as e:
                        return f"下载异常: {str(e)}"

                def download_single(model_name, src, path, filenames):
                    if not model_name: return "请输入模型名称"
                    if not filenames or len(filenames) == 0: return "请先选择要下载的文件"
                    try:
                        prog = gr.Progress()
                        prog(0, desc=f"开始下载 {len(filenames)} 个文件...")
                        for i, filename in enumerate(filenames):
                            success = model_downloader.download_model(model_name, src, path, filename)
                            prog((i+1)/len(filenames), desc=f"正在下载 {filename}...")
                            if not success:
                                return f"文件 {filename} 下载失败，请查看控制台日志"
                        return f"已成功下载 {len(filenames)} 个文件到: {path}"
                    except Exception as e:
                        return f"下载异常: {str(e)}"

                fetch_files_btn.click(fetch_files, inputs=[model_input, source], outputs=[file_dropdown, status])
                download_full_btn.click(download_full, inputs=[model_input, source, save_path], outputs=status)
                download_single_btn.click(download_single, inputs=[model_input, source, save_path, file_dropdown], outputs=status)

            with gr.TabItem("下载历史"):
                def load_history_data():
                    return [[h["time"], h["model_name"], h["source"], h["type"], h["save_path"]] for h in model_downloader.download_history]

                history_table = gr.Dataframe(
                    headers=["时间", "模型名称", "来源", "类型", "路径"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="下载历史",
                    interactive=False,
                    value=load_history_data(),
                )
                
                def clear_history():
                    model_downloader.download_history.clear()
                    model_downloader._save_history()
                    return []
                
                clear_history_btn = gr.Button("清空历史")
                clear_history_btn.click(clear_history, outputs=history_table)
                
    return [(model_downloader_tab, "开源社区模型下载器", "model-downloader")]

def on_ui_tabs():
    return create_ui()

script_callbacks.on_ui_tabs(on_ui_tabs)
