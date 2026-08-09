from __future__ import annotations
import os
import sys
import time
import json
import ssl
import requests
import gradio as gr
from typing import List, Dict

os.environ['SSL_CERT_FILE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_VERIFICATION'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

ssl._create_default_https_context = ssl._create_unverified_context

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
        try:
            if source == "huggingface":
                session = requests.Session()
                session.verify = False
                session.headers.update(build_hf_headers())
                url = f"https://huggingface.co/api/models/{model_name}/tree/main?recursive=True&expand=False"
                response = session.get(url)
                hf_raise_for_status(response)
                data = response.json()
                files = []
                self._extract_files(data, files)
                return files
            else:
                if HubApi:
                    api = HubApi()
                    info = api.model_info(model_name)
                    files = []
                    if hasattr(info, 'siblings') and info.siblings:
                        for item in info.siblings:
                            if hasattr(item, 'rfilename'):
                                files.append(item.rfilename)
                    return files
                return []
        except Exception as e:
            print(f"[ModelDownloader] 获取文件列表失败: {e}")
            return []
    
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
                        return gr.update(choices=[], value=None), "请输入模型名称"
                    if len(model_name.split('/')) < 2:
                        return gr.update(choices=[], value=None), "模型 ID 格式错误，正确格式: 用户名/模型名"
                    files = model_downloader.list_model_files(model_name, src)
                    if not files:
                        return gr.update(choices=[], value=None), f"未能获取 {model_name} 的文件列表"
                    return gr.update(choices=files, value=None), f"获取到 {len(files)} 个文件"

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
