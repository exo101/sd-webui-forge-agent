"""
TRELLIS.2 - Native and Compact Structured Latents for 3D Generation
Extension for SD WebUI Forge Classic Neo

This extension integrates TRELLIS.2 image-to-3D generation capabilities
into the Stable Diffusion WebUI interface.
"""

import os
import sys
import gradio as gr
from modules import scripts, shared
from modules.paths import models_path


# Add TRELLIS.2 project to Python path
# scripts/trellis2_script.py is in the scripts/ subdirectory
# Need to go up one level to get the extension root directory
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSION_DIR = os.path.dirname(SCRIPTS_DIR)  # Go up from scripts/ to extension root

# Get webui root directory (go up 3 levels: scripts -> extension -> webui)
WEBUI_ROOT = os.path.dirname(os.path.dirname(EXTENSION_DIR))

# Only use bundled TRELLIS.2 inside extension directory
TRELLIS2_ROOT = os.path.join(EXTENSION_DIR, 'TRELLIS.2')

if os.path.exists(TRELLIS2_ROOT):
    sys.path.insert(0, TRELLIS2_ROOT)
    
    # Add source-only modules (must load from TRELLIS.2 directory)
    # These are Python packages that should be loaded from source
    source_only_modules = ['trellis2', 'o-voxel']
    
    for subdir in source_only_modules:
        subdir_path = os.path.join(TRELLIS2_ROOT, subdir)
        if os.path.exists(subdir_path):
            sys.path.insert(0, subdir_path)
    
    # Check if nvdiffrec exists in extension directory (local installation)
    NVDIFFREC_PATH = os.path.join(EXTENSION_DIR, 'nvdiffrec')
    if os.path.exists(NVDIFFREC_PATH):
        sys.path.insert(0, NVDIFFREC_PATH)
    
    # Check compiled modules - these should be installed as wheels via pip
    # Do NOT try to load them from TRELLIS.2 directory
    # Key: import_name (used in __import__), Value: display_name (shown in error messages)
    compiled_modules = {
        'nvdiffrast': 'nvdiffrast',
        'nvdiffrec': 'nvdiffrec',
        'cumesh': 'CuMesh',
        'flex_gemm': 'FlexGEMM',
        'utils3d': 'utils3d',
    }
        
    missing_modules = []
    for import_name, display_name in compiled_modules.items():
        try:
            __import__(import_name)
            # Module already installed as wheel, skip silently
        except ImportError:
            # Special handling for nvdiffrec - check if available locally
            if import_name == 'nvdiffrec' and os.path.exists(NVDIFFREC_PATH):
                try:
                    __import__('render.renderutils')
                    # Successfully loaded from local directory
                    continue
                except ImportError:
                    pass
            missing_modules.append(display_name)
    
    # Only show error if there are actually missing modules
    if missing_modules:
        print(f"[TRELLIS.2] Warning: Missing compiled modules: {', '.join(missing_modules)}")
        print(f"[TRELLIS.2] To install, run:")
        print(f"[TRELLIS.2]   cd {EXTENSION_DIR}")
        print(f"[TRELLIS.2]   python install_missing_modules.py")
        print(f"[TRELLIS.2] Or install manually using pip")
else:
    print(f"[TRELLIS.2] Warning: TRELLIS.2 not found at {TRELLIS2_ROOT}")
    print(f"[TRELLIS.2] Please ensure TRELLIS.2 is placed in the extension directory")


# Detect flash-attn availability
flash_attn_available = False
try:
    import flash_attn
    flash_attn_available = True
except ImportError:
    pass


# Model directory configuration
TRELLIS2_MODEL_DIR = os.path.join(models_path, 'trellis2', 'TRELLIS.2-4B')
os.makedirs(TRELLIS2_MODEL_DIR, exist_ok=True)

# Set environment variables for TRELLIS.2
os.environ['TRELLIS_MODEL_PATH'] = os.environ.get('TRELLIS_MODEL_PATH', TRELLIS2_MODEL_DIR)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Global pipeline instance - Lazy loading (only loaded when first needed)
trellis_pipeline = None
envmap_cache = {}


def load_trellis_pipeline():
    """Lazy load TRELLIS.2 pipeline - Only loads when generate_3d_from_image is called"""
    global trellis_pipeline, envmap_cache
    
    if trellis_pipeline is not None:
        return trellis_pipeline
    
    print("[TRELLIS.2] Loading model (this may take 2-5 minutes on first run)...")
    
    try:
        # Add TRELLIS.2 project root to Python path
        if TRELLIS2_ROOT not in sys.path:
            sys.path.insert(0, TRELLIS2_ROOT)
        
        # Disable SSL verification for HuggingFace Hub (needed for some network environments)
        import ssl
        import urllib3
        
        # Disable SSL warnings and verification
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['CURL_CA_BUNDLE'] = ''
        
        # Monkey patch SSL context to disable verification
        _create_unverified_https_context = ssl._create_unverified_context
        if not hasattr(ssl, '_create_default_https_context'):
            ssl._create_default_https_context = _create_unverified_https_context
        
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        
        # Convert Windows path to forward slashes for compatibility
        model_path = TRELLIS2_MODEL_DIR.replace('\\', '/')
        
        # Verify the path is valid before loading
        if not os.path.exists(TRELLIS2_MODEL_DIR):
            raise FileNotFoundError(f"Model directory does not exist: {TRELLIS2_MODEL_DIR}")
        
        config_file = os.path.join(TRELLIS2_MODEL_DIR, 'pipeline.json')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Pipeline config not found: {config_file}")
        
        trellis_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        trellis_pipeline.cuda()
        print("[TRELLIS.2] Model loaded successfully")
        
        # Load environment maps if available
        hdri_dir = os.path.join(TRELLIS2_ROOT, 'assets', 'hdri')
        if os.path.exists(hdri_dir):
            from trellis2.renderers import EnvMap
            import torch
            import cv2
            
            for env_name in ['forest', 'sunset']:
                env_file = os.path.join(hdri_dir, f'{env_name}.exr')
                if os.path.exists(env_file):
                    try:
                        env_data = cv2.cvtColor(cv2.imread(env_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                        envmap_cache[env_name] = EnvMap(torch.tensor(env_data, dtype=torch.float32, device='cuda'))
                    except Exception:
                        pass
        
        return trellis_pipeline
        
    except Exception as e:
        print(f"[TRELLIS.2] Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_3d_from_image(
    input_image,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    octree_resolution,
    simplify_ratio,
    texture_resolution,
    env_map,
    use_flash_attn,
    progress=gr.Progress()
):
    """Generate 3D asset from input image"""
    
    if input_image is None:
        return None, "请上传输入图像", None, None
    
    try:
        progress(0, desc="加载 TRELLIS.2 模型...")
        pipeline = load_trellis_pipeline()
        
        # Process seed
        if randomize_seed:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
        
        progress(0.1, desc="处理图像...")
        
        # Convert PIL Image to proper format
        from PIL import Image
        if isinstance(input_image, dict) and 'image' in input_image:
            pil_image = input_image['image']
        else:
            pil_image = input_image
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        progress(0.2, desc="生成3D结构...")
        
        # Map Chinese env_map values to English
        env_map_mapping = {
            "无": "none",
            "森林": "forest",
            "日落": "sunset"
        }
        env_map_value = env_map_mapping.get(env_map, "none")
        
        # Convert octree_resolution to int (it's a string from Dropdown)
        try:
            octree_res_int = int(octree_resolution) if isinstance(octree_resolution, str) else octree_resolution
        except (ValueError, TypeError):
            octree_res_int = 1024  # Default value
        
        # Map octree_resolution to pipeline_type
        # TRELLIS.2 supports: '512', '1024', '1024_cascade', '1536_cascade'
        if octree_res_int <= 512:
            pipeline_type = '512'
        elif octree_res_int <= 1024:
            pipeline_type = '1024'
        else:
            pipeline_type = '1536_cascade'
        
        # Prepare sampler parameters
        sparse_structure_sampler_params = {
            'guidance_strength': guidance_scale,
            'steps': steps,
        }
        shape_slat_sampler_params = {
            'guidance_strength': guidance_scale,
            'steps': steps,
        }
        tex_slat_sampler_params = {
            'guidance_strength': guidance_scale,
            'steps': steps,
        }
        
        # Generate 3D asset
        outputs = pipeline.run(
            pil_image,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
            pipeline_type=pipeline_type,
        )
        
        progress(0.6, desc="提取网格...")
        
        # Get mesh from outputs (outputs is a list, get first element)
        mesh = outputs[0]
        
        progress(0.8, desc="保存输出...")
        
        # Save outputs
        output_dir = os.path.join(WEBUI_ROOT, 'output', 'trellis2')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(__import__('time').time())
        glb_path = os.path.join(output_dir, f'trellis2_{timestamp}.glb')
        
        # Export mesh using o-voxel postprocess
        import o_voxel
        try:
            # Keep ALL data on GPU for o-voxel processing (it will handle device internally)
            vertices_gpu = mesh.vertices
            faces_gpu = mesh.faces
            attrs_gpu = mesh.attrs
            coords_gpu = mesh.coords
            
            # Calculate reasonable decimation target based on face count
            num_faces = len(faces_gpu)
            # Use simplify_ratio to determine target: ratio 0.5 means keep ~50% of faces
            decimation_target = max(10000, min(int(num_faces * simplify_ratio), 500000))
            
            # Convert octree_resolution string to int for grid_size
            grid_size = int(octree_resolution)
            
            # Export to GLB with decimation and texture extraction
            # o_voxel expects data on GPU and will handle device internally
            glb = o_voxel.postprocess.to_glb(
                vertices=vertices_gpu,
                faces=faces_gpu,
                attr_volume=attrs_gpu,
                coords=coords_gpu,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=grid_size,  # Use octree_resolution from UI
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=int(texture_resolution),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=False,
            )
            glb.export(glb_path, extension_webp=True)
            
        except Exception as export_err:
            import traceback
            print(f"[TRELLIS.2] Warning: Export failed: {export_err}")
            print(traceback.format_exc())
            
            # Fallback: save basic mesh data
            try:
                import numpy as np
                fallback_path = glb_path.replace('.glb', '.npz')
                np.savez(fallback_path, 
                        vertices=mesh.vertices.cpu().numpy(),
                        faces=mesh.faces.cpu().numpy())
            except:
                pass
        
        progress(1.0, desc="完成！")
        
        status_msg = f"✓ 生成完成！\n种子: {seed}\n输出: {glb_path}"
        
        # Return results (match the order of outputs in UI)
        return status_msg, glb_path if os.path.exists(glb_path) else None, pil_image
        
    except Exception as e:
        import traceback
        error_msg = f"✗ 错误: {str(e)}\n\n{traceback.format_exc()}"
        print(f"[TRELLIS.2] {error_msg}")
        return error_msg, None, None


def send_glb_to_blender(glb_path):
    """将 GLB 模型发送到 Blender"""
    if not glb_path:
        return "❌ 没有可发送的 GLB 模型，请先生成3D模型"
    
    try:
        import base64, json, urllib.request, urllib.error
        
        with open(glb_path, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        port = 7869
        if hasattr(shared, 'cmd_opts') and hasattr(shared.cmd_opts, 'port'):
            port = shared.cmd_opts.port
        elif hasattr(shared, 'args') and hasattr(shared.args, 'port'):
            port = shared.args.port
        url = f"http://127.0.0.1:{port}/sdapi/v1/ps-plugin/bridge/send-webui-glb-to-blender"
        req = urllib.request.Request(url, data=json.dumps({"glb_base64": glb_b64}).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read().decode("utf-8"))
        
        if result.get("status") == "success":
            return "✅ GLB 模型已发送到 Blender！请在 Blender 中点击「从 WebUI 导入 GLB」"
        else:
            return f"❌ {result.get('message', '发送失败')}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
        return f"❌ HTTP {e.code}: {detail}"
    except urllib.error.URLError as e:
        return f"❌ 连接失败: {e.reason}"
    except Exception as e:
        return f"❌ 发送失败: {str(e)}"


class Trellis2Script(scripts.Script):
    """TRELLIS.2 Image-to-3D Generation Script"""
    
    def title(self):
        return "TRELLIS.2 图生成3D"
    
    def show(self, is_img2img):
        # Return None to hide from txt2img/img2img tabs
        # We'll create a separate tab instead
        return None
    
    def ui(self, is_img2img):
        # This method won't be called since show() returns None
        return []


# Register independent UI tab using script_callbacks
def on_ui_tabs():
    """Create TRELLIS.2 independent tab"""
    
    with gr.Blocks(analytics_enabled=False) as trellis2_interface:
        gr.Markdown("""
        ### TRELLIS.2: 图像生成3D 模型
        """)
        
        # 安装注意事项和模型信息折叠模块
        with gr.Accordion("📦 安装说明与模型信息", open=False):
            gr.Markdown(r"""
            #### ⚠️ 安装前必读
            
            **1. Visual Studio 2022 安装**
            - 下载并安装 [Visual Studio 2022 Community](https://visualstudio.microsoft.com/zh-hans/downloads/)
            - 安装时务必勾选 **"使用 C++ 的桌面开发"** 工作负载
            - 此步骤用于编译 CUDA 扩展和 C++ 依赖库
            
            **2. CUDA Toolkit 安装**
            - 下载版本：`cuda_13.0.3_windows`
            - **重要**：使用默认安装路径，不要修改安装位置
            - 默认路径通常为：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`     
            ---
            
            #### 📁 模型文件放置
            
            **模型目录**：`models/trellis2/TRELLIS.2-4B/`
            
            **完整目录结构**：
            ```
            models/trellis2/
            ├── BiRefNet/                    # 背景移除模型
            ├── facebook/                    # DINOv2 特征提取器
            ├── TRELLIS.2-4B/               # 主模型（约15GB）
            │   ├── ckpts/                  # 模型权重文件
            │   ├── pipeline.json           # 管道配置
            │   └── ...
            └── TRELLIS-image-large/        # 图像预处理模型
            ```
            
            **模型下载方式**：
            - HuggingFace Repo: `microsoft/TRELLIS.2-4B`
            - 使用命令：`huggingface-cli download microsoft/TRELLIS.2-4B --local-dir models/trellis2/TRELLIS.2-4B`
            - 或使用镜像加速：设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`
            
            ---
            
            #### 🔧 常见问题
            
            - **缺少 C++ 编译器**：重新安装 VS2022 并确保勾选了 C++ 桌面开发
            - **CUDA 版本不匹配**：确认安装的 CUDA 版本为 13.0，且使用默认路径
            - **模型加载失败**：检查 `models/trellis2/TRELLIS.2-4B/pipeline.json` 是否存在
            - **显存不足**：降低八叉树分辨率（512/1024）或减少纹理分辨率
            """)
        
        gr.Markdown("---")
        
        # 左右布局：左侧输入+参数，右侧输出
        with gr.Row():
            # 左侧列：输入和参数设置
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="输入图像",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )
                
                gr.Markdown("#### 生成参数")
                
                with gr.Row():
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0
                    )
                    randomize_seed = gr.Checkbox(
                        label="随机种子",
                        value=False
                    )
                
                guidance_scale = gr.Slider(
                    label="引导系数",
                    minimum=1.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.5
                )
                
                steps = gr.Slider(
                    label="推理步数",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5
                )
                
                octree_resolution = gr.Dropdown(
                    label="八叉树分辨率",
                    choices=["512", "1024", "1536"],
                    value="1024"
                )
                
                simplify_ratio = gr.Slider(
                    label="网格简化比例",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05
                )
                
                texture_resolution = gr.Slider(
                    label="纹理分辨率",
                    minimum=512,
                    maximum=4096,
                    value=2048,
                    step=256
                )
                
                env_map = gr.Dropdown(
                    label="环境贴图",
                    choices=["无", "森林", "日落"],
                    value="无"
                )
                
                use_flash_attn = gr.Checkbox(
                    label="使用 Flash Attention 加速",
                    value=flash_attn_available,
                    interactive=flash_attn_available,
                    info="启用后可提升注意力计算速度（需要安装 flash-attn）"
                )
                
                generate_btn = gr.Button("生成3D模型", variant="primary", size="lg")
            
            # 右侧列：输出结果
            with gr.Column(scale=1):
                gr.Markdown("#### 生成结果")
                
                status_output = gr.Textbox(label="状态", interactive=False, lines=3)
                
                preview_image = gr.Image(label="预览图", interactive=False, height=300)
                
                output_glb = gr.Model3D(label="3D模型 (GLB格式)", height=400)
                
                gr.Markdown("---")
                with gr.Row():
                    bridge_status = gr.Textbox(label="桥接状态", interactive=False, lines=1, visible=True)
                with gr.Row():
                    send_to_blender_btn = gr.Button("📤 发送GLB到 Blender", variant="secondary", size="sm")
        
        # Connect the button click event
        generate_btn.click(
            fn=generate_3d_from_image,
            inputs=[
                input_image,
                seed,
                randomize_seed,
                guidance_scale,
                steps,
                octree_resolution,
                simplify_ratio,
                texture_resolution,
                env_map,
                use_flash_attn
            ],
            outputs=[status_output, output_glb, preview_image]
        )
        
        # 发送 GLB 到 Blender 按钮事件
        send_to_blender_btn.click(
            fn=send_glb_to_blender,
            inputs=[output_glb],
            outputs=[bridge_status]
        )
    
    # Return tuple: (ui_component, page_title, page_id)
    return [(trellis2_interface, "TRELLIS.2 图生成3D", "trellis2_3d_generator")]


# Register the callback when this module is imported
import modules.script_callbacks as script_callbacks
script_callbacks.on_ui_tabs(on_ui_tabs)
