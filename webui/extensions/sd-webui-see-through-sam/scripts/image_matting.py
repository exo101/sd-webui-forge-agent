import gradio as gr
from PIL import Image
import os
import sys

from modules import shared

# 定义支持的高级抠图模型列表 (需要单独安装依赖)
ADVANCED_MATTING_MODELS = {
    "InSPyReNet-Base (金字塔细化)": "inspyrenet-base",
}

def create_image_matting_module():
    """创建智能抠图模块并返回组件结构"""
    result = {}

    # 移除Accordion包装，直接创建UI组件以适应新的布局
    with gr.Row():
        rm_upload = gr.Files(
            label="上传图片", 
            file_types=["image"],
            file_count="multiple",
            scale=3,
            height=400
        )
        
        with gr.Column(scale=1):
            rm_preview = gr.Image(
                label="图片预览",
                visible=False,
                height=300,
                interactive=False
            )
            
            rm_bg_color = gr.ColorPicker(
                label="背景颜色", 
                value="#FFFFFF",
                interactive=True,
                visible=True,
                show_label=True,
                container=True
            )
            rm_bg_transparent = gr.Button(
                "透明背景",
                size="sm",
                variant="secondary"
            )
            
            # 添加模型选择器
            rm_model_select = gr.Dropdown(
                choices=list(ADVANCED_MATTING_MODELS.keys()),
                value="InSPyReNet-Base (金字塔细化)",
                label="选择抠图模型",
                info="需手动安装依赖: pip install transparent-background"
            )
            
            # 添加模型说明（折叠起来节省空间）
            with gr.Accordion("📖 查看模型说明", open=False):
                gr.Markdown("**模型说明**:")
                gr.Markdown("- **InSPyReNet**: 金字塔细化，高分辨率图像专业级质量")
                gr.Markdown("")
                gr.Markdown("**模型下载说明**:")
                gr.Markdown("- InSPyReNet模型: `models/diffusers/InSPyReNet/` (需安装: `pip install transparent-background`)")
                gr.Markdown("- 首次使用时会自动从魔搭社区 ModelScope 下载")

    rm_process_btn = gr.Button(
        "开始处理",
        size="lg",
        variant="primary",
        elem_classes="orange-button",
        scale=1
    )
    
    # 添加打开输出目录按钮
    open_output_dir_btn = gr.Button("打开输出目录")
    
    def open_image_matting_output_dir():
        """打开智能抠图输出目录"""
        from modules import shared
        output_dir = os.path.join(shared.data_path, "outputs", "image-matting")
        os.makedirs(output_dir, exist_ok=True)
        import subprocess
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            print(f"打开目录失败: {e}")

    rm_output = gr.Gallery(
        label="处理结果",
        columns=4,
        height=300,
        visible=True,
        object_fit="contain"
    )
    
    rm_progress = gr.Textbox(
        label="处理进度",
        value="等待处理...",
        interactive=False,
        scale=1
    )
    
    # 将关键组件保存到result中供外部调用
    result["rm_upload"] = rm_upload
    result["rm_preview"] = rm_preview
    result["rm_bg_color"] = rm_bg_color
    result["rm_bg_transparent"] = rm_bg_transparent
    result["rm_model_select"] = rm_model_select
    result["rm_process_btn"] = rm_process_btn
    result["rm_output"] = rm_output
    result["rm_progress"] = rm_progress

    def update_preview(files):
        """根据上传的文件数量更新预览"""
        if files and len(files) == 1:
            # 只有一张图片时显示预览
            return gr.update(value=files[0].name, visible=True)
        else:
            # 多张图片或没有图片时不显示预览
            return gr.update(value=None, visible=False)

    def process_images(files, bg_color, model_name):
        if not files:
            raise gr.Error("请先上传图片")

        actual_model_name = ADVANCED_MATTING_MODELS.get(model_name)
        print(f"[INFO] 使用抠图模型: {model_name} ({actual_model_name})")
        return process_with_advanced_model(files, bg_color, actual_model_name)

    def process_with_advanced_model(files, bg_color, model_type):
        """使用高级抠图模型处理图像"""
        from modules import shared
        
        processed_images = []
        save_dir = os.path.join(shared.data_path, "outputs", "image-matting")
        os.makedirs(save_dir, exist_ok=True)

        total = len(files)
        
        if model_type.startswith("inspyrenet"):
            return process_with_inspyrenet(files, bg_color, model_type)
        else:
            # 其他高级模型暂时返回提示信息
            error_msg = f"""
⚠️ 高级模型 '{model_type}' 需要额外安装依赖！

安装方法:
1. InSPyReNet: pip install transparent-background

或者查看插件文档获取详细安装指南。

目前支持: InSPyReNet 模型。
            """
            raise gr.Error(error_msg)


    def process_with_inspyrenet(files, bg_color, model_type):
        """使用 InSPyReNet 模型处理图像"""
        from modules import shared
        import torch
        from PIL import Image
        import numpy as np
        
        # 确保 InSPyReNet 相关目录存在
        inspyrenet_dir = os.path.join(shared.data_path, "models", "diffusers", "InSPyReNet")
        os.makedirs(inspyrenet_dir, exist_ok=True)
        
        # 确保输出目录存在
        save_dir = os.path.join(shared.data_path, "outputs", "image-matting")
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取设备
        def get_device():
            try:
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except:
                return "cpu"
        
        device = get_device()
        print(f"[INFO] InSPyReNet 使用设备: {device}")
        
        # 加载 InSPyReNet 模型
        try:
            # 使用 transparent-background 包加载 InSPyReNet
            try:
                # transparent-background 可能通过 Hugging Face Hub 和 Torch Hub
                # 下载模型，全部指向 WebUI 模型目录，避免使用 C 盘默认缓存。
                import os as os_module
                cache_vars = {
                    # transparent-background 专用配置根目录；该库会在此目录下
                    # 创建 .transparent-background 并保存 InSPyReNet 权重。
                    'TRANSPARENT_BACKGROUND_FILE_PATH': inspyrenet_dir,
                    'TRANSPARENT_BACKGROUND_CACHE': inspyrenet_dir,
                    'TRANSPARENT_BACKGROUND_MODELS': inspyrenet_dir,
                    'HF_HOME': inspyrenet_dir,
                    'HF_HUB_CACHE': inspyrenet_dir,
                    'HUGGINGFACE_HUB_CACHE': inspyrenet_dir,
                    'TRANSFORMERS_CACHE': inspyrenet_dir,
                    'TORCH_HOME': inspyrenet_dir,
                }
                for name, path in cache_vars.items():
                    os_module.environ[name] = path

                # ModelScope 版本的 transparent-background 权重放在本地缓存目录。
                # 仅在默认权重不存在时下载，避免每次启动重复下载。
                ckpt_path = os.path.join(inspyrenet_dir, ".transparent-background", "ckpt_base.pth")
                if not os.path.isfile(ckpt_path):
                    print("[INFO] 正在从 ModelScope 下载 InSPyReNet: "
                          "shiertier/ComfyUI-transparent-background")
                    from modelscope import snapshot_download
                    snapshot_download(
                        model_id="shiertier/ComfyUI-transparent-background",
                        local_dir=os.path.join(inspyrenet_dir, ".transparent-background"),
                    )

                # 必须在设置环境变量之后导入/实例化 Remover。
                from transparent_background import Remover
                
                # 加载模型
                print(f"[INFO] 加载 InSPyReNet 模型...")
                remover = Remover(device=device)
                print(f"[INFO] InSPyReNet 模型加载成功")
                use_transparent_bg = True
                
            except ImportError as e:
                raise gr.Error(
                    "加载 InSPyReNet 模型失败：当前 WebUI Python 环境缺少 "
                    f"transparent-background。请重启 WebUI 自动安装，或执行：\n"
                    f"{sys.executable} -m pip install transparent-background\n"
                    f"原始错误：{e}"
                )
            
        except Exception as e:
            raise gr.Error(f"加载 InSPyReNet 模型失败: {str(e)}")
        
        processed_images = []
        total = len(files)
        
        for i, file in enumerate(files):
            try:
                # 打开图像
                orig_image = Image.open(file.name).convert("RGB")
                
                # 使用模型处理
                if use_transparent_bg:
                    # 使用 transparent-background 的 Remover
                    # 处理图像
                    result = remover.process(orig_image, type='rgba')
                    
                    # 转换为 PIL Image
                    if isinstance(result, np.ndarray):
                        img_rgba = Image.fromarray(result)
                    else:
                        img_rgba = result
                    
                    # 应用背景颜色
                    if bg_color != "transparent":
                        # 有背景颜色
                        bg = Image.new("RGBA", orig_image.size, bg_color)
                        bg.paste(img_rgba, (0, 0), img_rgba)
                        img_final = bg.convert("RGB")
                    else:
                        # 透明背景
                        img_final = img_rgba
                else:
                    raise gr.Error("InSPyReNet 处理失败: 无法加载模型")
                
                # 保存结果
                filename = os.path.splitext(os.path.basename(file.name))[0] + f"_inspyrenet_processed.png"
                save_path = os.path.join(save_dir, filename)
                img_final.save(save_path)
                
                # 清理资源
                orig_image.close()
                img_final.close()
                
                processed_images.append(save_path)
                print(f"[INFO] 已处理 {i+1}/{total}: {os.path.basename(file.name)}")
                
            except Exception as e:
                print(f"[ERROR] 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not processed_images:
            raise gr.Error("InSPyReNet 处理失败，请检查日志获取详细信息")
        
        success_count = len(processed_images)
        return processed_images, f"处理完成，共处理 {success_count}/{total} 张图片"

    def set_transparent():
        return "transparent"

    # 绑定上传事件到预览更新函数
    rm_upload.change(
        fn=update_preview,
        inputs=[rm_upload],
        outputs=[rm_preview]
    )

    rm_bg_transparent.click(
        fn=set_transparent,
        inputs=[],
        outputs=[rm_bg_color]
    )

    rm_process_btn.click(
        fn=process_images,
        inputs=[rm_upload, rm_bg_color, rm_model_select],
        outputs=[rm_output, rm_progress]
    )

    open_output_dir_btn.click(fn=open_image_matting_output_dir, inputs=[], outputs=[])

    return result  # 返回组件集合
