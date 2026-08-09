import gradio as gr
import os
import glob
from PIL import Image

def create_image_management_module():
    """创建图像管理模块并返回组件结构"""
    result = {}

    with gr.Accordion("图像管理", open=False):
        with gr.Row():
            dir_input = gr.Textbox(
                label="图片目录路径",
                placeholder="输入图片目录路径",
                scale=3
            )
            load_dir_btn = gr.Button(
                "加载目录",
                scale=1
            )

        gallery = gr.Gallery(
            label="图片预览",
            show_label=True,
            elem_id="gallery",
            columns=4,
            height=400,
            visible=True,
            object_fit="contain"
        )


        def load_images_from_dir(dir_path):
            """从指定目录加载图像文件并返回适合Gallery组件显示的数据结构"""
            if not os.path.isdir(dir_path):
                print(f"[ERROR] 目录不存在: {dir_path}")
                return []

            supported_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
            image_files = []

            # 搜索支持的图像格式
            for ext in supported_extensions:
                search_pattern = os.path.join(dir_path, f"*{ext}")
                matches = glob.glob(search_pattern, recursive=False)
                
                if matches:
                    print(f"[DEBUG] 找到 {len(matches)} 个 {ext} 文件")
                    image_files.extend(matches)

            # 构建Gallery组件所需的显示数据
            result = []
            for img_path in image_files:
                try:
                    # 验证文件是否为有效图像
                    with Image.open(img_path) as img:
                        img.verify()  # 确保是有效的图像文件
                    result.append((os.path.normpath(img_path), os.path.basename(img_path)))
                except Exception as e:
                    print(f"[ERROR] 图像文件损坏或不可读: {img_path} - {str(e)}")
                    continue

            if not result:
                print("[WARNING] 没有找到有效的图像文件")
                return []  # 返回空列表而不是默认提示图

            print(f"[INFO] 总共找到 {len(result)} 个有效图像文件")
            return result

        load_dir_btn.click(
            fn=load_images_from_dir,
            inputs=[dir_input],
            outputs=[gallery]
        )

        # 将关键组件保存到result中供外部调用
        result["dir_input"] = dir_input
        result["load_dir_btn"] = load_dir_btn
        result["gallery"] = gallery

    return result  # 返回组件集合