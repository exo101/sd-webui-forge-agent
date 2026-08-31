import os
import sys
import gradio as gr
import datetime
import numpy as np
from PIL import Image
import torch
try:
    from litelama import LiteLama2
    CLEANER_AVAILABLE = True
except ImportError:
    CLEANER_AVAILABLE = False
    # 不再显示警告信息

# 添加自定义CSS样式
custom_css = ""

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 尝试导入WebUI模块
try:
    from modules.shared import opts
    from modules.ui_components import ToolButton, ResizeHandleRow
    MODULES_AVAILABLE = True
except ImportError:
    # 如果在WebUI环境外运行，创建模拟对象
    class MockOpts:
        def __init__(self):
            self.data = {"cleaner_use_gpu": True}
    
    class MockToolButton:
        def __init__(self, *args, **kwargs):
            pass
    
    class MockResizeHandleRow:
        def __init__(self, *args, **kwargs):
            pass
        
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
    
    opts = MockOpts()
    ToolButton = MockToolButton
    ResizeHandleRow = MockResizeHandleRow
    MODULES_AVAILABLE = False
    print("Warning: Running outside WebUI environment")

# 尝试多种方式导入parameters_copypaste
parameters_copypaste = None
try:
    import modules.generation_parameters_copypaste as parameters_copypaste
except ImportError:
    try:
        from modules import generation_parameters_copypaste as parameters_copypaste
    except ImportError:
        try:
            # 查找正确的导入路径
            import modules
            parameters_copypaste = modules.generation_parameters_copypaste
        except:
            # WebUI Forge 中没有此模块，静默处理
            parameters_copypaste = None

# 直接导入依赖库
try:
    from litelama import LiteLama
    from litelama.model import download_file
    CLEANER_AVAILABLE = True
except Exception as e:
    CLEANER_AVAILABLE = False
    print(f"Warning: Could not import litelama library: {e}")
    print("Image cleaner functionality will be disabled.")


class LiteLama2(LiteLama):
    _instance = None
    
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
        
    def __init__(self, checkpoint_path=None, config_path=None):
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
            
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._model = None
        
        if self._checkpoint_path is None:
            # 使用WebUI的models目录下的cleaner文件夹
            MODELS_PATH = os.path.join(project_root, "models", "diffusers", "cleaner")
            checkpoint_path = os.path.join(MODELS_PATH, "big-lama.safetensors")
            
            # 确保模型目录存在
            os.makedirs(MODELS_PATH, exist_ok=True)
            
            if not os.path.exists(checkpoint_path) or not os.path.isfile(checkpoint_path):
                try:
                    print(f"正在下载big-lama模型到: {checkpoint_path}")
                    download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors", checkpoint_path)
                    print("模型下载完成")
                except Exception as e:
                    print(f"模型下载失败: {e}")
                    print("请手动下载模型文件并放置到正确位置")
            self._checkpoint_path = checkpoint_path
        
        try:
            print(f"正在加载模型: {self._checkpoint_path}")
            self.load(location="cpu")
            self._initialized = True
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

    def to(self, device):
        """重写to方法以添加调试信息"""
        try:
            super().to(device)
        except Exception as e:
            raise e

    def predict(self, image, mask):
        """重写predict方法以添加调试信息"""
        try:
            result = super().predict(image, mask)
            return result
        except Exception as e:
            raise e


def convert_to_white_mask(mask_image):
    """
    将任意颜色的遮罩转换为白色遮罩
    """
    try:
        # 确保输入是PIL图像
        if not hasattr(mask_image, 'convert'):
            return mask_image
            
        # 转换为RGBA模式以处理透明度
        rgba_mask = mask_image.convert('RGBA')
        
        # 创建新的遮罩图像
        from PIL import Image
        white_mask = Image.new('RGB', rgba_mask.size, (0, 0, 0))  # 默认黑色背景
        
        # 获取像素数据
        pixels = rgba_mask.load()
        white_pixels = white_mask.load()
        
        # 遍历所有像素，将非透明和非黑色像素转换为白色
        for x in range(rgba_mask.size[0]):
            for y in range(rgba_mask.size[1]):
                r, g, b, a = pixels[x, y]
                # 如果像素不透明（alpha > 0）且不是黑色，则在遮罩中设为白色
                if a > 0 and (r > 0 or g > 0 or b > 0):  # 非透明且非黑色像素
                    white_pixels[x, y] = (255, 255, 255)  # 白色
        
        return white_mask
    except Exception as e:
        return mask_image  # 出错时返回原始遮罩


def clean_object_init_img_with_mask(init_image_with_mask):
    """
    清理带遮罩的图像
    """
    try:
        # 检查输入
        if init_image_with_mask is None:
            print("输入图像为None")  # 调试信息
            return []
        
        # 从复合输入中提取图像和遮罩
        if isinstance(init_image_with_mask, dict):
            init_image = init_image_with_mask.get("background", None)
            mask_image = init_image_with_mask.get("layers", [None])[0] if init_image_with_mask.get("layers") else None
            print(f"从字典中提取图像: {init_image is not None}, 遮罩: {mask_image is not None}")  # 调试信息
        elif isinstance(init_image_with_mask, (list, tuple)) and len(init_image_with_mask) >= 2:
            init_image, mask_image = init_image_with_mask[0], init_image_with_mask[1]
            print(f"从列表中提取图像: {init_image is not None}, 遮罩: {mask_image is not None}")  # 调试信息
        else:
            init_image = init_image_with_mask
            mask_image = None
            print(f"直接使用输入作为图像: {init_image is not None}")  # 调试信息

        # 检查图像是否有效
        if init_image is None:
            print("输入图像为None")  # 调试信息
            return []
            
        # 确保图像是PIL图像
        if isinstance(init_image, np.ndarray):
            init_image = Image.fromarray(init_image)
        elif not isinstance(init_image, Image.Image):
            print(f"不支持的图像类型: {type(init_image)}")  # 调试信息
            return []
        
        # 确保图像模式为RGB
        print(f"输入图像模式: {init_image.mode}")  # 调试信息
        if init_image.mode != 'RGB':
            init_image = init_image.convert('RGB')
            print(f"转换后图像模式: {init_image.mode}")  # 调试信息
            
        print(f"输入图像尺寸: {init_image.size}")  # 调试信息
        
        # 处理遮罩
        if mask_image is not None:
            if isinstance(mask_image, np.ndarray):
                mask_image = Image.fromarray(mask_image)
            elif not isinstance(mask_image, Image.Image):
                print(f"不支持的遮罩类型: {type(mask_image)}")  # 调试信息
                mask_image = None
        else:
            # 如果没有遮罩，创建一个全白的遮罩
            mask_image = Image.new("L", init_image.size, 255)
            
        print(f"遮罩图像尺寸: {mask_image.size if mask_image else 'None'}")  # 调试信息

        # 确保设备设置正确
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")  # 调试信息

        # 创建LiteLama2实例
        Lama = LiteLama2()
        
        result = None
        try:
            Lama.to(device)
            result = Lama.predict(init_image, mask_image)
            print(f"Lama预测结果: {result}")  # 调试信息
            print(f"预测结果类型: {type(result)}")  # 调试信息
            if result is not None:
                print(f"预测结果尺寸: {result.size}")  # 调试信息
        except Exception as e:
            print(f"Lama预测出错: {str(e)}")  # 调试信息
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            pass
        finally:
            try:
                Lama.to("cpu")
            except Exception as e:
                pass
        
        # 确保返回正确的格式
        if result is not None:
            print("返回包含结果的列表")  # 调试信息
            return [result]
        else:
            print("返回空列表")  # 调试信息
            return []
    except Exception as e:
        print(f"清理函数出错: {str(e)}")  # 调试信息
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        return []


def clean_object(image, mask):
    if not CLEANER_AVAILABLE or image is None or mask is None:
        return [None]
        
    try:
        # 确保我们有PIL图像对象，而不是路径字符串
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)
            
        if isinstance(mask, str):
            from PIL import Image
            mask = Image.open(mask)
        
        # 确保我们有PIL图像对象
        if not hasattr(image, 'convert'):
            return [None]
            
        if not hasattr(mask, 'convert'):
            return [None]
        
        # 转换图像格式
        init_image = image.convert("RGB")
        mask_image = mask.convert("RGB")

        # 获取设备设置
        device_used = opts.data.get("cleaner_use_gpu", True)
        device = "cuda:0" if device_used else "cpu"

        # 创建LiteLama2实例
        Lama = LiteLama2()
        
        result = None
        try:
            Lama.to(device)
            result = Lama.predict(init_image, mask_image)
            print(f"Lama预测结果: {result}")  # 调试信息
            print(f"预测结果类型: {type(result)}")  # 调试信息
            if result is not None:
                print(f"预测结果尺寸: {result.size}")  # 调试信息
        except Exception as e:
            print(f"Lama预测出错: {str(e)}")  # 调试信息
            pass
        finally:
            try:
                Lama.to("cpu")
            except Exception as e:
                pass
        
        # 确保返回正确的格式
        if result is not None:
            return [result]
        else:
            return []
    except Exception as e:
        print(f"清理函数出错: {str(e)}")  # 调试信息
        return []

def send_to_cleaner(result):
    if not result or len(result) == 0 or result[0] is None:
        return None
    try:
        return result[0]
    except Exception as e:
        return None

# 添加一个新的函数来处理Gallery组件的返回值
def process_gallery_output(result):
    """
    处理返回给Gallery组件的结果，确保格式正确
    """
    print(f"处理Gallery输出，接收到的结果: {result}")  # 调试信息
    print(f"结果类型: {type(result)}")  # 调试信息
    
    if result is None or (isinstance(result, list) and len(result) == 0):
        # 返回空列表而不是包含None的列表
        print("返回空列表")  # 调试信息
        return []
    elif isinstance(result, list):
        # 过滤掉None值
        filtered_result = [img for img in result if img is not None]
        print(f"过滤后的结果: {filtered_result}")  # 调试信息
        # 如果结果是包含文件路径和遮罩的元组列表，则提取实际图像对象
        if filtered_result and isinstance(filtered_result[0], tuple):
            # 提取元组中的实际图像对象（通常是第二个元素）
            processed_result = []
            for item in filtered_result:
                if isinstance(item, tuple) and len(item) >= 2 and item[1] is not None:
                    processed_result.append(item[1])  # 使用实际图像对象
                elif isinstance(item, tuple) and len(item) >= 1:
                    processed_result.append(item[0])  # 回退到第一个元素
                else:
                    processed_result.append(item)
            print(f"处理后的元组结果: {processed_result}")  # 调试信息
            return processed_result
        return filtered_result if filtered_result else []
    else:
        # 如果是单个图像对象，包装成列表
        output = [result] if result is not None else []
        print(f"单个图像包装成列表: {output}")  # 调试信息
        return output


# 添加一个新的函数来处理ImageEditor组件的返回值
def process_image_editor_output(result):
    """
    处理返回给ImageEditor组件的结果，确保格式正确
    """
    if result is None or (isinstance(result, list) and len(result) == 0):
        # 返回None而不是空列表
        return None
    elif isinstance(result, list) and len(result) > 0:
        # 返回第一个非None元素
        for img in result:
            if img is not None:
                # 如果是元组，提取第一个元素
                if isinstance(img, tuple):
                    if len(img) > 0:
                        extracted_img = img[0]
                        return extracted_img
                    else:
                        continue
                else:
                    return img
        # 如果所有元素都是None
        return None
    else:
        # 如果是单个图像对象
        if result is not None:
            # 如果是元组，提取第一个元素
            if isinstance(result, tuple):
                if len(result) > 0:
                    extracted_img = result[0]
                    return extracted_img
                else:
                    return None
            else:
                return result
        else:
            return None

def create_cleaner_ui():
    """
    创建图像清理UI模块
    """
    if not CLEANER_AVAILABLE:
        with gr.Group():

            gr.Markdown("LiteLama库未安装，功能不可用。请手动安装依赖：\n\n"
                       "1. 关闭WebUI\n"
                       "2. 打开命令行并运行: `pip install litelama`\n"
                       "3. 重新启动WebUI\n\n"
                       "如果仍有问题，请检查Python环境或尝试降级numpy版本:\n"
                       "pip install numpy==1.24.4")
        return None

    with gr.Column():  # 修改为gr.Column以与其他UI组件保持一致
        # 应用自定义CSS样式
        gr.Markdown(f"<style>{custom_css}</style>", visible=False)
        gr.Markdown("## 图像清理器 (Cleaner)")
        
        # 添加模型下载说明
        gr.Markdown("## 模型下载说明")
        gr.Markdown("""
        - **模型存储位置**: `WebUI根目录/models/cleaner`
        - **所需模型文件**: `big-lama.safetensors`
        - **下载方式**: 首次使用时会自动下载
        - **下载来源**: 所有模型均已包含在整合包中，如需更新请从群主网盘中获取
        """)
        
        with ResizeHandleRow(equal_height=False):
            init_img_with_mask = gr.ImageMask(
                label="带遮罩的清理图像", 
                elem_id="xykc_cleanup_img2maskimg", 
                sources=["upload"],
                interactive=True, 
                type="pil", 
                container=True,
                height=600,
                brush=gr.Brush(default_size=32, default_color="#FFFFFF"),
            )
            
            with gr.Column(elem_id="xykc_cleanup_gallery_container"):
                clean_button = gr.Button("清理图像", variant="primary", elem_id="xykc_clean_button")
                result_gallery = gr.Gallery(
                    label='输出结果', 
                    show_label=False, 
                    elem_id="xykc_cleanup_gallery", 
                    preview=True, 
                    height=400,  # 调整高度以匹配Sketchpad组件
                    object_fit="contain"  # 确保图像完整显示在容器内
                )
                
                # 添加打开输出目录按钮和保存状态显示
                with gr.Row():
                    open_cleaner_output_dir_btn = gr.Button("打开输出目录")
                
                # 添加保存状态文本框
                save_status = gr.Textbox(label="保存状态", interactive=False, visible=True)
                
                def auto_save_cleaned_images(images):
                    """自动保存清理后的图像到输出目录"""
                    print(f"自动保存函数接收到的图像数据: {images}")  # 调试信息
                    print(f"图像数据类型: {type(images)}")  # 调试信息
                    
                    # 处理不同格式的输入数据
                    if images is None:
                        return "没有图像需要保存: 图像数据为None"
                    
                    # 确保我们处理的是列表格式
                    if not isinstance(images, list):
                        images = [images]
                    
                    print(f"处理后的图像列表长度: {len(images)}")  # 调试信息
                    for i, img in enumerate(images):
                        print(f"图像 {i} 类型: {type(img)}")  # 调试信息
                        if img is not None and hasattr(img, 'size'):
                            print(f"图像 {i} 尺寸: {img.size}")  # 调试信息
                    
                    try:
                        # 创建保存目录 - 使用WebUI的outputs目录
                        from modules import shared
                        save_dir = os.path.join(shared.data_path, "outputs", "cleaner")
                        os.makedirs(save_dir, exist_ok=True)
                        print(f"保存目录: {save_dir}")  # 调试信息
                        
                        saved_files = []
                        
                        for i, img in enumerate(images):
                            print(f"处理图像 {i}: {img}")  # 调试信息
                            
                            # 处理元组格式的图像数据
                            if isinstance(img, tuple) and len(img) >= 1:
                                # 提取实际的图像对象（通常是第二个元素）
                                if len(img) >= 2 and img[1] is not None:
                                    img = img[1]  # 使用实际图像对象
                                else:
                                    img = img[0]  # 回退到第一个元素
                                print(f"从元组中提取图像: {img}")  # 调试信息
                            
                            # 如果是文件路径，尝试打开图像
                            if isinstance(img, str) and os.path.isfile(img):
                                try:
                                    img = Image.open(img)
                                    print(f"从文件路径加载图像: {img}")  # 调试信息
                                except Exception as e:
                                    print(f"从文件路径加载图像失败: {str(e)}")  # 调试信息
                                    continue
                            
                            if img is not None:
                                # 生成文件名
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"cleaned_image_{timestamp}_{i+1}.png"
                                save_path = os.path.join(save_dir, filename)
                                print(f"保存路径: {save_path}")  # 调试信息
                                
                                # 统一处理不同类型的图像对象，确保转换为RGB模式
                                try:
                                    # 处理元组或列表中的图像数据
                                    if isinstance(img, (tuple, list)) and len(img) > 0:
                                        img = img[0]  # 取第一个元素作为图像

                                    # 转换为PIL图像（如果需要）
                                    if isinstance(img, np.ndarray):
                                        print("转换numpy数组为PIL图像")  # 调试信息
                                        img = Image.fromarray(img)
                                    elif isinstance(img, str) and os.path.isfile(img):
                                        print("从文件路径加载图像")  # 调试信息
                                        img = Image.open(img)

                                    # 确保是有效的PIL图像对象
                                    if hasattr(img, 'convert'):
                                        # 统一转换为RGB模式
                                        print(f"图像模式: {img.mode}")  # 调试信息
                                        if img.mode != 'RGB':
                                            img = img.convert('RGB')
                                            print(f"已转换为RGB模式")  # 调试信息
                                        
                                        # 保存图像
                                        img.save(save_path)
                                        saved_files.append(save_path)
                                        print(f"已保存图像: {save_path}")  # 调试信息
                                    else:
                                        print(f"无法处理的图像类型: {type(img)}")  # 调试信息
                                        continue
                                except Exception as e:
                                    print(f"处理单个图像时出错: {str(e)}")  # 调试信息
                                    continue
                                
                        if saved_files:
                            return f"已自动保存 {len(saved_files)} 张图像到: {save_dir}"
                        else:
                            return "没有有效的图像需要保存: 图像列表中没有有效图像"
                    except Exception as e:
                        print(f"保存图像时出错: {str(e)}")  # 调试信息
                        import traceback
                        traceback.print_exc()  # 打印详细错误信息
                        return f"保存图像时出错: {str(e)}"
                
                def open_cleaner_output_dir():
                    """打开图像清理输出目录"""
                    from modules import shared
                    output_dir = os.path.join(shared.data_path, "outputs", "cleaner")
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
                
                # 移除保存按钮的点击事件处理
                
                open_cleaner_output_dir_btn.click(fn=open_cleaner_output_dir, inputs=[], outputs=[])

                send_to_cleaner_button = gr.Button("发送回清理器", elem_id="xykc_send_to_cleaner")

        # 设置事件处理
        clean_button.click(
            fn=lambda x: process_gallery_output(clean_object_init_img_with_mask(x)),
            inputs=[init_img_with_mask],
            outputs=[result_gallery],
        ).then(
            fn=auto_save_cleaned_images,
            inputs=[result_gallery],
            outputs=[save_status]
        )

        send_to_cleaner_button.click(
            fn=process_image_editor_output,
            inputs=[result_gallery],
            outputs=[init_img_with_mask]
        )

    # 返回UI组件字典，以便在主程序中引用
    return {
        "init_img_with_mask": init_img_with_mask,
        "result_gallery": result_gallery,
    }


def create_cleaner_module():
    """
    创建图像清理UI模块的入口函数
    """
    return create_cleaner_ui()
