import os
import numpy as np
import torch
import gradio as gr
import cv2
from collections import OrderedDict
from PIL import Image
import copy
import datetime
import sys

# 假设已安装 segment_anything 库，并且可以导入真实模型
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    

def _download_sam_checkpoint(save_path, model_type):
    """从多个镜像自动下载SAM模型检查点
    
    Args:
        save_path: 模型保存路径
        model_type: 模型类型 (vit_h/vit_l/vit_b)
    
    Returns:
        bool: 下载是否成功
    """
    import urllib.request
    
    # 模型文件名映射
    model_filenames = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    
    filename = model_filenames.get(model_type)
    if not filename:
        print(f"[SAM Matting] 未知的模型类型: {model_type}")
        return False
    
    # 如果文件已存在（如之前下载中断留下的），先删除
    if os.path.lexists(save_path):
        print(f"[SAM Matting] 删除已存在的文件: {save_path}")
        os.remove(save_path)
    
    # 多镜像下载列表
    env_mirror = os.environ.get("SAM_DOWNLOAD_MIRROR", "")
    mirror_urls = []
    
    if env_mirror:
        mirror_urls.append(env_mirror + filename)
    
    # 默认镜像列表（按优先级排序）
    mirror_urls += [
        f"https://dl.fbaipublicfiles.com/segment_anything/{filename}",
        f"https://hf-mirror.com/facebook/sam-{model_type.replace('_', '-')}/resolve/main/{filename}",
    ]
    
    print(f"[SAM Matting] 模型文件不存在，开始自动下载...")
    last_error = None
    for url in mirror_urls:
        try:
            print(f"[SAM Matting] 尝试从镜像下载: {url}")
            print(f"[SAM Matting] 保存到: {save_path}")
            
            def report_progress(block_count, block_size, total_size):
                downloaded = block_count * block_size / (1024 * 1024)
                if total_size > 0:
                    total_mb = total_size / (1024 * 1024)
                    percent = min(100, downloaded / total_mb * 100)
                    print(f"\r[SAM Matting] 下载中: {percent:.1f}% ({downloaded:.1f}/{total_mb:.1f} MB)", end="")
                else:
                    print(f"\r[SAM Matting] 下载中: {downloaded:.1f} MB...", end="")
            
            urllib.request.urlretrieve(url, save_path, reporthook=report_progress)
            print()
            
            # 验证文件大小
            file_size = os.path.getsize(save_path)
            if file_size > 1024 * 1024:  # 至少 1MB 才认为有效
                print(f"[SAM Matting] 下载成功: {save_path} ({file_size / (1024*1024):.1f} MB)")
                return True
            else:
                print(f"[SAM Matting] 文件太小 ({file_size} bytes)，可能无效")
                os.remove(save_path)
                continue
                
        except Exception as e:
            print(f"\n[SAM Matting] 下载失败: {e}")
            last_error = e
            if os.path.exists(save_path):
                os.remove(save_path)
            continue
    
    print(f"[SAM Matting] 所有镜像下载失败。最后一个错误: {last_error}")
    print(f"[SAM Matting] 您可以手动下载模型:")
    print(f"[SAM Matting]   {mirror_urls[0] if not env_mirror else env_mirror + filename}")
    print(f"[SAM Matting] 并放置到: {os.path.dirname(save_path)}")
    return False


def initialize_sam_model(model_type=None):
    """初始化SAM模型"""
    if not SAM_AVAILABLE:
        print("无法找到 segment_anything 库，请先安装依赖")
        return None
        
    # 定义模型文件路径 - 使用WebUI的models目录 (models/sam)
    # 通过环境变量或默认方式获取WebUI根目录
    webui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_dir = os.path.join(webui_root, "models", "sams")
    os.makedirs(model_dir, exist_ok=True)
    model_paths = {
        "vit_h": os.path.join(model_dir, "sam_vit_h_4b8939.pth"),  # 2.38G
        "vit_l": os.path.join(model_dir, "sam_vit_l_0b3195.pth")   # 1.25G
    }
    
    # 如果没有指定模型类型，则查找第一个存在的模型
    if model_type is None:
        for m_type, m_path in model_paths.items():
            if os.path.exists(m_path):
                model_type = m_type
                print(f"找到模型文件: {m_type} ({m_path})")
                break
        else:
            # 如果没有找到任何模型文件，尝试自动下载 vit_h
            print("未找到任何模型文件，尝试自动下载...")
            model_type = "vit_h"
            model_path = model_paths[model_type]
            if not _download_sam_checkpoint(model_path, model_type):
                print("自动下载失败，请确保模型文件已下载并放置在正确路径下")
                return None
    
    model_path = model_paths.get(model_type, model_paths["vit_h"])
    
    # 检查指定的模型文件是否存在，不存在则尝试自动下载
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("尝试自动下载...")
        if not _download_sam_checkpoint(model_path, model_type):
            print("自动下载失败，请确保模型文件已下载并放置在正确路径下")
            return None
    
    try:
        # 加载真实的 SAM 模型
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建预测器用于点分割
        predictor = SamPredictor(sam)
        
        print(f"成功加载模型: {model_type}")
        return sam, predictor  # 返回预测器用于点分割
    except Exception as e:
        print(f"SAM模型初始化失败: {str(e)}")
        return None

# 不在模块加载时初始化模型，改为在实际使用时初始化
sam_components = None  # 延迟初始化，在需要时才加载模型

def save_segmentation_results(results, segmentation_type):
    """
    保存分割结果到插件目录
    
    Args:
        results: 分割结果列表，包含PIL图像对象
        segmentation_type: 分割类型（"point_segmentation" 或 "random_segmentation"）
    """
    try:
        # 创建保存目录 - 更改为WebUI的outputs目录
        webui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        save_dir = os.path.join(webui_root, "outputs", "segment-anything", segmentation_type)
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳用于文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存每个结果
        saved_count = 0
        for i, result in enumerate(results):
            if result is not None:
                # 生成文件名
                filename = f"{segmentation_type}_{timestamp}_{i+1}.png"
                filepath = os.path.join(save_dir, filename)
                
                # 保存图像
                result.save(filepath, "PNG")
                print(f"已保存分割结果: {filepath}")
                saved_count += 1
        
        print(f"总共保存了 {saved_count} 个分割结果到: {save_dir}")
        
    except Exception as e:
        print(f"保存分割结果时出错: {e}")
        import traceback
        traceback.print_exc()


# 存储原始图像
original_image = None
points_state = []  # 存储当前标记点

def create_mask_output(image_np, masks):
    """创建mask输出图像，只返回最终的分割图像"""
    global points_state
    print(f"Creating output image, received {len(masks)} masks")
    print(f"Masks shape: {masks.shape}, image shape: {image_np.shape}")
    
    matted_images = []  # 只需要最终的分割图像
    for i, mask in enumerate(masks):
        # 确保只处理最多6个mask
        if i >= 6:
            print("Reached maximum number of masks (6), stopping processing")
            break
            
        # 获取mask的2D表示
        try:
            mask_2d = np.any(mask, axis=0)
        except Exception as e:
            print(f"Error processing mask {i}: {e}")
            continue
            
        print(f"Processing mask {i}, shape: {mask_2d.shape}, non-zero pixels: {np.count_nonzero(mask_2d)}")
        
        # 检查mask是否为空
        if np.count_nonzero(mask_2d) == 0:
            print(f"Skipping empty mask {i}")
            continue
            
        # 检查mask维度是否正确
        if mask_2d.ndim != 2:
            print(f"Skipping invalid mask {i} with wrong dimensions: {mask_2d.ndim}")
            continue
            
        # 检查图像形状是否匹配
        if mask_2d.shape[0] != image_np.shape[0] or mask_2d.shape[1] != image_np.shape[1]:
            print(f"Skipping mask {i} with mismatched dimensions. Mask: {mask_2d.shape}, Image: {image_np.shape}")
            continue
            
        try:
            # matted_images包含透明背景的原图，不移除标记点，因为结果图像中不应包含标记点
            if image_np.shape[2] == 4:
                # 图像已经有alpha通道
                image_np_copy = copy.deepcopy(image_np)
                image_np_copy[~mask_2d, 3] = 0
                # 不在结果图像中移除标记点，因为结果图像中本就不应该包含标记点
                matted_images.append(Image.fromarray(image_np_copy))
            else:
                # 图像没有alpha通道，需要添加
                image_np_copy = copy.deepcopy(image_np)
                # 添加alpha通道
                alpha_channel = np.full((image_np_copy.shape[0], image_np_copy.shape[1]), 255, dtype=np.uint8)
                alpha_channel[~mask_2d] = 0
                image_with_alpha = np.dstack([image_np_copy, alpha_channel])
                # 不在结果图像中移除标记点，因为结果图像中本就不应该包含标记点
                matted_images.append(Image.fromarray(image_with_alpha))
        except Exception as e:
            print(f"Error creating output for mask {i}: {e}")
            continue
            
    print(f"Generated {len(matted_images)} output images")
    # 只返回最终的分割图像
    # 过滤掉无效图像（尺寸为0的图像）
    filtered_images = []
    for img in matted_images:
        if img is not None and img.size[0] > 0 and img.size[1] > 0:
            filtered_images.append(img)
        else:
            print(f"Skipping invalid image: size={img.size if img is not None else 'None'}")
    
    print(f"Returning {len(filtered_images)} valid images after filtering")
    return filtered_images

def point_segmentation(img, points, model_type="vit_h"):
    """基于标记点的分割实现，每个点独立分割，只保留前两个掩码，并保存结果到插件目录"""
    global sam_components, points_state, original_image
    
    # 更新points_state
    points_state = points
    
    # 如果模型尚未初始化，则在此时初始化
    if sam_components is None:
        sam_components = initialize_sam_model(model_type)
        
    # 如果模型仍然不可用，返回错误信息
    if sam_components is None or not SAM_AVAILABLE:
        gr.Warning("无法初始化SAM模型，请确保模型文件已下载并放置在正确路径下")
        return []
    
    if img is None:
        gr.Warning("请先上传图片")
        return []
    
    # 如果没有标记点，返回错误信息
    if len(points) == 0:
        gr.Warning("请至少添加一个标记点")
        return []
    
    try:
        # 获取SAM组件
        sam, predictor = sam_components
        
        # 使用原始图像进行分割，而不是带标记点的图像
        segmentation_img = original_image if original_image is not None else img
        
        # 转换图像格式
        if segmentation_img.ndim == 3 and segmentation_img.shape[2] == 4:
            img_rgb = cv2.cvtColor(segmentation_img, cv2.COLOR_BGRA2RGB)
        elif segmentation_img.ndim == 2:
            img_rgb = cv2.cvtColor(segmentation_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = segmentation_img[..., :3]
        
        # 设置图像
        predictor.set_image(img_rgb)
        
        # 为每个点独立进行分割
        all_masks = []
        all_scores = []
        
        for point in points:
            # 准备单个点的坐标和标签
            input_points = np.array([point])
            input_labels = np.array([1])  # 1表示正点
            
            # 执行分割
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            print(f"Point SAM为点 {point} 生成 {len(masks)} 个mask，scores: {scores}")
            
            # 检查mask是否有效
            if masks is not None and len(masks) > 0:
                # 如果masks是列表或元组，转换为numpy数组
                if isinstance(masks, (list, tuple)):
                    masks = np.array(masks)
                
                # 只保留前两个掩码（删除第三个较大的掩码）
                if masks.shape[0] > 2:
                    masks = masks[:2]  # 只保留前两个掩码
                    scores = scores[:2] if len(scores) > 2 else scores
                
                # 确保mask具有正确的维度
                if masks.ndim == 3:
                    # SAM返回的masks形状为 (num_masks, height, width)
                    # 我们需要将其转换为 (num_masks, 1, height, width) 以与create_mask_output兼容
                    masks = masks[:, None, ...]
                elif masks.ndim == 4:
                    # masks已经是正确的形状，但只保留前两个
                    masks = masks[:2] if masks.shape[0] > 2 else masks
                else:
                    continue  # 跳过无效的mask
                
                # 添加到总结果中
                all_masks.append(masks)
                all_scores.extend(scores[:2] if len(scores) > 2 else scores)
        
        if not all_masks:
            gr.Warning("未能生成有效的分割结果")
            return []
        
        # 合并所有点的mask
        combined_masks = np.concatenate(all_masks, axis=0)
        print(f"总共生成 {len(combined_masks)} 个mask（已删除较大的第三个掩码）")
        
        # 计算每个掩码的面积并删除最小的掩码
        if len(combined_masks) > 1:
            mask_areas = []
            for i, mask in enumerate(combined_masks):
                try:
                    mask_2d = np.any(mask, axis=0)
                    area = np.count_nonzero(mask_2d)
                    mask_areas.append((area, i))
                    print(f"Mask {i} 面积: {area}")
                except Exception as e:
                    print(f"计算掩码 {i} 面积失败: {e}")
                    mask_areas.append((0, i))
            
            # 按面积排序，找到最小的掩码
            if mask_areas:
                mask_areas.sort(key=lambda x: x[0])
                smallest_mask_index = mask_areas[0][1]
                print(f"删除最小的掩码 {smallest_mask_index}，面积: {mask_areas[0][0]}")
                # 删除最小的掩码
                combined_masks = np.delete(combined_masks, smallest_mask_index, axis=0)
                print(f"删除后剩余 {len(combined_masks)} 个掩码")
        
        # 使用create_mask_output函数处理结果，使用原始图像而不是带标记点的图像
        results = create_mask_output(segmentation_img, combined_masks)
        
        # 过滤掉空的图像结果并去重
        filtered_results = []
        seen_hashes = set()
        
        for img_result in results:
            # 检查图像是否有效
            if img_result is not None:
                # 检查PIL图像的尺寸
                if hasattr(img_result, 'size') and img_result.size[0] > 0 and img_result.size[1] > 0:
                    # 进一步检查是否包含有效像素
                    img_array = np.array(img_result)
                    if img_array.ndim >= 2 and np.sum(img_array) > 0:
                        # 创建图像的哈希值用于去重
                        img_hash = hash(img_array.tobytes())
                        if img_hash not in seen_hashes:
                            seen_hashes.add(img_hash)
                            filtered_results.append(img_result)
        
        if len(filtered_results) == 0:
            gr.Warning("未能生成有效的分割结果")
            return []
        
        # 保存结果到插件目录
        save_segmentation_results(filtered_results, "point_segmentation")
        
        print(f"Returning {len(filtered_results)} valid results")
        # 限制返回结果数量为6个
        return filtered_results[:6]
        
    except Exception as e:
        gr.Warning(f"点分割过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []



def create_sam_segmentation():
    """创建Segment Anything图像分割界面组件"""
    global sam_components, original_image
    
    # 只有在用户实际使用功能时才尝试初始化模型
    if sam_components is None:
        # 可以选择在这里初始化默认模型，或者在用户第一次点击运行时初始化
        pass
    
    with gr.Row():
        gr.Markdown("## Segment Anything 模型参数说明")
    with gr.Row():
        gr.Markdown("""
        - **sam_vit_h_4b8939**: 最大模型，精度最高，但需要更多显存
        - **sam_vit_l_0b3195**: 中等模型，精度和资源消耗的平衡
        """)
    
    with gr.Row():
        sam_input = gr.Image(type="numpy", label="上传图像", interactive=True, height=400)
        sam_output = gr.Gallery(label="分割结果", columns=4, object_fit="contain")
    
    with gr.Row():
        model_type = gr.Dropdown(
            label="选择模型",
            choices=[("vit_h (2.38G)", "vit_h"), ("vit_l (1.25G)", "vit_l")],
            value="vit_h",
            interactive=True
        )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 手动分割")
            gr.Markdown("在上传图像点击添加标记点进行分割，边缘瑕疵通过图生图进行修复")
            point_run_button = gr.Button("根据标记点运行分割")
            point_clear_button = gr.Button("清除标记点和结果")
    
    # 添加打开输出目录按钮
    with gr.Row():
        open_output_folder_button = gr.Button("打开输出目录")
    
    # 用于存储标记点的状态变量
    points_state = gr.State([])
    
    def change_model(model_type):
        """切换模型"""
        global sam_components
        sam_components = initialize_sam_model(model_type)
        return f"模型已切换到: {model_type}"
    
    def save_original_image(img):
        """保存原始图像"""
        global original_image
        if img is not None:
            original_image = img.copy()
        return img
    
    def add_point(evt: gr.SelectData, img, points):
        """添加正点"""
        global original_image, points_state
        if img is None:
            return img, points
    
        # 保存原始图像（如果还没有保存）
        if original_image is None:
            original_image = img.copy()
        
        # 获取点击坐标
        x, y = evt.index
        point = [x, y]
        
        # 添加点
        points = points + [point]
        points_state = points  # 更新全局points_state
        
        # 在图像上绘制点 (增大点的大小)
        img_copy = img.copy()
        # 绘制所有点（红色）- 增大点的大小从5到12，增加边框使其更清晰
        for p in points:
            cv2.circle(img_copy, (p[0], p[1]), 12, (0, 0, 255), -1)  # 增大点的半径
            cv2.circle(img_copy, (p[0], p[1]), 15, (255, 255, 255), 2)  # 添加白色边框使点更清晰
    
        return img_copy, points
    
    def clear_all_points(img):
        """清除所有标记点"""
        global original_image
        # 恢复原始图像并清除点列表
        if original_image is not None:
            return original_image.copy(), []
        elif img is not None:
            return img.copy(), []
        else:
            return None, []
    
    def open_output_folder():
        """打开输出目录"""
        try:
            # 获取WebUI根目录
            webui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            # 构造输出目录路径
            output_dir = os.path.join(webui_root, "outputs", "segment-anything")
            # 确保目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 尝试使用系统默认方式打开文件夹
            import platform
            import subprocess
            
            if not os.path.exists(output_dir):
                msg = f'Folder "{output_dir}" does not exist. after you save an image, the folder will be created.'
                print(msg)
                gr.Info(msg)
                return
            elif not os.path.isdir(output_dir):
                msg = f'WARNING: Path "{output_dir}" is not a folder.'
                print(msg)
                gr.Warning(msg)
                return

            output_dir = os.path.normpath(output_dir)
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", output_dir])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                subprocess.Popen(["explorer.exe", subprocess.check_output(["wslpath", "-w", output_dir]).decode('utf-8').strip()])
            else:
                subprocess.Popen(["xdg-open", output_dir])
                
            return f"已打开目录: {output_dir}"
        except Exception as e:
            error_msg = f"打开目录失败: {str(e)}"
            print(error_msg)
            return error_msg
    
    # 绑定事件
    point_run_button.click(
        point_segmentation,
        inputs=[sam_input, points_state, model_type],
        outputs=[sam_output]
    )
    
    point_clear_button.click(
        clear_all_points,
        inputs=[sam_input],
        outputs=[sam_input, points_state]
    )
    
    # 绑定打开输出目录按钮事件
    open_output_folder_button.click(
        open_output_folder,
        inputs=[],
        outputs=[]
    )
    
    # 保存上传的原始图像
    sam_input.upload(
        save_original_image,
        inputs=[sam_input],
        outputs=[sam_input]
    )
    
    # 点击添加正点
    sam_input.select(
        add_point,
        inputs=[sam_input, points_state],
        outputs=[sam_input, points_state]
    )
    
    model_type.change(
        change_model,
        inputs=[model_type],
        outputs=[]
    )

    return {
        "sam_input": sam_input,
        "sam_output": sam_output,
        "model_type": model_type,
        "point_run_button": point_run_button,
        "point_clear_button": point_clear_button,
        "open_output_folder_button": open_output_folder_button,
        "points_state": points_state
    }


def create_sam_ui():
    """
    创建图像分割UI模块的入口函数
    """
    return create_sam_segmentation()