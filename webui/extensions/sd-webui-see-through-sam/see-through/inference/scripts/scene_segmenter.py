"""Scene segmentation using SAM (Segment Anything Model)."""
import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional

def install_dependencies():
    """安装SAM所需的依赖"""
    try:
        import importlib
        
        dependencies = {
            'segment_anything': 'segment-anything'
        }
        
        missing_deps = []
        for module_name, package_name in dependencies.items():
            try:
                importlib.import_module(module_name)
                print(f"[SceneSegmenter] {package_name} 已安装")
            except ImportError:
                print(f"[SceneSegmenter] {package_name} 未安装，正在安装...")
                missing_deps.append(package_name)
        
        if missing_deps:
            print("[SceneSegmenter] 正在安装缺失的依赖...")
            for package in missing_deps:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"[SceneSegmenter] {package} 安装成功")
                except subprocess.CalledProcessError as e:
                    print(f"[SceneSegmenter] {package} 安装失败: {e}")
    except Exception as e:
        print(f"[SceneSegmenter] 依赖安装检查失败: {e}")

install_dependencies()

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ImportError as e:
    print(f"[SceneSegmenter] 导入SAM模块失败: {e}")
    print("[SceneSegmenter] 请手动安装: pip install segment-anything")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "segment-anything"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        print("[SceneSegmenter] SAM模块安装成功")
    except Exception as e:
        print(f"[SceneSegmenter] 安装SAM模块失败: {e}")
        raise


class SceneSegmenter:
    """Scene segmenter using SAM for automatic object detection and segmentation."""
    
    def __init__(self, device=None, model_type="vit_h", models_dir=None):
        """Initialize scene segmenter.
        
        Args:
            device: Device to run model on (cuda/cpu)
            model_type: SAM model type (vit_h, vit_l, vit_b)
            models_dir: Custom models directory (from caller, e.g. shared.data_path/models)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.models_dir = models_dir  # 由调用方传入的模型目录（兼容本地/云端路径差异）
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM model."""
        try:
            print(f"[SceneSegmenter] Loading SAM model (type: {self.model_type})...")
            
            checkpoint = self._find_checkpoint()
            if checkpoint is None:
                print("[SceneSegmenter] SAM checkpoint not found, trying to download...")
                # 自动下载模型
                checkpoint = self._download_checkpoint()
            
            if checkpoint is None:
                print("[SceneSegmenter] SAM checkpoint not found and download failed")
                # 抛出明确的错误，提示用户需要手动下载模型
                checkpoint_names = {
                    "vit_h": "sam_vit_h_4b8939.pth",
                    "vit_l": "sam_vit_l_0b3195.pth",
                    "vit_b": "sam_vit_b_01ec64.pth"
                }
                checkpoint_name = checkpoint_names.get(self.model_type, "sam_vit_b_01ec64.pth")
                # 显示正确的模型目录路径
                if self.models_dir and os.path.isdir(os.path.join(self.models_dir, "sams")):
                    display_sams_dir = os.path.join(self.models_dir, "sams")
                elif self.models_dir:
                    display_sams_dir = os.path.join(self.models_dir, "sams")
                else:
                    current_file = os.path.abspath(__file__)
                    webui_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", "..", ".."))
                    display_sams_dir = os.path.join(webui_root, "models", "sams")
                raise FileNotFoundError(f"SAM 模型未找到！请从群主网盘中下载模型并放置到以下目录：\n" 
                                      f"{display_sams_dir}\n" 
                                      f"需要的模型文件：{checkpoint_name}\n"
                                      f"[调试] models_dir={self.models_dir}")
            
            self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            self.sam.to(device=self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            self.predictor = SamPredictor(self.sam)
            
            print(f"[SceneSegmenter] SAM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[SceneSegmenter] Failed to load SAM model: {e}")
            self.sam = None
            self.mask_generator = None
            self.predictor = None
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find SAM checkpoint in common locations, trying all model types."""
        checkpoint_names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        # 优先使用调用方传入的 models_dir（兼容本地/云端路径差异）
        if self.models_dir and os.path.isdir(os.path.join(self.models_dir, "sams")):
            sams_dir = os.path.join(self.models_dir, "sams")
        else:
            # fallback: 从脚本路径计算
            current_file = os.path.abspath(__file__)
            webui_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", "..", ".."))
            models_dir = os.path.join(webui_root, "models")
            sams_dir = os.path.join(models_dir, "sams")
        
        print(f"[SceneSegmenter] SAMS directory: {sams_dir}")
        print(f"[SceneSegmenter] models_dir from caller: {self.models_dir}")
        print(f"[SceneSegmenter] sams_dir exists: {os.path.isdir(sams_dir)}")
        if os.path.isdir(sams_dir):
            print(f"[SceneSegmenter] Files in sams_dir: {os.listdir(sams_dir)}")
        
        # 先尝试指定的模型类型，如果找不到则尝试所有类型
        model_types_to_try = [self.model_type]
        for mt in ["vit_h", "vit_l", "vit_b"]:
            if mt not in model_types_to_try:
                model_types_to_try.append(mt)
        
        for mt in model_types_to_try:
            checkpoint_name = checkpoint_names.get(mt)
            if not checkpoint_name:
                continue
            
            possible_paths = [
                os.path.join(sams_dir, checkpoint_name),
                os.path.join(os.path.expanduser("~"), ".cache", "sam", checkpoint_name),
            ]
            
            for path in possible_paths:
                lexists = os.path.lexists(path)
                islink = os.path.islink(path)
                print(f"[SceneSegmenter] Checking: {path} -> lexists={lexists}, islink={islink}")
                if lexists:
                    # 即使是符号链接也尝试加载（云端环境使用符号链接指向对象存储）
                    try:
                        # 验证文件是否可读（打开并关闭）
                        with open(path, 'rb') as f:
                            pass
                        print(f"[SceneSegmenter] Found checkpoint at: {path} (model: {mt})")
                        self.model_type = mt
                        return path
                    except (IOError, OSError) as e:
                        print(f"[SceneSegmenter] File exists but cannot read: {e}")
                        # 继续尝试其他路径
                # 尝试用 glob 直接搜索（绕过可能的符号链接问题）
                if os.path.isdir(sams_dir):
                    import glob
                    matches = glob.glob(os.path.join(sams_dir, "*.pth"))
                    if matches:
                        print(f"[SceneSegmenter] glob found: {matches}")
                        for m in matches:
                            for mt2, cn in checkpoint_names.items():
                                if m.endswith(cn):
                                    print(f"[SceneSegmenter] Found via glob: {m} (model: {mt2})")
                                    self.model_type = mt2
                                    return m
        
        return None
    
    def _download_checkpoint(self) -> Optional[str]:
        """Download SAM checkpoint from multiple mirrors."""
        try:
            import urllib.request
            import shutil
            
            # 模型文件名映射
            model_filenames = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            
            filename = model_filenames.get(self.model_type)
            if not filename:
                return None
            
            # 确定保存目录
            if self.models_dir and os.path.isdir(os.path.join(self.models_dir, "sams")):
                sams_dir = os.path.join(self.models_dir, "sams")
            elif self.models_dir:
                sams_dir = os.path.join(self.models_dir, "sams")
            else:
                current_file = os.path.abspath(__file__)
                webui_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", "..", ".."))
                sams_dir = os.path.join(webui_root, "models", "sams")
            
            os.makedirs(sams_dir, exist_ok=True)
            checkpoint_path = os.path.join(sams_dir, filename)
            
            # 如果文件已存在（如之前下载中断留下的），先删除
            if os.path.lexists(checkpoint_path):
                print(f"[SceneSegmenter] Removing existing file: {checkpoint_path}")
                os.remove(checkpoint_path)
            
            # 多镜像下载列表
            # 可通过环境变量 SAM_DOWNLOAD_MIRROR 设置自定义镜像前缀
            env_mirror = os.environ.get("SAM_DOWNLOAD_MIRROR", "")
            mirror_urls = []
            
            if env_mirror:
                mirror_urls.append(env_mirror + filename)
            
            # 默认镜像列表（按优先级排序）
            mirror_urls += [
                f"https://dl.fbaipublicfiles.com/segment_anything/{filename}",
                # 国内可用镜像
                f"https://hf-mirror.com/facebook/sam-{self.model_type.replace('_', '-')}/resolve/main/{filename}",
            ]
            
            # 尝试从每个镜像下载
            last_error = None
            for url in mirror_urls:
                try:
                    print(f"[SceneSegmenter] Trying to download from: {url}")
                    print(f"[SceneSegmenter] Saving to: {checkpoint_path}")
                    
                    # 使用 urlretrieve 并显示进度
                    def report_progress(block_count, block_size, total_size):
                        downloaded = block_count * block_size / (1024 * 1024)
                        if total_size > 0:
                            total_mb = total_size / (1024 * 1024)
                            percent = min(100, downloaded / total_mb * 100)
                            print(f"\r[SceneSegmenter] Downloading: {percent:.1f}% ({downloaded:.1f}/{total_mb:.1f} MB)", end="")
                        else:
                            print(f"\r[SceneSegmenter] Downloading: {downloaded:.1f} MB...", end="")
                    
                    urllib.request.urlretrieve(url, checkpoint_path, reporthook=report_progress)
                    print()  # 换行
                    
                    # 验证文件大小
                    file_size = os.path.getsize(checkpoint_path)
                    if file_size > 1024 * 1024:  # 至少 1MB 才认为有效
                        print(f"[SceneSegmenter] Downloaded successfully: {checkpoint_path} ({file_size / (1024*1024):.1f} MB)")
                        return checkpoint_path
                    else:
                        print(f"[SceneSegmenter] File too small ({file_size} bytes), might be invalid")
                        os.remove(checkpoint_path)
                        continue
                        
                except Exception as e:
                    print(f"\n[SceneSegmenter] Download failed: {e}")
                    last_error = e
                    # 清理可能的部分下载文件
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    continue
            
            print(f"[SceneSegmenter] All mirrors failed. Last error: {last_error}")
            print(f"[SceneSegmenter] You can manually download the model from:")
            print(f"[SceneSegmenter]   {mirror_urls[0] if not env_mirror else env_mirror + filename}")
            print(f"[SceneSegmenter] And place it in: {sams_dir}")
            return None
            
        except Exception as e:
            print(f"[SceneSegmenter] Failed to download checkpoint: {e}")
            return None
    
    def segment_image(self, image_path: str, min_area: int = 1000, 
                     max_masks: int = 10) -> List[Dict]:
        """Segment image and return masks.
        
        Args:
            image_path: Path to input image
            min_area: Minimum mask area in pixels
            max_masks: Maximum number of masks to return
            
        Returns:
            List of dictionaries containing mask info
        """
        if self.mask_generator is None:
            print("[SceneSegmenter] Model not loaded, cannot segment")
            return []
        
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"[SceneSegmenter] Processing image: {image.shape}")
        
        print("[SceneSegmenter] Generating masks...")
        masks = self.mask_generator.generate(image)
        print(f"[SceneSegmenter] Generated {len(masks)} masks")
        
        filtered_masks = []
        for mask_data in masks:
            area = mask_data['area']
            if area >= min_area:
                filtered_masks.append(mask_data)
        
        # 去重处理：移除重叠度高的掩码
        unique_masks = []
        used_area = np.zeros(image.shape[:2], dtype=bool)
        
        for mask_data in sorted(filtered_masks, key=lambda x: x['area'], reverse=True):
            mask = mask_data['segmentation']
            
            # 计算与已选掩码的重叠度
            overlap = np.sum(np.logical_and(mask, used_area)) / np.sum(mask)
            
            # 如果重叠度低于阈值，则保留该掩码
            if overlap < 0.3:  # 30% 重叠度阈值
                unique_masks.append(mask_data)
                used_area = np.logical_or(used_area, mask)
            
            # 如果达到最大数量，停止
            if len(unique_masks) >= max_masks:
                break
        
        filtered_masks = unique_masks
        
        # 删除最小的掩码
        if len(filtered_masks) > 1:
            print(f"[SceneSegmenter] 原始掩码数量: {len(filtered_masks)}")
            # 按面积排序，找到最小的掩码
            filtered_masks.sort(key=lambda x: x['area'])
            smallest_mask = filtered_masks[0]
            print(f"[SceneSegmenter] 删除最小的掩码，面积: {smallest_mask['area']}")
            # 删除最小的掩码
            filtered_masks = filtered_masks[1:]
            # 重新按面积降序排序
            filtered_masks.sort(key=lambda x: x['area'], reverse=True)
            print(f"[SceneSegmenter] 删除后掩码数量: {len(filtered_masks)}")
        
        print(f"[SceneSegmenter] Filtered to {len(filtered_masks)} masks")
        
        return filtered_masks
    
    def create_layer_images(self, image_path: str, masks: List[Dict], 
                           output_dir: str) -> List[str]:
        """Create layer images from masks.
        
        Args:
            image_path: Path to input image
            masks: List of mask dictionaries
            output_dir: Directory to save layer images
            
        Returns:
            List of output file paths
        """
        image = np.array(Image.open(image_path).convert('RGBA'))
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            
            layer = np.zeros_like(image)
            layer[mask] = image[mask]
            
            output_name = f"layer_{i:02d}.png"
            output_path = os.path.join(output_dir, output_name)
            Image.fromarray(layer).save(output_path)
            output_paths.append(output_path)
            
            print(f"[SceneSegmenter] Saved: {output_name} (area: {mask_data['area']})")
        
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask_data in masks:
            combined_mask |= mask_data['segmentation']
        
        background = image.copy()
        background[combined_mask] = 0
        
        background_path = os.path.join(output_dir, "background.png")
        Image.fromarray(background).save(background_path)
        output_paths.append(background_path)
        
        print(f"[SceneSegmenter] Saved: background.png")
        
        return output_paths
    
    def create_psd(self, image_path: str, masks: List[Dict], output_dir: str, keywords: List[str]) -> str:
        """Create PSD file from masks.
        
        Args:
            image_path: Path to input image
            masks: List of mask dictionaries
            output_dir: Directory to save PSD file
            keywords: List of keywords for layer names
            
        Returns:
            Path to created PSD file
        """
        try:
            from psd_tools import PSDImage
            print("[SceneSegmenter] Successfully imported psd-tools")
            
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            layer_paths = []
            for i, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                
                layer_image = Image.new("RGB", (width, height), (255, 255, 255))
                layer_np = np.array(layer_image)
                image_np = np.array(image)
                layer_np[mask] = image_np[mask]
                layer_image = Image.fromarray(layer_np)
                
                layer_name = keywords[i] if i < len(keywords) else f"layer_{i+1}"
                layer_path = os.path.join(output_dir, f"{layer_name}.png")
                layer_image.save(layer_path)
                layer_paths.append((layer_name, layer_path))
            
            try:
                psd = PSDImage.new('RGB', (width, height))
                print("[SceneSegmenter] Successfully created PSD with PSDImage.new")
                
                try:
                    image_np = np.array(image)
                    combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
                    for mask_data in masks:
                        combined_mask |= mask_data['segmentation']
                    
                    background_np = image_np.copy()
                    background_np[combined_mask] = 0
                    background = Image.fromarray(background_np)
                    
                    if hasattr(psd, 'create_pixel_layer'):
                        background_layer = psd.create_pixel_layer(background, name='background', top=0, left=0, opacity=255)
                        print("[SceneSegmenter] Successfully added background layer")
                    else:
                        print("[SceneSegmenter] create_pixel_layer method not available")
                except Exception as e:
                    print(f"[SceneSegmenter] Error adding background layer: {e}")
                
                for i, mask_data in enumerate(masks):
                    mask = mask_data['segmentation']
                    
                    layer_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    layer_np = np.array(layer_image)
                    image_np = np.array(image.convert("RGBA"))
                    
                    layer_np[mask] = image_np[mask]
                    layer_image = Image.fromarray(layer_np)
                    
                    try:
                        layer_name = keywords[i] if i < len(keywords) else f"layer_{i+1}"
                        if hasattr(psd, 'create_pixel_layer'):
                            layer = psd.create_pixel_layer(layer_image, name=layer_name, top=0, left=0, opacity=255)
                            print(f"[SceneSegmenter] Successfully added layer: {layer_name}")
                        else:
                            print(f"[SceneSegmenter] Cannot add layer {layer_name}: create_pixel_layer not available")
                    except Exception as e:
                        print(f"[SceneSegmenter] Error adding layer {i}: {e}")
                
                psd_path = os.path.join(output_dir, "scene_seg.psd")
                psd.save(psd_path)
                print(f"[SceneSegmenter] Created PSD: {psd_path}")
                return psd_path
            except Exception as e:
                print(f"[SceneSegmenter] Error creating PSD: {e}")
                return ""
        except ImportError as e:
            print(f"[SceneSegmenter] psd-tools not installed: {e}")
            return ""
        except Exception as e:
            print(f"[SceneSegmenter] Error creating PSD: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
    
    def segment_with_keywords(self, image_path: str, keywords: List[str], 
                             output_dir: str) -> List[str]:
        """Segment image using keywords with CLIP-based filtering.
        
        Args:
            image_path: Path to input image
            keywords: List of keywords to segment
            output_dir: Directory to save layer images
            
        Returns:
            List of output file paths
        """
        masks = self.segment_image(image_path)
        masks = masks[:len(keywords)]
        layer_paths = self.create_layer_images(image_path, masks, output_dir)
        psd_path = self.create_psd(image_path, masks, output_dir, keywords)
        
        return layer_paths + [psd_path]


def segment_scene(image_path: str, output_dir: str, 
                  min_area: int = 1000, max_masks: int = 10,
                  model_type: str = "vit_b", models_dir: str = None) -> List[str]:
    """Segment scene and create layer images.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save layer images
        min_area: Minimum mask area in pixels
        max_masks: Maximum number of masks to return
        model_type: SAM model type (vit_h, vit_l, vit_b)
        models_dir: Custom models directory (from caller, for cross-platform compatibility)
        
    Returns:
        List of output file paths
    """
    segmenter = SceneSegmenter(model_type=model_type, models_dir=models_dir)
    masks = segmenter.segment_image(image_path, min_area=min_area, max_masks=max_masks)
    return segmenter.create_layer_images(image_path, masks, output_dir)


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="场景分割 (SAM)")
    parser.add_argument("--srcp", type=str, help="输入图像路径")
    parser.add_argument("--save_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--resolution", type=int, default=1024, help="处理分辨率")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM模型类型 (vit_h, vit_l, vit_b)")
    parser.add_argument("--min_area", type=int, default=1000, help="最小区域大小")
    parser.add_argument("--max_masks", type=int, default=10, help="最大分割数量")
    parser.add_argument("--models_dir", type=str, default=None, help="自定义模型目录（兼容本地/云端路径差异）")
    # 也支持直接传 image_path 参数（兼容旧用法）
    parser.add_argument("image_path", type=str, nargs="?", default=None, help="输入图像路径（旧用法）")
    
    args = parser.parse_args()
    
    # 确定输入图像路径
    image_path = args.srcp or args.image_path
    if not image_path:
        print("错误: 请指定输入图像路径 (--srcp <path>)")
        sys.exit(1)
    
    # 确定输出目录
    if args.save_dir:
        output_dir = args.save_dir
    else:
        output_dir = os.path.join(os.path.dirname(image_path), "segmented")
    
    print(f"场景分割 - 输入: {image_path}")
    print(f"场景分割 - 输出: {output_dir}")
    print(f"场景分割 - 分辨率: {args.resolution}")
    print(f"场景分割 - 模型: {args.model_type}")
    
    output_paths = segment_scene(image_path, output_dir, 
                                  min_area=args.min_area, 
                                  max_masks=args.max_masks,
                                  model_type=args.model_type,
                                  models_dir=args.models_dir)
    
    # 尝试创建PSD文件
    try:
        from psd_tools import PSDImage
        segmenter = SceneSegmenter(model_type=args.model_type, models_dir=args.models_dir)
        masks = segmenter.segment_image(image_path, min_area=args.min_area, max_masks=args.max_masks)
        segmenter.create_psd(image_path, masks, output_dir, 
                            [f"layer_{i+1}" for i in range(len(masks))])
        print(f"场景分割 - PSD文件已生成")
    except Exception as e:
        print(f"场景分割 - PSD文件生成失败: {e}")
    
    print(f"\n场景分割完成! 生成 {len(output_paths)} 个图层:")
    for path in output_paths:
        print(f"  - {path}")
