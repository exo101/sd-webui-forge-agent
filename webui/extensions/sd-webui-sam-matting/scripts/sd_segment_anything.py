import gradio as gr
from modules import script_callbacks
from pathlib import Path
import sys
import subprocess

# 添加当前插件目录到 Python 路径
plugin_dir = Path(__file__).parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

# 添加 scripts 目录到 Python 路径
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# 添加 nobg 目录到 Python 路径（用于 nobg 抠图模型）
nobg_dir = plugin_dir / "nobg"
if str(nobg_dir) not in sys.path:
    sys.path.insert(0, str(nobg_dir))

# 自动安装依赖
def auto_install_dependencies():
    """自动安装 SAM Matting 所需的依赖包"""
    # 定义需要检查的包 - 格式：{导入名：pip 包名}
    required_packages = {
        "rembg": "rembg",                      # 背景移除工具
        "onnxruntime": "onnxruntime-gpu",      # GPU 加速 ONNX 运行时（注意：导入名是 onnxruntime）
        "litelama": "litelama",                # 轻量级图像修复模型
        "segment_anything": "segment-anything", # Meta 分割一切模型（注意：导入名是 segment_anything）
    }
    
    missing_packages = []
    
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n⚠️  SAM Matting: 发现 {len(missing_packages)} 个缺失的依赖包，正在自动安装...")
        
        # 执行安装脚本
        install_script = plugin_dir / "install_dependencies.py"
        if install_script.exists():
            import subprocess
            python_exe = sys.executable
            
            try:
                # 不捕获输出，让 pip 的输出直接显示到控制台
                result = subprocess.run(
                    [python_exe, str(install_script)],
                    check=True,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                # 检查是否成功
                if result.returncode == 0:
                    pass  # 成功时不显示额外信息
                else:
                    print(f"\n❌ SAM Matting 依赖安装失败，退出码：{result.returncode}")
                    print("\n您可以手动安装:")
                    for pkg in missing_packages:
                        print(f"  python -m pip install {pkg}")
                    
            except subprocess.CalledProcessError as e:
                print(f"\n❌ SAM Matting 依赖安装失败：{e}")
                print("\n您可以手动安装:")
                for pkg in missing_packages:
                    print(f"  python -m pip install {pkg}")
            except Exception as e:
                print(f"\n❌ SAM Matting 依赖安装出错：{e}")
                print("\n您可以手动安装:")
                for pkg in missing_packages:
                    print(f"  python -m pip install {pkg}")
        else:
            print(f"\n❌ SAM Matting: 未找到安装脚本")
            print("请手动安装缺失的依赖包:")
            for pkg in missing_packages:
                print(f"  python -m pip install {pkg}")
    # 如果所有依赖都已安装，不显示任何信息

# 在模块加载时自动执行安装
auto_install_dependencies()

# 导入各个功能模块
try:
    from segment_anything_ui import create_sam_ui, SAM_AVAILABLE
except ImportError as e:
    create_sam_ui = None
    SAM_AVAILABLE = False
    print(f"Warning: Could not import segment_anything_ui: {e}")

try:
    from image_matting import create_image_matting_module
except ImportError as e:
    create_image_matting_module = None
    print(f"Warning: Could not import image_matting: {e}")

try:
    from cleaner_ui import create_cleaner_module, CLEANER_AVAILABLE
except ImportError as e:
    create_cleaner_module = None
    CLEANER_AVAILABLE = False
    print(f"Warning: Could not import cleaner_ui: {e}")


def segmentation_tab():
    """创建图像分割/抠图/清理标签页"""
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            # 智能抠图标签页
            with gr.TabItem("智能抠图"):
                if create_image_matting_module is not None:
                    try:
                        create_image_matting_module()
                    except Exception as e:
                        gr.Markdown(f"智能抠图模块加载失败：{e}")
                else:
                    gr.Markdown("智能抠图模块当前不可用。")
            
            # 点选分割标签页
            with gr.TabItem("点选分割"):
                if SAM_AVAILABLE and create_sam_ui is not None:
                    try:
                        sam_ui_components = create_sam_ui()
                    except Exception as e:
                        with gr.Group():
                            gr.Markdown("## 图像分割")
                            gr.Markdown(f"图像分割模块加载时出现错误：{str(e)}")
                            gr.Markdown("请检查控制台输出以获取详细错误信息。")
                        import traceback
                        traceback.print_exc()
                else:
                    gr.Markdown("图像分割模块不可用。请确保已安装 segment-anything 库。")
            
            # 图像清理标签页
            with gr.TabItem("图像清理"):
                if CLEANER_AVAILABLE and create_cleaner_module is not None:
                    try:
                        cleaner_ui_components = create_cleaner_module()
                    except Exception as e:
                        with gr.Group():
                            gr.Markdown("## 图像清理")
                            gr.Markdown(f"图像清理模块加载时出现错误：{str(e)}")
                            gr.Markdown("请检查控制台输出以获取详细错误信息。")
                        import traceback
                        traceback.print_exc()
                else:
                    gr.Markdown("图像清理模块不可用。请确保已安装 litelama 库。")
    
    return [(ui, "智能抠图与图像清理", "Segmentation_Tab")]

# 注册标签页
script_callbacks.on_ui_tabs(segmentation_tab)
