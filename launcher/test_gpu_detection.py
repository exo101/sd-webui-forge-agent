"""
多GPU检测测试脚本

使用方法:
    cd d:\ai\sd-webui-forge-neo-v3\launcher
    python test_gpu_detection.py

输出示例:
    检测到 2 个GPU设备:
    
    GPU 0: NVIDIA GeForce RTX 4090
      - 显存: 24.0 GB (24576 MB)
      - 计算能力: 8.9
    
    GPU 1: NVIDIA GeForce RTX 3090
      - 显存: 24.0 GB (24576 MB)
      - 计算能力: 8.6
"""

import sys
import os

# 添加core目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from env_checker import check_all_gpus, check_cuda


def main():
    print("=" * 60)
    print("🔍 GPU设备检测工具")
    print("=" * 60)
    print()
    
    # 基础CUDA检测
    cuda_info = check_cuda()
    print(f"PyTorch版本: {cuda_info.get('torch', '未知')}")
    print(f"CUDA可用: {'✅ 是' if cuda_info.get('cuda') else '❌ 否'}")
    print()
    
    if not cuda_info.get('cuda'):
        print("❌ CUDA不可用，无法检测GPU设备")
        return
    
    # 检测所有GPU
    all_gpus = check_all_gpus()
    
    if not all_gpus:
        print("⚠️  未检测到任何GPU设备")
        return
    
    print(f"✅ 检测到 {len(all_gpus)} 个GPU设备:")
    print()
    
    for gpu in all_gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    - 显存: {gpu['vram_gb']} GB ({gpu['vram_mb']} MB)")
        print(f"    - 计算能力: {gpu['compute_capability']}")
        print()
    
    # 显示配置建议
    if len(all_gpus) > 1:
        print("💡 提示: 检测到多个GPU，可以在启动器的「环境检测」页面选择使用哪个GPU")
        print()
        print("可选配置:")
        print(f'  - "gpu_device": ""       # 使用所有GPU（默认）')
        for gpu in all_gpus:
            print(f'  - "gpu_device": "{gpu["index"]}"      # 只使用GPU {gpu["index"]}')
    else:
        print("ℹ️  系统只有一个GPU，无需切换")


if __name__ == "__main__":
    main()
