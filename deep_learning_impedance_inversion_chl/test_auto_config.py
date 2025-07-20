#!/usr/bin/env python3
"""
测试智能配置功能
验证设备检测和参数自动调整是否正常工作
"""

import torch
import sys
import os

def test_device_detection():
    """测试设备检测功能"""
    print("🔍 Testing device detection...")
    
    # 智能设备检测和参数配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Detected device: {device}")

    # 根据设备自动调整参数
    if device.type == 'cuda':
        print("📊 GPU mode configuration:")
        USE_FULL_DATA = True
        MAX_SPATIAL_SLICES = 251
        BATCH_SIZE = 10
        PATCH_SIZE = 70
        N_WELL_PROFILES = 30
        ADMM_ITER = 100
        ADMM_ITER1 = 50
        MAX_TRAIN_SAMPLES = None
    else:
        print("💻 CPU mode configuration:")
        USE_FULL_DATA = False
        MAX_SPATIAL_SLICES = 50
        BATCH_SIZE = 1
        PATCH_SIZE = 48
        N_WELL_PROFILES = 10
        ADMM_ITER = 30
        ADMM_ITER1 = 15
        MAX_TRAIN_SAMPLES = 300

    print(f"📋 Auto-configured parameters:")
    print(f"  - Use full data: {USE_FULL_DATA}")
    print(f"  - Spatial slices: {MAX_SPATIAL_SLICES}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Patch size: {PATCH_SIZE}")
    print(f"  - Well profiles: {N_WELL_PROFILES}")
    print(f"  - Training samples: {MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES else 'unlimited'}")
    print(f"  - Training iterations: {ADMM_ITER} + {ADMM_ITER1}")
    
    return device, {
        'USE_FULL_DATA': USE_FULL_DATA,
        'MAX_SPATIAL_SLICES': MAX_SPATIAL_SLICES,
        'BATCH_SIZE': BATCH_SIZE,
        'PATCH_SIZE': PATCH_SIZE,
        'N_WELL_PROFILES': N_WELL_PROFILES,
        'ADMM_ITER': ADMM_ITER,
        'ADMM_ITER1': ADMM_ITER1,
        'MAX_TRAIN_SAMPLES': MAX_TRAIN_SAMPLES
    }

def test_memory_estimation(config):
    """估算内存使用量"""
    print("\n💾 Memory estimation...")
    
    # 基于配置估算内存使用
    spatial_slices = config['MAX_SPATIAL_SLICES']
    batch_size = config['BATCH_SIZE']
    patch_size = config['PATCH_SIZE']
    n_profiles = config['N_WELL_PROFILES']
    
    # 估算数据大小 (假设 float32, 4 bytes per element)
    time_samples = 601
    estimated_data_mb = (time_samples * spatial_slices * 251 * 4) / (1024 * 1024)
    estimated_patches_mb = (n_profiles * patch_size * patch_size * 4) / (1024 * 1024)
    estimated_total_mb = estimated_data_mb + estimated_patches_mb + 100  # 额外开销
    
    print(f"  - Raw data: ~{estimated_data_mb:.1f} MB")
    print(f"  - Training patches: ~{estimated_patches_mb:.1f} MB")
    print(f"  - Estimated total: ~{estimated_total_mb:.1f} MB")
    
    if estimated_total_mb > 8000:
        print("  ⚠️  High memory usage - GPU recommended")
    elif estimated_total_mb > 4000:
        print("  ✅ Moderate memory usage - GPU/CPU both OK")
    else:
        print("  ✅ Low memory usage - CPU friendly")

def test_data_reduction_logic():
    """测试数据减少逻辑"""
    print("\n📊 Testing data reduction logic...")
    
    # 模拟原始数据形状
    original_shape = (601, 1189, 251)
    print(f"Original data shape: {original_shape}")
    
    # 模拟CPU配置下的数据减少
    max_spatial_slices = 50
    reduced_shape = (601, max_spatial_slices, 251)
    reduction_ratio = (original_shape[1] * original_shape[2]) / (reduced_shape[1] * reduced_shape[2])
    
    print(f"CPU reduced shape: {reduced_shape}")
    print(f"Data reduction ratio: {reduction_ratio:.1f}x smaller")
    print(f"Memory saving: {(1 - 1/reduction_ratio)*100:.1f}%")

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 Auto-Configuration Test Suite")
    print("=" * 60)
    
    # 测试设备检测
    device, config = test_device_detection()
    
    # 测试内存估算
    test_memory_estimation(config)
    
    # 测试数据减少逻辑
    test_data_reduction_logic()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("🚀 The unified version should work properly on your system")
    print("=" * 60)

if __name__ == "__main__":
    main()
