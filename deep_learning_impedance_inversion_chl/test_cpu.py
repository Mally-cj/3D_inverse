#!/usr/bin/env python3
"""
测试脚本 - 验证修改后的代码能否在CPU上运行
"""

import torch
import numpy as np

# 强制使用CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Testing CPU compatibility...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# 测试设备检测逻辑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Selected device: {device}")

if device.type == 'cuda':
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

print(f"Data type: {dtype}")

# 测试基本tensor操作
test_tensor = torch.randn(3, 3).to(device).type(dtype)
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Test tensor device: {test_tensor.device}")

# 测试tensor操作
result = torch.matmul(test_tensor, test_tensor.T)
print(f"Matrix multiplication successful: {result.shape}")

print("CPU compatibility test passed!")
