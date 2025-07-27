#!/usr/bin/env python3
"""
测试导入脚本
验证所有必要的模块和函数是否可以正常导入
"""

import sys
import os

def test_imports():
    """测试所有必要的导入"""
    print("🔍 开始测试导入...")
    
    try:
        # 测试基础模块
        print("✅ 测试基础模块...")
        import numpy as np
        import torch
        import scipy
        from scipy.signal import filtfilt
        from scipy import signal
        from obspy.io.segy.segy import _read_segy
        from tqdm import tqdm
        print("   ✅ 基础模块导入成功")
        
        # 测试Model模块
        print("✅ 测试Model模块...")
        from Model.net2D import UNet, forward_model
        from Model.utils import image2cols
        from Model.joint_well import add_labels
        print("   ✅ Model模块导入成功")
        
        # 测试数据处理模块
        print("✅ 测试数据处理模块...")
        from data_processor import SeismicDataProcessor
        print("   ✅ 数据处理模块导入成功")
        
        # 测试cpp_to_py模块
        print("✅ 测试cpp_to_py模块...")
        sys.path.append('deep_learning_impedance_inversion_chl')
        from cpp_to_py import generate_well_mask as generate_well_mask2
        from cpp_to_py import get_wellline_and_mask as get_wellline_and_mask2
        print("   ✅ cpp_to_py模块导入成功")
        
        print("\n🎉 所有导入测试通过！")
        return True, SeismicDataProcessor
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False, None
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False, None

def test_data_files():
    """测试数据文件是否存在"""
    print("\n🔍 测试数据文件...")
    
    data_files = [
        "data/yyf_smo_train_Volume_PP_IMP.sgy",
        "data/PSTM_resample1_lf_extension2.sgy"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} 存在")
        else:
            print(f"   ❌ {file_path} 不存在")
            return False
    
    return True

def test_data_processor(SeismicDataProcessor):
    """测试数据处理器"""
    print("\n🔍 测试数据处理器...")
    
    try:
        # 创建数据处理器
        processor = SeismicDataProcessor(cache_dir='test_cache', device='cpu')
        print("   ✅ 数据处理器创建成功")
        
        # 测试缓存目录创建
        if os.path.exists('test_cache'):
            print("   ✅ 缓存目录创建成功")
        else:
            print("   ❌ 缓存目录创建失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ 数据处理器测试失败: {e}")
        return False

def cleanup():
    """清理测试文件"""
    import shutil
    if os.path.exists('test_cache'):
        shutil.rmtree('test_cache')
        print("🧹 清理测试缓存目录")

if __name__ == "__main__":
    print("="*60)
    print("🧪 开始导入测试")
    print("="*60)
    
    # 测试导入
    import_success, SeismicDataProcessor = test_imports()
    
    # 测试数据文件
    data_success = test_data_files()
    
    # 测试数据处理器
    processor_success = test_data_processor(SeismicDataProcessor) if SeismicDataProcessor else False
    
    # 清理
    cleanup()
    
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    print(f"导入测试: {'✅ 通过' if import_success else '❌ 失败'}")
    print(f"数据文件: {'✅ 通过' if data_success else '❌ 失败'}")
    print(f"数据处理器: {'✅ 通过' if processor_success else '❌ 失败'}")
    
    if all([import_success, data_success, processor_success]):
        print("\n🎉 所有测试通过！可以运行主程序了。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。") 