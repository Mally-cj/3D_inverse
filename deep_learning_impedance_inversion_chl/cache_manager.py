#!/usr/bin/env python3
"""
数据缓存管理脚本
"""

import os
import pickle
import sys

CACHE_FILE = 'processed_data_cache.pkl'

def show_cache_info():
    """显示缓存文件信息"""
    if os.path.exists(CACHE_FILE):
        size = os.path.getsize(CACHE_FILE) / (1024*1024)  # MB
        print(f"缓存文件存在: {CACHE_FILE}")
        print(f"文件大小: {size:.2f} MB")
        
        # 尝试加载并显示内容信息
        try:
            with open(CACHE_FILE, 'rb') as f:
                data_dict = pickle.load(f)
            print("缓存内容:")
            for key, value in data_dict.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        except Exception as e:
            print(f"读取缓存文件时出错: {e}")
    else:
        print("缓存文件不存在")

def clear_cache():
    """清理缓存文件"""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("缓存文件已删除")
    else:
        print("缓存文件不存在，无需删除")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'clear':
            clear_cache()
        elif sys.argv[1] == 'info':
            show_cache_info()
        else:
            print("用法:")
            print("  python cache_manager.py info  - 显示缓存信息")
            print("  python cache_manager.py clear - 清理缓存")
    else:
        show_cache_info()

if __name__ == "__main__":
    main()
