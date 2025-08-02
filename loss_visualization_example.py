"""
Loss可视化功能使用示例
展示如何使用Model/utils.py中封装的loss记录和可视化函数
"""

import numpy as np
import matplotlib.pyplot as plt
from Model.utils import (
    save_stage1_loss_data, 
    save_stage2_loss_data, 
    save_complete_training_loss,
    load_loss_data,
    plot_loss_comparison
)

def example_usage():
    """示例：如何使用loss可视化功能"""
    
    # 创建保存目录
    import os
    save_dir = 'example_loss_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 Loss可视化功能使用示例")
    print("="*50)
    
    # 1. 模拟训练数据
    print("1. 生成模拟训练数据...")
    admm_iter = 100   # 阶段1训练轮次
    admm_iter1 = 50   # 阶段2训练轮次
    
    # 模拟阶段1的loss数据（子波矫正损失）
    total_lossF = []
    for i in range(admm_iter):
        # 模拟指数衰减的loss
        loss = 0.1 * np.exp(-i/30) + 0.01 + np.random.normal(0, 0.005)
        total_lossF.append(loss)
    
    # 模拟阶段2的loss数据
    stage2_total_loss = []
    stage2_sup_loss = []
    stage2_unsup_loss = []
    stage2_tv_loss = []
    
    for i in range(admm_iter1):
        # 模拟各项损失
        sup_loss = 0.05 * np.exp(-i/15) + 0.005 + np.random.normal(0, 0.002)
        unsup_loss = 0.08 * np.exp(-i/20) + 0.008 + np.random.normal(0, 0.003)
        tv_loss = 0.03 * np.exp(-i/25) + 0.003 + np.random.normal(0, 0.001)
        total_loss = sup_loss + unsup_loss + tv_loss
        
        stage2_sup_loss.append(sup_loss)
        stage2_unsup_loss.append(unsup_loss)
        stage2_tv_loss.append(tv_loss)
        stage2_total_loss.append(total_loss)
    
    # 2. 保存阶段1的loss数据
    print("2. 保存阶段1loss数据...")
    save_stage1_loss_data(save_dir, total_lossF, admm_iter)
    
    # 3. 保存阶段2的loss数据
    print("3. 保存阶段2loss数据...")
    save_stage2_loss_data(save_dir, stage2_total_loss, stage2_sup_loss, 
                         stage2_unsup_loss, stage2_tv_loss, admm_iter1)
    
    # 4. 保存完整训练过程对比图
    print("4. 保存完整训练过程对比图...")
    save_complete_training_loss(save_dir, total_lossF, stage2_total_loss, 
                               stage2_sup_loss, stage2_unsup_loss, stage2_tv_loss, 
                               admm_iter, admm_iter1)
    
    # 5. 加载已保存的loss数据
    print("5. 加载已保存的loss数据...")
    loss_data = load_loss_data(save_dir, stage='both')
    
    # 6. 生成额外的分析图表
    print("6. 生成额外的分析图表...")
    plot_loss_comparison(loss_data, save_dir, "训练损失详细分析")
    
    print("\n✅ 示例完成！")
    print(f"📁 所有文件已保存到: {save_dir}/")
    print("\n生成的文件包括:")
    print("  - stage1_loss_data.npy: 阶段1loss数据")
    print("  - stage2_loss_data.npy: 阶段2loss数据")
    print("  - stage1_wavelet_loss.png: 阶段1loss曲线")
    print("  - stage2_unet_loss.png: 阶段2详细loss分析")
    print("  - complete_training_loss.png: 完整训练过程对比")
    print("  - loss_comparison_analysis.png: 额外分析图表")

def analyze_existing_data():
    """分析已存在的训练数据"""
    print("\n🔍 分析已存在的训练数据")
    print("="*50)
    
    # 指定包含loss数据的目录
    save_dir = 'logs/model/20241201-10:30:00'  # 替换为实际的训练目录
    
    if os.path.exists(save_dir):
        # 加载loss数据
        loss_data = load_loss_data(save_dir, stage='both')
        
        if loss_data:
            # 生成分析图表
            plot_loss_comparison(loss_data, save_dir, "实际训练损失分析")
            print("✅ 分析完成！")
        else:
            print("⚠️  未找到loss数据文件")
    else:
        print(f"⚠️  目录不存在: {save_dir}")

if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    # 可选：分析已存在的训练数据
    # analyze_existing_data() 