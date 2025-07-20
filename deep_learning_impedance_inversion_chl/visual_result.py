"""
精简版地震阻抗反演结果可视化工具

功能说明：
1. 从 .npy 文件读取推理结果
2. 专注于使用seisvis进行专业井位可视化分析
3. 计算并显示专业评估指标

使用方法：
1. 确保主程序已运行并生成了 .npy 文件
2. 直接运行此脚本：python visual_result.py
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import sys


def get_result_path(filename):
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"✅ 已创建results目录: {results_dir}")
    return os.path.join(results_dir, os.path.basename(filename))
# 井位配置 - 从主程序获取
def get_well_positions():
    """获取真实井位信息"""
    # 根据数据形状判断是CPU模式还是GPU模式
    if os.path.exists('prediction_impedance.npy'):
        pred_shape = np.load('prediction_impedance.npy').shape
        if pred_shape[1] <= 50:  # CPU模式
            well_positions = [[10, 10], [20, 20], [30, 30], [40, 40]]
            mode = "CPU"
        else:  # GPU模式
            # 原始8口井的位置
            pos = [[594,295], [572,692], [591,996], [532,1053], [603,1212], [561,842], [504,846], [499,597]]
            basex, basey = 450, 212
            well_positions = [[y-basey, x-basex] for [x, y] in pos]
            mode = "GPU"
        ##打印well_positions
        print("well positions:",well_positions)
        
        # 筛选有效井位
        valid_wells = []
        for line, cmp in well_positions:
            ##打印line和cmp
            print(line,cmp)
            
            if 0 <= line < pred_shape[0] and 0 <= cmp < pred_shape[1]:
                valid_wells.append([line, cmp])
            else:
                print(f"⚠️ 无效井位: 行={line}, 列={cmp}")
            
        return valid_wells, mode
    else:
        return [], "Unknown"

# 尝试导入seisvis
# try:
# sys.path.append('/Users/chenjie53/Documents/3D_inverse/3D_inverse/seisvis')
from seisvis import plot1d
from seisvis.plot2d import Seis2DPlotter
from seisvis.plot_config import PlotConfig
from seisvis.data_config import DataCube
SEISVIS_AVAILABLE = True
print("✅ Seisvis库已加载")


def load_results(data_dir='/Users/chenjie53/Documents/3D_inverse/3D_inverse/deep_learning_impedance_inversion_chl'):
    """
    加载推理结果数据，支持指定目录
    Returns:
        dict: 包含所有结果数据的字典
    """
    import os
    import numpy as np
    print("📂 加载推理结果数据...")
    required_files = [
        'prediction_sample.npy',
        'true_sample.npy',
        'input_sample.npy',
        'prediction_impedance.npy',
        'true_impedance.npy'
    ]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    if missing_files:
        print(f"❌ 缺少以下文件: {missing_files}")
        print("💡 请先运行主程序进行推理以生成数据文件")
        return None
    data = {}
    try:
        data['prediction_sample'] = np.load(os.path.join(data_dir, 'prediction_sample.npy'))
        data['true_sample'] = np.load(os.path.join(data_dir, 'true_sample.npy'))
        data['input_sample'] = np.load(os.path.join(data_dir, 'input_sample.npy'))
        data['prediction_impedance'] = np.load(os.path.join(data_dir, 'prediction_impedance.npy'))
        data['true_impedance'] = np.load(os.path.join(data_dir, 'true_impedance.npy'))
        print(f"✅ 数据加载完成:")
        print(f"   - 预测样本形状: {data['prediction_sample'].shape}")
        print(f"   - 真实样本形状: {data['true_sample'].shape}")
        print(f"   - 预测阻抗形状: {data['prediction_impedance'].shape}")
        print(f"   - 真实阻抗形状: {data['true_impedance'].shape}")
        return data
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def calculate_metrics(data):
    """
    计算专业地球物理评估指标，自动适配3D输入
    """
    print("\n📊 计算评估指标...")
    pred_norm = data['prediction_sample']
    true_norm = data['true_sample']
    pred_imp = data['prediction_impedance']
    true_imp = data['true_impedance']
    # flatten for metrics
    metrics = {}
    metrics['correlation'] = np.corrcoef(pred_norm.flatten(), true_norm.flatten())[0, 1]
    metrics['rmse_norm'] = np.sqrt(np.mean((pred_norm - true_norm)**2))
    metrics['rmse_linear'] = np.sqrt(np.mean((pred_imp - true_imp)**2))
    metrics['relative_error'] = np.mean(np.abs(pred_imp - true_imp) / true_imp) * 100
    from skimage.metrics import structural_similarity as ssim
    # 取中间切片做ssim
    if pred_norm.ndim == 3:
        ssim_val = ssim(pred_norm[pred_norm.shape[0]//2], true_norm[true_norm.shape[0]//2], data_range=1.0)
    else:
        ssim_val = ssim(pred_norm, true_norm, data_range=1.0)
    metrics['ssim'] = ssim_val
    metrics['pred_range'] = (pred_imp.min(), pred_imp.max())
    metrics['true_range'] = (true_imp.min(), true_imp.max())
    print(f"✅ 指标计算完成:")
    print(f"   - 相关系数: {metrics['correlation']:.4f}")
    print(f"   - 归一化RMSE: {metrics['rmse_norm']:.4f}")
    print(f"   - 线性阻抗RMSE: {metrics['rmse_linear']:.0f} (kg/m³·m/s)")
    print(f"   - 相对误差: {metrics['relative_error']:.2f}%")
    print(f"   - 结构相似性(SSIM): {metrics['ssim']:.4f}")
    print(f"   - 预测阻抗范围: [{metrics['pred_range'][0]:.0f}, {metrics['pred_range'][1]:.0f}]")
    print(f"   - 真实阻抗范围: [{metrics['true_range'][0]:.0f}, {metrics['true_range'][1]:.0f}]")
    return metrics

# 删除 create_well_focused_visualization 函数，只使用 seisvis

def create_seisvis_well_curves(data, metrics, save_path='results/seisvis_well_curves.png'):
    """
    使用seisvis绘制专业井曲线对比
    
    Args:
        data: 结果数据字典
        metrics: 评估指标字典  
        save_path: 保存路径
    """
    print(f"\n� 使用Seisvis生成专业井曲线...")
    
    try:
        pred_imp = data['prediction_impedance']
        true_imp = data['true_impedance']
        
        # 获取真实井位
        well_positions, mode = get_well_positions()
        
        ##若没有找到
        if len(well_positions) == 0:
            
            print("   ⚠️  未找到有效井位")
            return
        
        # 配置seisvis 1D绘图
        config1d = PlotConfig()
        config1d.label_fontsize = 12
        config1d.tick_labelsize = 10
        config1d.legend_fontsize = 10
        config1d.title_fontsize = 12
        
        plotter1d = plot1d.Seis1DPlotter(config1d)
        
        # 准备数据：每口井一组曲线对比
        data_groups = []
        well_titles = []
        
        # 选择前4口井进行展示
        for i, (line, cmp) in enumerate(well_positions[:4]):
            true_curve = true_imp[:, cmp]
            pred_curve = pred_imp[:, cmp]
            
            # 每组包含真实和预测两条曲线
            well_group = [true_curve, pred_curve]
            data_groups.append(well_group)
            
            # 计算精度
            well_corr = np.corrcoef(pred_curve, true_curve)[0, 1]
            well_titles.append(f'Well-{i+1} @({line},{cmp}) R={well_corr:.3f}')
        
        # 曲线图例和样式
        legends = ['True Impedance', 'Predicted Impedance']
        line_styles = ['-', '--']
        
        # 使用seisvis专业绘图
        result_path = get_result_path(save_path)
        print(f"   📈 绘制专业井曲线到: {result_path}")
        plotter1d.plot_groups(
            data_groups=data_groups,
            t_start=0,
            titles=well_titles,
            legends=legends,
            line_styles=line_styles,
            vis_type='v',  # 垂直显示（地球物理标准）
            figsize=(12, 8),
            save_path=result_path
        )
        
        print(f"   ✅ Seisvis专业井曲线已保存: {result_path}")
        
    except Exception as e:
        
        print(f"   ❌ Seisvis井曲线绘制失败: {e}")
   

        
  

def create_scatter_analysis(data, metrics, save_path='scatter_analysis.png'):
    """
    创建散点图分析，自动适配3D输入
    """
    print(f"\n📈 生成散点图分析...")
    pred_imp = data['prediction_impedance'].flatten()
    true_imp = data['true_impedance'].flatten()
    n_samples = min(10000, len(pred_imp))
    indices = np.random.choice(len(pred_imp), n_samples, replace=False)
    pred_sample = pred_imp[indices]
    true_sample = true_imp[indices]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    ax1.scatter(true_sample, pred_sample, alpha=0.5, s=1)
    min_val = min(true_sample.min(), pred_sample.min())
    max_val = max(true_sample.max(), pred_sample.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Impedance (kg/m³·m/s)')
    ax1.set_ylabel('Predicted Impedance (kg/m³·m/s)')
    ax1.set_title(f'Prediction vs Truth (R={metrics["correlation"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    error = pred_sample - true_sample
    ax2.hist(error, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (kg/m³·m/s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution (Mean={error.mean():.0f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    result_path = get_result_path(save_path)
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 散点图分析已保存: {result_path}")
    plt.close()

def plot_impedance_section(data, save_prefix='results/impedance_section', slice_axis='inline', slice_idx=None):
    import os
    import matplotlib.pyplot as plt
    pred_imp = data['prediction_impedance']
    true_imp = data['true_impedance']
    os.makedirs(os.path.dirname(save_prefix + '_pred.png'), exist_ok=True)
    # 自动检测井点
    well_positions = []
    if 'well_positions' in data:
        well_positions = data['well_positions']
    else:
        try:
            from deep_learning_impedance_inversion_chl.visual_result import get_well_positions
            well_positions, _ = get_well_positions()
        except Exception:
            well_positions = []
    # 3D体自动切片
    def get_slice(arr, axis, idx):
        if arr.ndim == 2:
            return arr
        if axis == 'inline':
            if idx is None:
                idx = arr.shape[0] // 2
            return arr[idx, :, :]
        elif axis == 'crossline':
            if idx is None:
                idx = arr.shape[1] // 2
            return arr[:, idx, :]
        elif axis == 'time':
            if idx is None:
                idx = arr.shape[2] // 2
            return arr[:, :, idx]
        else:
            raise ValueError('axis must be inline, crossline, or time')
    pred_slice = get_slice(pred_imp, slice_axis, slice_idx)
    true_slice = get_slice(true_imp, slice_axis, slice_idx)
    # 画预测阻抗
    plt.figure(figsize=(10, 6))
    plt.imshow(pred_slice, aspect='auto', cmap='jet')
    if well_positions and pred_slice.ndim == 2:
        y, x = zip(*well_positions)
        plt.scatter(x, y, c='k', marker='v', s=50, edgecolors='w', label='Well')
        plt.legend()
    plt.title(f'Predicted Impedance Section ({slice_axis}={slice_idx if slice_idx is not None else "mid"})')
    plt.colorbar(label='Impedance')
    plt.xlabel('Trace')
    plt.ylabel('Time')
    plt.savefig(save_prefix + '_pred.png', dpi=200, bbox_inches='tight')
    plt.close()
    # 画真实阻抗
    plt.figure(figsize=(10, 6))
    plt.imshow(true_slice, aspect='auto', cmap='jet')
    if well_positions and true_slice.ndim == 2:
        y, x = zip(*well_positions)
        plt.scatter(x, y, c='k', marker='v', s=50, edgecolors='w', label='Well')
        plt.legend()
    plt.title(f'True Impedance Section ({slice_axis}={slice_idx if slice_idx is not None else "mid"})')
    plt.colorbar(label='Impedance')
    plt.xlabel('Trace')
    plt.ylabel('Time')
    plt.savefig(save_prefix + '_true.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 2D阻抗剖面图已保存到: {save_prefix}_pred.png 和 _true.png")

def visualize_results(show_plots=False, use_seisvis=True, data_dir='deep_learning_impedance_inversion_chl'):
    """
    主要的可视化函数 - 简化版专注于井位分析和2D阻抗剖面
    """
    print("="*80)
    print("🎨 简化井位专项可视化分析")
    print("="*80)
    data = load_results(data_dir=data_dir)
    if data is None:
        return False
    metrics = calculate_metrics(data)
    print_detailed_metrics(data)
    try:
        if use_seisvis and SEISVIS_AVAILABLE:
            print(f"\n🌟 Seisvis专业井曲线...")
            create_seisvis_well_curves(data, metrics)
        plot_impedance_section(data)
        create_scatter_analysis(data, metrics)
        if show_plots:
            plt.show()
        print(f"\n🎉 可视化完成! 图表保存在 results 目录")
        return True
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_detailed_metrics(data):
    """
    打印详细的评估指标
    
    Args:
        data: 结果数据字典
    """
    print("\n📋 详细评估报告:")
    print("-" * 60)
    
    pred_norm = data['prediction_sample']
    true_norm = data['true_sample']
    pred_imp = data['prediction_impedance']
    true_imp = data['true_impedance']
    
    # 基础指标
    correlation = np.corrcoef(pred_norm.flatten(), true_norm.flatten())[0, 1]
    rmse_norm = np.sqrt(np.mean((pred_norm - true_norm)**2))
    rmse_linear = np.sqrt(np.mean((pred_imp - true_imp)**2))
    mae_linear = np.mean(np.abs(pred_imp - true_imp))
    relative_error = np.mean(np.abs(pred_imp - true_imp) / true_imp) * 100
    
    # SSIM
    ssim_value = ssim(pred_norm, true_norm, data_range=1.0)
    
    # 数据范围
    pred_range = (pred_imp.min(), pred_imp.max())
    true_range = (true_imp.min(), true_imp.max())
    
    print(f"🎯 整体反演精度:")
    print(f"   - 相关系数 (R): {correlation:.4f}")
    print(f"   - 归一化RMSE: {rmse_norm:.4f}")
    print(f"   - 线性域RMSE: {rmse_linear:.0f} (kg/m³·m/s)")
    print(f"   - 线性域MAE: {mae_linear:.0f} (kg/m³·m/s)")
    print(f"   - 相对误差: {relative_error:.2f}%")
    print(f"   - 结构相似性(SSIM): {ssim_value:.4f}")
    
    print(f"\n📊 数据范围对比:")
    print(f"   - 预测阻抗范围: [{pred_range[0]:.0f}, {pred_range[1]:.0f}] (kg/m³·m/s)")
    print(f"   - 真实阻抗范围: [{true_range[0]:.0f}, {true_range[1]:.0f}] (kg/m³·m/s)")
    print(f"   - 范围匹配度: {min(pred_range[1]/true_range[1], true_range[1]/pred_range[1]):.3f}")
    
    # 分段分析
    print(f"\n🔍 分段精度分析:")
    n_time, n_space = pred_imp.shape
    
    # 时间分段（浅层、中层、深层）
    shallow = slice(0, n_time//3)
    middle = slice(n_time//3, 2*n_time//3)
    deep = slice(2*n_time//3, n_time)
    
    for name, time_slice in [("浅层", shallow), ("中层", middle), ("深层", deep)]:
        pred_seg = pred_imp[time_slice, :]
        true_seg = true_imp[time_slice, :]
        seg_corr = np.corrcoef(pred_seg.flatten(), true_seg.flatten())[0, 1]
        seg_rmse = np.sqrt(np.mean((pred_seg - true_seg)**2))
        print(f"   - {name}段相关系数: {seg_corr:.3f}, RMSE: {seg_rmse:.0f}")

def run_well_focused_analysis(data, metrics):
    """
    运行井位专项分析套件
    
    Args:
        data: 结果数据字典
        metrics: 评估指标字典
    """
    print("\n🎯 运行井位专项分析套件...")
    
    # 1. 井位专项可视化
    # create_well_focused_visualization(data, metrics)
    
    # 2. 带井位标记的剖面图
    # create_profile_with_wells(data, metrics)
    
    # 3. 尝试seisvis专业井曲线
    if SEISVIS_AVAILABLE:
        create_seisvis_well_curves(data, metrics)
    
    print("✅ 井位专项分析套件完成!")

if __name__ == "__main__":
    """
    直接运行此脚本进行可视化
    """
    print("\n🚀 运行完整分析（标准 + 井位专项）...")
    success = visualize_results(show_plots=False, use_seisvis=True, data_dir='deep_learning_impedance_inversion_chl')
    
    # 额外运行井位专项分析
    data = load_results()
    # if data:
    #     metrics = calculate_metrics(data)
    #     run_well_focused_analysis(data, metrics)
    
    if success:
        # 打印详细指标
        data = load_results()
        if data:
            print_detailed_metrics(data)
            
        results_dir = "results"
        # 列出生成的图片文件
        image_files = [f for f in os.listdir(results_dir) if f.endswith('.png') and 
                      any(keyword in f for keyword in ['impedance', 'well', 'scatter', 'seisvis', 'analysis'])]
        
        for img_file in sorted(image_files):
            print(f"   📈 {img_file}")
            
    else:
        print("❌ 可视化失败")
        sys.exit(1)
