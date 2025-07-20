# 地震阻抗反演可视化工具使用指南

## 文件结构

```
deep_learning_impedance_inversion_chl/
├── seismic_imp_2D_high_channel_model_bgp_corrected.py  # 主训练和推理脚本
├── visual_result.py                                    # 独立可视化工具
├── *.npy                                              # 推理结果数据文件
└── *.png                                              # 生成的可视化图表
```

## 使用流程

### 1. 运行主程序进行推理

```bash
# 设置推理模式
# 在主文件中确保 Train = False
python seismic_imp_2D_high_channel_model_bgp_corrected.py
```

主程序会生成以下数据文件：
- `prediction_sample.npy`: 归一化预测结果
- `true_sample.npy`: 归一化真实结果
- `input_sample.npy`: 输入地震数据
- `prediction_impedance.npy`: 线性域预测阻抗
- `true_impedance.npy`: 线性域真实阻抗

### 2. 运行可视化工具

```bash
# 生成专业可视化图表
python visual_result.py
```

可视化工具会生成：
- `impedance_inversion_analysis.png`: 综合分析图
- `well_impedance_comparison.png`: 井位对比图
- `scatter_analysis.png`: 散点图分析

## 可视化功能详解

### 1. 综合分析图 (impedance_inversion_analysis.png)

包含四个子图：
- **1D阻抗曲线对比**: 真实vs预测阻抗曲线
- **真实阻抗剖面**: 2D真实阻抗分布
- **预测阻抗剖面**: 2D预测阻抗分布
- **误差分布图**: 绝对误差的空间分布

### 2. 井位对比图 (well_impedance_comparison.png)

显示多个代表性位置的1D阻抗曲线对比，每个子图包含：
- 真实阻抗曲线（蓝色实线）
- 预测阻抗曲线（红色虚线）
- 相关系数评估

### 3. 散点图分析 (scatter_analysis.png)

包含两个子图：
- **预测vs真实散点图**: 评估整体预测精度
- **误差直方图**: 分析误差分布特征

## 专业评估指标

### 整体精度指标
- **相关系数 (R)**: 衡量预测与真实值的线性相关性
- **RMSE**: 均方根误差，反映平均预测误差
- **相对误差**: 百分比形式的平均相对偏差
- **SSIM**: 结构相似性指数，评估2D图像的结构一致性

### 数据范围分析
- 预测和真实阻抗的数值范围对比
- 范围匹配度评估

### 分段精度分析
- 按深度分层（浅层、中层、深层）分析精度
- 识别不同深度的反演效果差异

## 高级使用

### 自定义可视化

```python
from visual_result import visualize_results, load_results, calculate_metrics

# 基础可视化
visualize_results(show_plots=True)  # 显示图表

# 自定义分析
data = load_results()
metrics = calculate_metrics(data)
print(f"相关系数: {metrics['correlation']:.4f}")
```

### 批量处理

如果有多个推理结果需要对比，可以修改 `visual_result.py` 中的文件路径：

```python
# 在 load_results() 函数中修改文件路径
data['prediction_sample'] = np.load('path/to/prediction_sample.npy')
```

## 故障排除

### 常见问题

1. **文件未找到错误**
   ```
   ❌ 缺少以下文件: ['prediction_sample.npy', ...]
   ```
   - 解决：先运行主程序进行推理

2. **图表不显示中文**
   - 已解决：使用英文标签确保兼容性

3. **内存不足**
   - 解决：可视化工具已优化内存使用，支持大数据

### 性能优化建议

- 对于大数据集，散点图会自动采样显示
- 可视化过程会自动关闭图表窗口释放内存
- 所有图表保存为高分辨率PNG格式

## 扩展功能

### 添加新的可视化类型

在 `visual_result.py` 中添加新函数：

```python
def create_custom_analysis(data, metrics, save_path='custom_analysis.png'):
    """自定义分析函数"""
    # 你的可视化代码
    pass
```

### 修改评估指标

在 `calculate_metrics()` 函数中添加新指标：

```python
# 添加新的地球物理指标
metrics['new_metric'] = your_calculation(pred_data, true_data)
```

## 技术规格

- **支持的数据格式**: NumPy .npy 文件
- **图像格式**: PNG, 300 DPI
- **颜色映射**: 
  - 阻抗剖面: viridis (地球物理标准)
  - 误差分布: hot (误差可视化标准)
- **字体**: Arial, DejaVu Sans (跨平台兼容)

---

**注意**: 此工具专为地震阻抗反演结果分析设计，生成的图表符合地球物理学术发表标准。
