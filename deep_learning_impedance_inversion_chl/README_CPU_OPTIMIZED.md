# 地震阻抗反演深度学习代码 - CPU优化版本

## 概述
这是一个用于地震阻抗反演的深度学习代码，已经优化为可以在CPU和GPU上运行。

## 文件说明

### 主要文件
1. **`seismic_imp_2D_high_channel_model_bgp.py`** - 原始代码（已修改为CPU兼容）
2. **`seismic_imp_2D_high_channel_model_bgp_optimized.py`** - 带数据缓存的优化版本
3. **`seismic_imp_2D_high_channel_model_bgp_lite.py`** - 轻量级测试版本
4. **`cache_manager.py`** - 数据缓存管理工具
5. **`test_cpu.py`** - CPU兼容性测试脚本
6. **`test_main_cpu.py`** - 主要功能测试脚本

### 模型文件
- **`Model/`** 目录包含网络定义、工具函数等

## 主要修改内容

### 1. 设备兼容性
- 自动检测GPU/CPU并设置相应的数据类型
- 所有硬编码的CUDA调用都已替换为动态设备分配
- 修复了模型加载时的设备映射问题

### 2. 数据优化
- **轻量级版本**: 使用数据子集，减少内存使用
- **缓存版本**: 将数据处理结果保存到磁盘，避免重复处理
- **路径修复**: 修正了数据文件的相对路径

### 3. 性能优化
- 减少了默认的训练迭代次数（用于快速测试）
- 优化了数据加载流程
- 添加了进度提示

## 使用方法

### 1. 快速测试（推荐）
```bash
# 强制使用CPU
export CUDA_VISIBLE_DEVICES=""

# 运行轻量级版本进行测试
python seismic_imp_2D_high_channel_model_bgp_lite.py
```

### 2. 完整训练
```bash
# 编辑文件，将 Train = False 改为 Train = True
# 然后运行
python seismic_imp_2D_high_channel_model_bgp_lite.py
```

### 3. 使用数据缓存版本
```bash
# 第一次运行会处理数据并保存缓存
python seismic_imp_2D_high_channel_model_bgp_optimized.py

# 后续运行会直接加载缓存数据
python seismic_imp_2D_high_channel_model_bgp_optimized.py
```

### 4. 管理数据缓存
```bash
# 查看缓存信息
python cache_manager.py info

# 清理缓存
python cache_manager.py clear
```

## 配置选项

### 轻量级版本配置
在 `seismic_imp_2D_high_channel_model_bgp_lite.py` 中：

```python
Train = False          # True为训练模式，False为测试模式
USE_SUBSET = True      # 是否使用数据子集
SUBSET_SIZE = 50       # 数据子集大小
```

### 训练参数调整
```python
# 减少训练迭代次数用于快速测试
admm_iter = 5         # 前向网络训练轮数
admm_iter1 = 5        # UNet训练轮数
batch_size = 5        # 批次大小
```

## 故障排除

### 1. 内存不足
- 使用轻量级版本: `seismic_imp_2D_high_channel_model_bgp_lite.py`
- 减少 `SUBSET_SIZE` 和 `batch_size`
- 清理缓存: `python cache_manager.py clear`

### 2. 磁盘空间不足
- 避免使用缓存版本
- 使用轻量级版本
- 清理不必要的文件

### 3. 数据文件路径错误
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/yyf_smo_train_Volume_PP_IMP.sgy'
```
- 确保数据文件在 `../data/` 目录中
- 检查文件名是否正确

### 4. 模块导入错误
```
无法解析导入"cpp_to_py"
```
- 确保 `codes/` 目录中有相应的Python文件
- 检查路径设置

## 性能建议

### CPU优化
1. 使用轻量级版本进行初步测试
2. 逐步增加数据规模
3. 考虑使用多进程数据加载

### GPU优化
1. 设置 `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` 使用GPU
2. 增加批次大小
3. 使用完整数据集

## 输出示例

成功运行后会看到类似输出：
```
Using device: cpu
=== Processing Data ===
Processing subset of data...
Step 1: Loading wavelet...
Step 2: Loading impedance model (subset)...
Impedance model shape: (601, 50, 251)
...
✓ Network forward pass successful!
=== Program completed successfully! ===
```

## 注意事项

1. **首次运行**: 需要从SEGY文件加载数据，可能需要几分钟
2. **内存使用**: 完整数据集需要大量内存，建议先用轻量级版本测试
3. **训练时间**: 在CPU上训练会比较慢，建议用于测试和验证
4. **结果保存**: 网络权重会自动保存为 `.pth` 文件

## 技术支持

如果遇到问题，请检查：
1. Python环境和依赖包是否正确安装
2. 数据文件路径是否正确
3. 磁盘空间是否充足
4. 内存是否足够
