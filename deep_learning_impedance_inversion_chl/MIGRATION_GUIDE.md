# 版本迁移指南

## 🚀 统一版本优势

我们已经将原来的3个版本整合为1个智能统一版本，具有以下优势：

### ✅ 自动适配
- **智能设备检测**：自动识别GPU/CPU环境
- **参数自动调整**：根据硬件能力优化配置
- **内存友好**：CPU模式下数据减少95.8%，内存需求降低至~129MB

### ✅ 一致体验
- **统一接口**：无论GPU还是CPU，使用相同的代码
- **完整算法**：CPU模式仍保持完整的反演算法
- **实时监控**：显示内存使用和处理进度

## 📋 配置对比

| 环境 | 数据规模 | 批大小 | 训练轮次 | 内存需求 |
|------|----------|--------|----------|----------|
| **GPU** | 601×1189×251 | 10 | 100+50 | 8GB+ |
| **CPU** | 601×50×251 | 1 | 30+15 | 2-4GB |

## 🔄 迁移步骤

### 从原始版本迁移
```bash
# 原来
python seismic_imp_2D_high_channel_model_bgp.py  # 只能GPU

# 现在
python seismic_imp_2D_high_channel_model_bgp.py  # 自动GPU/CPU
```

### 从Lite版本迁移
```bash
# 原来
python seismic_imp_2D_high_channel_model_bgp_lite.py  # 虚拟数据，不可用

# 现在
python seismic_imp_2D_high_channel_model_bgp.py  # 真实数据，完整算法
```

### 从CPU实用版本迁移
```bash
# 原来
python seismic_imp_2D_high_channel_model_bgp_cpu_practical.py  # 手动配置

# 现在
python seismic_imp_2D_high_channel_model_bgp.py  # 自动配置
```

## 💡 使用建议

### 新用户
直接使用统一版本即可，无需了解其他版本。

### 现有用户
建议迁移到统一版本，享受更好的用户体验：
- 无需手动选择版本
- 无需手动配置参数
- 获得最优性能

## ⚠️ 注意事项

1. **数据路径**：确保 `../data/` 目录存在且包含必要的SEGY文件
2. **依赖库**：统一版本需要额外的依赖：`psutil`, `tqdm`
3. **训练模式**：修改代码中的 `Train = True/False` 来切换训练/测试模式

## 🛠️ 安装依赖

```bash
pip install psutil tqdm
```

## 📞 支持

如果遇到问题，请检查：
1. 数据文件是否存在
2. 依赖库是否安装完整
3. 内存是否充足（至少4GB推荐）
