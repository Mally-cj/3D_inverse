# 深度学习地震波阻抗反演项目

本项目实现了基于深度学习与物理约束的地震波阻抗反演，采用半监督学习框架，结合井位掩码与正演物理模型，提升稀疏井约束下的反演精度。

---

## 1. 快速开始

### 1.1 环境依赖

- Python >= 3.7
- torch >= 1.10.0
- numpy, scipy, obspy, pylops

安装依赖：
```bash
pip install torch numpy scipy obspy pylops
```

### 1.2 数据准备

- 数据目录需包含地震数据（SEGY）、测井阻抗（SEGY/CSV）、井位信息（CSV）等。
- 运行前请确保数据路径在 `run.py` 中已正确配置。

#### 数据目录与文件说明（补充）

- **地震数据（3D体）**
  - `PSTM_resample1_lf_extension2.sgy` （753MB）：三维地震观测数据，空间范围 Line: 450-700，CMP: 212-1400，时间范围 3300.00-3900.00 ms。
  - `PSTM_resample1_lf_extension2.txt`：地震数据辅助信息。

- **阻抗数据**
  - `yyf_smo_train_Volume_PP_IMP.sgy` （753MB）：基于测井曲线插值生成的三维阻抗体，井位处为真实阻抗，其余为插值。
  - `yyf_smo_train_Volume_PP_IMP.txt`：阻抗数据辅助信息。

- **井数据（真实测井曲线）**
  - 井名及对应空间位置（Line, CMP）：
    - MO011: 594, 296
    - MO16: 572, 692
    - QIANS2: 591, 996
    - QIANS201: 532, 1053
    - QIANS3: 603, 1212
    - QIANS4: 561, 842
    - QIANS401: 504, 846
    - QIANS8: 499, 597
  - 井曲线数据文件（如 `MO011.txt`, `QIANS3.txt` 等）：每个文件为单井测井阻抗曲线。

- **井信息与辅助文件**
  - `WellGroup1_info.txt`：井的详细信息。
  - 其他井名.txt：各井的阻抗曲线数据。

##### 数据维度与范围

- **地震体/阻抗体维度**：与 `.sgy` 文件一致，空间范围 Line: 450-700，CMP: 212-1400，时间范围 3300.00-3900.00 ms。
- **井位**：上述8口井，空间坐标基于Line和CMP。

---

### 1.3 运行主流程

```bash
python run.py
```
- 训练与推理的所有中间结果将自动保存在 `logs/` 目录。

### 1.4 结果可视化

```bash
python visual_results.py
```
- 支持剖面、井点、掩码等多种可视化方式。

---

## 2. 算法流程与数学原理

### 2.1 数据流与核心变量

- **地震观测数据** $S_{obs}$：三维SEGY体，$(T, X, Y)$
- **完整阻抗体** $Z_{full}$：测井插值+真实井点，$(T, X, Y)$
- **井位掩码** $M$：二维权重掩码，$(X, Y)$，井位为1，远离井为0，过渡区为$(0,1)$
- **低频背景阻抗** $Z_{back}$：低通滤波获得

### 2.2 两阶段反演流程

#### 阶段一：子波学习（ForwardNet）
- 目标：利用井位高可信度区域，学习最优子波 $w_{learned}$
- 数学模型：
  $$L_{wavelet} = \| M \odot [F(\nabla Z_{full}, w)] - M \odot S_{obs} \|^2$$
- 代码片段：
```python
# 井位掩码M，完整阻抗Z_full，观测地震S_obs
synthetic, learned_wavelet = forward_net(DIFFZ(Z_full), w_init)
loss_wavelet = mse(M * synthetic, M * S_obs)
```

#### 阶段二：阻抗反演（UNet）
- 目标：利用学习到的子波和低频背景，反演全空间阻抗
- 数学模型：
  - 最小二乘初始化：
    $$Z_{init} = \arg\min_Z \| W \nabla Z - (S_{obs} - W \nabla Z_{back}) \|^2 + \epsilon \|Z\|^2$$
  - UNet残差学习：
    $$Z_{pred} = UNet([Z_{init}, S_{obs}]) + Z_{init}$$
- 代码片段：
```python
# 最小二乘初始化
Z_init = least_squares_init(S_obs, learned_wavelet, Z_back)
# UNet反演
Z_pred = unet(torch.cat([Z_init, S_obs], dim=1)) + Z_init
```

### 2.3 掩码机制与损失函数

- **井位掩码 $M$**：高可信度区域主导监督损失，过渡区平滑衰减
- **总损失函数**：
  $$ L_{total} = L_{unsup} + L_{sup} + L_{tv} $$
  - $L_{unsup}$：物理约束（正演一致性）
  - $L_{sup}$：井位有监督损失（掩码加权）
  - $L_{tv}$：总变分正则化（空间平滑）
- 代码片段：
```python
loss_unsup = mse(forward_net(DIFFZ(Z_pred), learned_wavelet)[0], S_obs)
loss_sup = gamma * mse(M * Z_pred, M * Z_full)
loss_tv = tv_loss(Z_pred, alpha)
total_loss = loss_unsup + loss_sup + loss_tv
```

### 2.4 关键算子实现

- **线性反射系数**：
```python
def DIFFZ(z):
    DZ = torch.zeros_like(z)
    DZ[..., :-1, :] = 0.5 * (z[..., 1:, :] - z[..., :-1, :])
    return DZ
```
- **井位掩码生成**：
```python
def generate_well_mask(well_positions, grid_shape, radius=15, sigma=5):
    mask = np.zeros(grid_shape)
    for (wx, wy) in well_positions:
        for x in range(max(0, wx-radius), min(grid_shape[0], wx+radius)):
            for y in range(max(0, wy-radius), min(grid_shape[1], wy+radius)):
                d = np.sqrt((x-wx)**2 + (y-wy)**2)
                if d <= radius:
                    mask[x, y] = max(mask[x, y], np.exp(-d**2/(2*sigma**2)))
    return mask
```

---

## 2.2.1 地震正演公式与物理量关系（补充）

#### 1. 波阻抗（Acoustic Impedance, $Z$）
- 定义：$Z = \rho v$，其中$\rho$为地层密度，$v$为地层纵波速度。
- 物理意义：反映地层对地震波传播的阻抗能力，是地震反演的核心目标。

#### 2. 反射系数（Reflection Coefficient, $R$）
- 定义：界面$k$处的反射系数$R_k$为：
  $$ R_k = \frac{Z_{k+1} - Z_k}{Z_{k+1} + Z_k} $$
- 物理意义：描述地震波在不同地层界面处的反射强度，是地震记录的主要成因。
- 线性近似（小对比）：
  $$R_k \approx 0.5 \cdot \frac{Z_{k+1} - Z_k}{Z_k}$$

#### 3. 地震记录（Seismic Trace, $S$）
- 由反射系数序列$R$与子波$w$卷积得到：
  $$S = w * R + n$$
  其中$*$为卷积，$n$为噪声。
- 结合阻抗表达：
  $$S = w * \nabla Z + n$$
  其中$\nabla Z$为阻抗差分（近似反射系数）。

#### 4. 低频背景阻抗（Low-frequency Background, $Z_{back}$）
- 由阻抗体低通滤波获得，补充地震数据中缺失的低频信息。
- 反演时，$Z_{back}$作为先验或初始模型，有助于约束反演结果。

#### 5. 正演流程总结
- 地震观测数据$S_{obs}$的生成过程：
  1. 真实阻抗体$Z_{full}$ → 差分/反射系数$R$（$\nabla Z$）
  2. $R$与子波$w$卷积，得到合成地震$S_{syn}$
  3. $S_{syn}$与观测地震$S_{obs}$对比，优化模型参数

- 公式链路：
  $$Z \xrightarrow{\nabla} R \xrightarrow{*w} S$$

#### 6. 代码与公式对应
- 差分算子：
  ```python
  def DIFFZ(z):
      DZ = torch.zeros_like(z)
      DZ[..., :-1, :] = 0.5 * (z[..., 1:, :] - z[..., :-1, :])
      return DZ
  ```
- 正演：
  ```python
  synthetic, learned_wavelet = forward_net(DIFFZ(Z_full), w_init)
  loss_wavelet = mse(M * synthetic, M * S_obs)
  ```

---

## 3. 代码结构说明

- `run.py`：主流程入口，包含数据加载、掩码生成、两阶段反演、模型训练与推理、结果保存等完整流程。
- `visual_results.py`：结果可视化脚本，支持剖面、井点、掩码等多种展示方式。
- `logs/`：所有训练、推理过程中的中间结果、模型权重、可视化图片等均自动存放于此目录。
- `Model/`：网络结构与工具函数（如UNet、ForwardNet、掩码生成等）。

---

## 4. 常见问题

- **数据格式要求？**
  - 地震数据、阻抗数据建议为SEGY或CSV，井位信息为CSV。
- **内存不足？**
  - 可在 `run.py` 中调整 batch_size、patch_size 或数据子集。
- **如何切换训练/推理？**
  - 修改 `run.py` 中的 `Train = True/False`。
- **可视化报错？**
  - 检查 `logs/` 目录下是否有推理结果。
---



