"""
数据来源说明：
- S_obs: 真实野外地震观测数据 (PSTM_resample1_lf_extension2.sgy)
- Z_full: 测井插值阻抗数据 (yyf_smo_train_Volume_PP_IMP.sgy)
  * 井位处：真实测井数据（8口井的精确阻抗值）
  * 其他位置：插值估计值（可用但不够精确）
- M_mask: 井位掩码，标记数据可信度分布
- 两阶段算法充分利用真实观测数据和井位约束
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch.optim
from Model.net2D import UNet, forward_model
from Model.utils import *
from torch.utils import data
from Model.joint_well import *
import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.utils.wavelets import ricker
from scipy.signal import filtfilt
from scipy import signal
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from obspy.io.segy.segy import _read_segy
import pdb
import sys
sys.path.append('..')
# import data_tools as tools
from icecream import ic 
sys.path.append('../codes')
sys.path.append('deep_learning_impedance_inversion_chl')
from cpp_to_py import generate_well_mask as generate_well_mask2
from cpp_to_py import get_wellline_and_mask as get_wellline_and_mask2
import psutil
import gc
from tqdm import tqdm

# 训练/测试模式切换
Train = False  # 设置为 True 进行训练并保存Forward网络权重；设置为 False 进行推理测试

# 📝 重要说明：
# - 首次使用时，建议先设置 Train = True 运行一次训练，生成Forward网络权重文件
# - Forward网络学习数据驱动的最优子波，对反演精度至关重要
# - 训练完成后，可设置 Train = False 进行推理测试

# 智能设备检测和参数配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# 根据设备自动调整参数
if device.type == 'cuda':
    print("📊 GPU mode: Using full dataset")
    dtype = torch.cuda.FloatTensor
    # GPU参数配置
    USE_FULL_DATA = True
    MAX_SPATIAL_SLICES = 251  # 完整数据
    BATCH_SIZE = 10
    PATCH_SIZE = 70
    N_WELL_PROFILES = 30
    ADMM_ITER = 100
    ADMM_ITER1 = 50
    MAX_TRAIN_SAMPLES = None  # 不限制
else:
    print("💻 CPU mode: Using optimized subset")
    dtype = torch.FloatTensor
    # CPU优化参数配置
    USE_FULL_DATA = False
    MAX_SPATIAL_SLICES = 50   # 减少数据量
    BATCH_SIZE = 1
    PATCH_SIZE = 48
    N_WELL_PROFILES = 10
    ADMM_ITER = 30
    ADMM_ITER1 = 15
    MAX_TRAIN_SAMPLES = 300

print(f"📋 Configuration:")
print(f"  - Spatial slices: {MAX_SPATIAL_SLICES}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Patch size: {PATCH_SIZE}")
print(f"  - Training samples: {MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES else 'unlimited'}")
print(f"  - Training iterations: {ADMM_ITER} + {ADMM_ITER1}")


#############################################################################################################
### 第2部分：数据加载 - 区分演示数据和工程数据
#############################################################################################################

print("\n" + "="*80) 
print("📂 第2部分：数据加载")
print("="*80)

print("\n🔄 加载工程用完整阻抗数据 (训练用)...")
print("   说明：是用测井数据插值后的完整阻抗，井位处精确，其他位置插值估计")
# 工程数据：测井插值得到的完整阻抗（这是实际工程的起点）
segy = _read_segy("data/yyf_smo_train_Volume_PP_IMP.sgy")
impedance_model_full = []
for i in range(0, len(segy.traces)):
    impedance_model_full.append(segy.traces[i].data)

impedance_model_full = np.array(impedance_model_full).reshape(251, len(impedance_model_full)//251, 601).transpose(2, 1, 0)

# 根据设备配置调整数据大小
if not USE_FULL_DATA:
    impedance_model_full = impedance_model_full[:, :MAX_SPATIAL_SLICES, :]

impedance_model_full = np.log(impedance_model_full)
print(f"✅ 完整阻抗数据加载完成: {impedance_model_full.shape}")
print(f"   数据构成：井位处为真实测井值，其他位置为插值估计值")

print("\n🌊 从完整阻抗数据提取低频背景...")
# 低频背景阻抗：从完整阻抗数据中提取低频成分
Z_back = []
for i in range(impedance_model_full.shape[2]):
    B, A = signal.butter(2, 0.012, 'low')  # 截止频率约12Hz
    m_loww = signal.filtfilt(B, A, impedance_model_full[..., i].T).T
    nsmooth = 3
    m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_loww)  # 时间方向平滑
    nsmooth = 3
    m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_low.T).T  # 空间方向平滑
    Z_back.append(m_low[..., None])
Z_back = np.concatenate(Z_back, axis=2)
print(f"✅ 低频背景阻抗生成完成: {Z_back.shape}")
print(f"   用途：为最小二乘初始化提供低频约束")

print("\n🌊 加载真实观测地震数据...")
# 观测地震数据：直接加载野外观测的地震数据
print("   📂 加载PSTM地震数据文件...")
segy_seismic = _read_segy("data/PSTM_resample1_lf_extension2.sgy")
S_obs = []
for i in range(0, len(segy_seismic.traces)):
    S_obs.append(segy_seismic.traces[i].data)

S_obs = np.array(S_obs).reshape(251, len(S_obs)//251, 601).transpose(2, 1, 0)

# 根据设备配置调整数据大小
if not USE_FULL_DATA:
    S_obs = S_obs[:, :MAX_SPATIAL_SLICES, :]

print(f"✅ 真实观测地震数据加载完成: {S_obs.shape}")
print(f"   数据来源：野外地震勘探观测数据")
print(f"   数据用途：作为反演算法的观测约束条件")

#############################################################################################################
### 第3部分：井位数据处理 - 定义井位和生成掩码
#############################################################################################################

print("\n" + "="*80)
print("🎯 第3部分：井位数据处理") 
print("="*80)

# 网格参数
nx, ny = S_obs.shape[1:3]
basex = 450
basey = 212

# 已知测井位置的绝对坐标（这些位置有准确的测井数据）
print("📍 定义已知测井位置...")

if not USE_FULL_DATA:
    # CPU模式：使用适合缩减网格的井位
    print("   💻 CPU模式：使用适配的井位配置")
    well_pos = [[10, 10], [20, 20], [30, 30], [40, 40]]  # 适合(50, 251)网格的井位
else:
    # GPU模式：使用原始完整井位
    print("   🖥️  GPU模式：使用完整井位配置")
    pos = [[594,295], [572,692], [591,996], [532,1053], [603,1212], [561,842], [504,846], [499,597]]
    # 转换为相对网格坐标
    well_pos = [[y-basey, x-basex] for [x, y] in pos]

print(f"✅ 井位信息:")
print(f"   - 测井位置数量: {len(well_pos)}")
print(f"   - 井位坐标 (网格): {well_pos}")
print(f"   - 这些位置的阻抗数据是准确的（真实测井值）")
print(f"   - 其他位置的阻抗数据通过插值获得（估计值）")

print("\n🎯 生成井位掩码...")
# 生成井位掩码：标记数据可信度分布
grid_shape = S_obs.shape[1:3]
M_well_mask_dict = generate_well_mask2(well_pos, grid_shape, well_range=15, sigma=5)

# 将字典转换为2D数组格式
M_well_mask = np.zeros(grid_shape)
for (line, cmp), weight in M_well_mask_dict.items():
    M_well_mask[line, cmp] = weight

print(f"✅ 井位掩码生成完成:")
print(f"   - 网格形状: {grid_shape}")
print(f"   - 井位影响范围: 15个网格点")
print(f"   - 掩码形状: {M_well_mask.shape}")
print(f"   - 井位数量: {len(M_well_mask_dict)}")
print(f"   - 掩码值范围: [{M_well_mask.min():.3f}, {M_well_mask.max():.3f}]")
print(f"   - 掩码值含义:")
print(f"     * M=1.0: 井位处，完整阻抗数据为真实测井值")
print(f"     * M=0.0: 远离井位，完整阻抗数据为插值估计值")
print(f"     * M∈(0,1): 井影响范围，可信度渐变过渡")

#############################################################################################################
### 第4部分：训练数据构建 - 三步骤生成训练样本
#############################################################################################################

print("\n" + "="*80)
print("📦 第4部分：训练数据构建")
print("="*80)

print("🔗 步骤1：生成随机连井剖面...")
print(f"   - 基于{len(well_pos)}口井生成{N_WELL_PROFILES}条随机连接路径")
print(f"   - 每条剖面垂直高度: 601个时间采样点")
print(f"   - 水平长度: 变长（根据井间路径决定）")

# 训练井位（添加标签用于路径生成）
train_well = add_labels(well_pos)
extension_length = 10  # 路径延拓长度

# 存储各类剖面数据
Z_back_profiles = []       # 低频背景剖面
Z_full_profiles = []       # 完整阻抗剖面  
S_obs_profiles = []        # 观测地震数据剖面
M_mask_profiles = []       # 井位掩码剖面
path_coords = []           # 剖面路径坐标

print("   正在生成连井剖面...")
for i in tqdm(range(N_WELL_PROFILES), desc="生成剖面"):
    # 生成第i条随机连井剖面的坐标点
    interpolated_points, vMask = get_wellline_and_mask2(well_pos, grid_shape, M_well_mask_dict)
    
    # 记录路径信息
    path_coords.append(interpolated_points)
    
    # 扩展掩码到时间维度 (601 × 剖面长度)
    vMask_time_extended = np.tile(vMask, (601, 1))
    M_mask_profiles.append(vMask_time_extended)
    
    # 步骤2：沿连井剖面提取各类数据
    # 提取低频背景沿剖面的数据
    Z_back_profiles.append(Z_back[:, interpolated_points[:, 0], interpolated_points[:, 1]])
    
    # 提取完整阻抗沿剖面的数据（关键：这是训练目标）
    Z_full_profiles.append(impedance_model_full[:, interpolated_points[:, 0], interpolated_points[:, 1]])
    
    # 提取观测地震数据沿剖面的数据
    S_obs_profiles.append(S_obs[:, interpolated_points[:, 0], interpolated_points[:, 1]])

print(f"✅ 步骤1&2完成：生成{N_WELL_PROFILES}条伪2D剖面")
print(f"   每条剖面包含4类数据：")
print(f"   - S_obs: 观测地震数据 (用于物理约束)")
print(f"   - Z_full: 完整阻抗数据 (训练目标，结合掩码使用)")
print(f"   - Z_back: 低频背景阻抗 (用于最小二乘初始化)")
print(f"   - M: 井位掩码 (标记数据可信度)")

print(f"\n📦 步骤3：滑窗切分统一尺寸...")
print(f"   - 输入: {N_WELL_PROFILES}条变长剖面 (601×变长)")
print(f"   - 输出: 统一尺寸训练块 (601×{PATCH_SIZE})")
print(f"   - 重叠步长: 5个点 (数据增强)")

patchsize = PATCH_SIZE
oversize = 5

# 存储切分后的训练数据
Z_back_patches = []       # 低频背景训练块
Z_full_patches = []       # 完整阻抗训练块
S_obs_patches = []        # 地震数据训练块
M_mask_patches = []       # 井位掩码训练块

print("   正在切分训练块...")
for i in tqdm(range(N_WELL_PROFILES), desc="切分数据"):
    # 使用滑窗方式将每条剖面切分成多个固定尺寸的训练块
    Z_back_patches.append(torch.tensor(image2cols(Z_back_profiles[i], (S_obs.shape[0], patchsize), (1, oversize))))
    Z_full_patches.append(torch.tensor(image2cols(Z_full_profiles[i], (S_obs.shape[0], patchsize), (1, oversize))))
    S_obs_patches.append(torch.tensor(image2cols(S_obs_profiles[i], (S_obs.shape[0], patchsize), (1, oversize))))
    M_mask_patches.append(torch.tensor(image2cols(M_mask_profiles[i], (S_obs.shape[0], patchsize), (1, oversize))))

# 拼接所有训练块 [N_samples, 1, 601, PATCH_SIZE]
Z_back_train_set = torch.cat(Z_back_patches, 0)[..., None].permute(0, 3, 1, 2).type(dtype)
Z_full_train_set = torch.cat(Z_full_patches, 0)[..., None].permute(0, 3, 1, 2).type(dtype)
S_obs_train_set = torch.cat(S_obs_patches, 0)[..., None].permute(0, 3, 1, 2).type(dtype)
M_mask_train_set = torch.cat(M_mask_patches, 0)[..., None].permute(0, 3, 1, 2).type(dtype)

print(f"✅ 步骤3完成：生成统一训练数据集")
print(f"   - 训练样本总数: {len(S_obs_train_set)}")
print(f"   - 每个样本大小: {S_obs_train_set.shape[2]}×{S_obs_train_set.shape[3]} (时间×空间)")
print(f"   - 数据类型: 4类 (地震、阻抗、背景、掩码)")

#############################################################################################################
### 第5部分：数据归一化和数据加载器
#############################################################################################################

print("\n" + "="*80)
print("🔧 第5部分：数据归一化和加载器构建")
print("="*80)

# 计算归一化参数（基于完整阻抗数据的范围）
logimpmax = impedance_model_full.max()
logimpmin = impedance_model_full.min()
print(f"📊 阻抗数据范围: [{logimpmin:.3f}, {logimpmax:.3f}]")

# 训练数据归一化
print("🔄 归一化训练数据...")
Z_full_norm = (Z_full_train_set - logimpmin) / (logimpmax - logimpmin)  # 阻抗归一化到[0,1]
S_obs_norm = 2 * (S_obs_train_set - S_obs_train_set.min()) / (S_obs_train_set.max() - S_obs_train_set.min()) - 1  # 地震数据归一化到[-1,1]
Z_back_norm = (Z_back_train_set - logimpmin) / (logimpmax - logimpmin)  # 低频背景归一化到[0,1]
# 掩码不需要归一化，保持[0,1]范围

# 应用训练样本限制（CPU模式）
if MAX_TRAIN_SAMPLES is not None:
    print(f"🔄 CPU模式：限制训练样本数量到 {MAX_TRAIN_SAMPLES}...")
    total_samples = len(S_obs_norm)
    if total_samples > MAX_TRAIN_SAMPLES:
        indices = torch.randperm(total_samples)[:MAX_TRAIN_SAMPLES]
        S_obs_norm = S_obs_norm[indices]
        Z_full_norm = Z_full_norm[indices]
        Z_back_norm = Z_back_norm[indices]
        M_mask_train_set = M_mask_train_set[indices]
        print(f"📊 训练样本数量: {total_samples} → {len(S_obs_norm)}")

print("🔧 准备测试数据...")
# 测试数据归一化（全尺寸数据）
S_obs_test_norm = 2 * (S_obs - S_obs.min()) / (S_obs.max() - S_obs.min()) - 1
Z_back_test_norm = (Z_back - logimpmin) / (logimpmax - logimpmin)
Z_full_test_norm = (impedance_model_full - logimpmin) / (logimpmax - logimpmin)
Z_true_test_norm = (impedance_model_full - logimpmin) / (logimpmax - logimpmin)  # 仅用于评估

# 转换为PyTorch张量格式
test_S_obs = torch.tensor(S_obs_test_norm[..., None]).permute(2, 3, 0, 1).type(dtype)
test_Z_back = torch.tensor(Z_back_test_norm[..., None]).permute(2, 3, 0, 1).type(dtype)
test_Z_full = torch.tensor(Z_full_test_norm[..., None]).permute(2, 3, 0, 1).type(dtype)
test_Z_true = torch.tensor(Z_true_test_norm[..., None]).permute(2, 3, 0, 1).type(dtype)

print("📦 构建数据加载器...")
# 构建训练数据加载器（修正：确保数据对应关系正确）
Train_loader = data.DataLoader(
    data.TensorDataset(
        S_obs_norm,          # 观测地震数据 - 用于UNet输入和物理约束
        Z_full_norm,         # 完整阻抗数据 - 训练目标（结合掩码）
        Z_back_norm,         # 低频背景阻抗 - 用于最小二乘初始化
        M_mask_train_set     # 井位掩码 - 标记数据可信度权重
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 测试数据加载器
Test_loader = data.DataLoader(
    data.TensorDataset(
        test_S_obs,          # 测试地震数据
        test_Z_full,         # 测试完整阻抗
        test_Z_back,         # 测试低频背景
        test_Z_true          # 测试真实阻抗（仅用于评估）
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

print(f"✅ 数据加载器构建完成:")
print(f"   - 训练批大小: {BATCH_SIZE}")
# print(f"   - 训练批数: {l# 运行修正版训练
# python seismic_imp_2D_high_channel_model_bgp_corrected.py

#############################################################################################################
### 第6部分：子波初始化和数学算子定义
#############################################################################################################

print("\n" + "="*80)
print("🔧 第6部分：子波初始化和算子定义")
print("="*80)

# 初始子波生成（用于加速子波模块收敛）
print("🌊 生成初始子波...")
wav0 = wavelet_init(S_obs_norm.cpu().type(torch.float32), 101).squeeze().numpy()
size = S_obs.shape[0]

# 构建卷积矩阵
W = pylops.utils.signalprocessing.convmtx(wav0, size, len(wav0) // 2)[:size]
W = torch.tensor(W, dtype=torch.float32, device=device)

# 高斯窗函数（用于子波平滑）
N = len(wav0)
fp = 30  # 主频
fs = 1000  # 采样频率
std = int((fs/fp)/2)  # 标准差
gaussian_window = gaussian(N, std)
gaus_win = torch.tensor(gaussian_window[None, None, :, None]).type(dtype)

print(f"✅ 子波初始化完成:")
print(f"   - 初始子波长度: {len(wav0)}")
print(f"   - 卷积矩阵大小: {W.shape}")
print(f"   - 高斯窗参数: std={std}")

print("🔧 定义数学算子...")
# 阻抗差分算子（计算反射系数）
def DIFFZ(z):
    """
    计算阻抗的空间梯度，得到反射系数
    输入: z - 阻抗数据 [batch, channel, time, space]
    输出: DZ - 反射系数 [batch, channel, time, space]
    """
    DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]], device=device).type(dtype)
    DZ[..., :-1, :] = 0.5 * (z[..., 1:, :] - z[..., :-1, :])
    return DZ



# 总变分正则化损失函数
def tv_loss(x, alfa):
    """
    总变分正则化损失，保持空间连续性
    输入: x - 预测阻抗 [batch, channel, time, space]
          alfa - 正则化权重
    输出: TV损失值
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])    # 水平梯度
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])    # 垂直梯度
    return alfa * torch.mean(2*dh[..., :-1, :] + dw[..., :, :-1])

# 子波初始化函数
def wavelet_init(seismic_data, wavelet_length):
    """
    从地震数据估计初始子波
    输入: seismic_data - 地震数据
          wavelet_length - 子波长度
    输出: 估计的初始子波
    """
    # 简化实现：使用Ricker子波作为初始估计
    dt = 0.001
    t = np.arange(wavelet_length) * dt
    f0 = 30  # 主频30Hz
    wav = (1 - 2*np.pi**2*f0**2*t**2) * np.exp(-np.pi**2*f0**2*t**2)
    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("✅ 辅助函数定义完成")

#############################################################################################################
### 第7部分：网络初始化
#############################################################################################################

print("\n" + "="*80)
print("🤖 第7部分：网络初始化")
print("="*80)

# UNet阻抗反演网络
print("🏗️  初始化UNet反演网络...")
net = UNet(
    in_ch=2,                 # 输入通道：[最小二乘初始解, 观测地震数据]
    out_ch=1,                # 输出通道：阻抗残差
    channels=[8, 16, 32, 64],
    skip_channels=[0, 8, 16, 32],
    use_sigmoid=True,        # 输出归一化到[0,1]
    use_norm=False
).to(device)

# Forward建模网络（子波学习）
print("⚡ 初始化Forward建模网络...")
forward_net = forward_model(nonlinearity="tanh").to(device)

print(f"✅ 网络初始化完成:")
print(f"   - UNet参数量: {sum(p.numel() for p in net.parameters()):,}")
print(f"   - Forward网络参数量: {sum(p.numel() for p in forward_net.parameters()):,}")
print(f"   - 设备: {device}")

#############################################################################################################
### 第8部分：训练算法 - 两阶段训练
#############################################################################################################

if Train:
    print("\n" + "="*80)
    print("🚀 第8部分：两阶段训练算法")
    print("="*80)
    
    # 训练参数
    lr = 1e-3
    yita = 1e-1    # 井约束损失权重
    mu = 5e-4      # TV正则化权重
    beta = 0       # 额外监督损失权重
    
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizerF = torch.optim.Adam(forward_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    
    # 损失函数
    mse = torch.nn.MSELoss()
    
    print(f"📋 训练参数配置:")
    print(f"   - 学习率: {lr}")
    print(f"   - 井约束权重: {yita}")
    print(f"   - TV正则化权重: {mu}")
    print(f"   - 阶段1轮次: {ADMM_ITER}")
    print(f"   - 阶段2轮次: {ADMM_ITER1}")
    
    #########################################################################################################
    ### 阶段1：子波学习
    #########################################################################################################
    
    print("\n" + "-"*60)
    print("🌊 阶段1：数据驱动的子波学习")
    print("-"*60)
    print("目标：利用完整阻抗数据（结合井位掩码）学习最优子波")
    print("损失：L_wavelet = ||M ⊙ [ForwardNet(∇Z_full, w_0)]_synth - M ⊙ S_obs||²")
    
    admm_iter = ADMM_ITER
    psnr_net_total = []
    total_lossF = []
    epsI = 0.1
    
    print("开始子波学习训练...")
    for i in range(admm_iter):
        epoch_loss = 0
        batch_count = 0
        
        # 修正：变量名对应我们的设计
        for S_obs_batch, Z_full_batch, Z_back_batch, M_mask_batch in Train_loader:
            optimizerF.zero_grad()
            
            # 阶段1核心：加权子波学习
            # 目标：在高可信度区域（M=1.0）学习最优子波
            # 公式：L_wavelet = ||M ⊙ [ForwardNet(∇Z_full, w_0)]_synth - M ⊙ S_obs||²
            
            # 计算完整阻抗的反射系数
            reflection_coeff = DIFFZ(Z_full_batch)
            
            # Forward网络：输出[合成地震数据, 学习的子波]
            synthetic_seismic, learned_wavelet = forward_net(
                reflection_coeff, 
                torch.tensor(wav0[None, None, :, None], device=device)
            )
            
            # 加权损失：井位掩码确保在高可信度区域主导子波学习
            # M_mask_batch=1.0的位置：Z_full_batch为真实测井值，可信度高
            # M_mask_batch≈0.0的位置：Z_full_batch为插值估计值，可信度低
            lossF = mse(
                M_mask_batch * synthetic_seismic, 
                M_mask_batch * S_obs_batch
            ) * S_obs_batch.shape[3]
            
            lossF.backward()
            optimizerF.step()
            
            epoch_loss += lossF.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        total_lossF.append(avg_loss)
        
        if i % 20 == 0:
            print(f"   Epoch {i:04d}/{admm_iter:04d}, 子波学习损失: {avg_loss:.6f}")
            print(f"      说明：损失越小，学习的子波在高可信度区域拟合观测数据越好")
    
    # 提取学习到的子波
    print("🎯 提取优化后的子波...")
    with torch.no_grad():
        _, wav_learned = forward_net(
            DIFFZ(Z_full_batch), 
            torch.tensor(wav0[None, None, :, None], device=device)
        )
        wav_learned_np = wav_learned.detach().cpu().squeeze().numpy()
    
    # 子波后处理（高斯窗平滑）
    N = len(wav_learned_np)
    std = 25
    gaussian_window = gaussian(N, std)
    wav_learned_smooth = gaussian_window * (wav_learned_np - wav_learned_np.mean())
    
    print(f"✅ 阶段1完成：子波学习")
    print(f"   - 训练轮次: {admm_iter}")
    print(f"   - 最终损失: {total_lossF[-1]:.6f}")
    print(f"   - 学习子波长度: {len(wav_learned_smooth)}")
    
    #########################################################################################################
    ### 阶段2：UNet阻抗反演
    #########################################################################################################
    
    print("\n" + "-"*60)
    print("🎯 阶段2：UNet阻抗反演")
    print("-"*60)
    print("目标：使用学习的子波进行高精度阻抗反演")
    print("策略：最小二乘初始化 + UNet残差学习")
    print("损失：L_total = L_unsup + L_sup + L_tv")
    
    # 构建基于学习子波的卷积算子
    print("🔧 构建学习子波的卷积算子...")
    nz = S_obs_batch.shape[2]
    S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(0.5 * torch.ones(nz - 1), diagonal=-1)
    S[0] = S[-1] = 0
    
    WW = pylops.utils.signalprocessing.convmtx(wav_learned_smooth/wav_learned_smooth.max(), size, len(wav_learned_smooth) // 2)[:size]
    WW = torch.tensor(WW, dtype=torch.float32, device=device)
    WW = WW @ S.to(device)
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device)
    
    admm_iter1 = ADMM_ITER1
    print(f"开始UNet反演训练 (共{admm_iter1}轮)...")
    
    for i in range(admm_iter1):
        epoch_loss = 0
        epoch_loss_sup = 0
        epoch_loss_unsup = 0
        epoch_loss_tv = 0
        batch_count = 0
        
        # 修正：确保变量名对应设计
        for S_obs_batch, Z_full_batch, Z_back_batch, M_mask_batch in Train_loader:
            optimizer.zero_grad()
            
            # 步骤1：最小二乘初始化
            # 使用学习的子波构建的算子进行初始化
            # 公式：Z_init = argmin ||W∇Z - (S_obs - W∇Z_back)||² + ε||Z||²
            datarn = torch.matmul(WW.T, S_obs_batch - torch.matmul(WW, Z_back_batch))
            x, _, _, _ = torch.linalg.lstsq(PP[None, None], datarn)
            Z_init = x + Z_back_batch  # 加回低频背景
            Z_init = (Z_init - Z_init.min()) / (Z_init.max() - Z_init.min())  # 归一化
            
            # 步骤2：UNet残差学习
            # UNet输入：[最小二乘初始解, 观测地震数据]
            # UNet输出：阻抗残差 ΔZ
            # 最终预测：Z_pred = ΔZ + Z_init
            Z_pred = net(torch.cat([Z_init, S_obs_batch], dim=1)) + Z_init
            
            # 三项损失函数计算
            
            # 1. 井约束损失（差异化监督）
            # 公式：L_sup = M ⊙ ||Z_pred - Z_full||²
            # 含义：在井位处（M=1.0）强制匹配真实测井值
            #       在插值处（M≈0.0）几乎无监督约束
            loss_sup = yita * mse(
                M_mask_batch * Z_pred, 
                M_mask_batch * Z_full_batch
            ) * Z_full_batch.shape[3] / 3
            
            # 2. 物理约束损失（正演一致性）
            # 公式：L_unsup = ||ForwardModel(Z_pred, W_learned) - S_obs||²
            # 含义：确保预测阻抗的正演结果与观测地震数据一致
            # 说明：forward_net是子波校正网络，输入初始子波，输出校正后的最优子波
            pred_reflection = DIFFZ(Z_pred)
            pred_seismic, _ = forward_net(
                pred_reflection, 
                torch.tensor(wav0[None, None, :, None], device=device)  # 始终使用初始子波
            )
            loss_unsup = mse(pred_seismic, S_obs_batch)
            
            # 3. 总变分正则化损失（空间平滑性）
            # 公式：L_tv = α·TV(Z_pred)
            # 含义：保证反演结果的空间连续性
            loss_tv = tv_loss(Z_pred, mu)
            
            # 总损失
            total_loss = loss_unsup + loss_tv + loss_sup
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            epoch_loss += total_loss.item()
            epoch_loss_sup += loss_sup.item()
            epoch_loss_unsup += loss_unsup.item()
            epoch_loss_tv += loss_tv.item()
            batch_count += 1
        
        # 输出训练进度
        if i % 2 == 0:
            avg_total = epoch_loss / batch_count
            avg_sup = epoch_loss_sup / batch_count
            avg_unsup = epoch_loss_unsup / batch_count
            avg_tv = epoch_loss_tv / batch_count
            print(f"   Epoch {i:04d}/{admm_iter1:04d}")
            print(f"      总损失: {avg_total:.6f}")
            print(f"      井约束损失: {avg_sup:.6f} (高可信度区域匹配)")
            print(f"      物理约束损失: {avg_unsup:.6f} (正演一致性)")
            print(f"      TV正则化损失: {avg_tv:.6f} (空间平滑性)")
    
    print(f"✅ 阶段2完成：UNet阻抗反演训练")
    
    # 保存模型
    save_path = 'logs/model/Uet_TV_IMP_7labels_channel3.pth'
    torch.save(net.state_dict(), save_path)
    print(f"💾 UNet模型已保存: {save_path}")
    
    # 保存Forward网络（子波学习网络）
    forward_save_path = 'logs/model/forward_net_wavelet_learned.pth'
    torch.save(forward_net.state_dict(), forward_save_path)
    print(f"💾 Forward网络已保存: {forward_save_path}")
    print(f"   说明：Forward网络包含训练时学习的最优子波参数")

#############################################################################################################
### 第9部分：测试和结果评估
#############################################################################################################

if not Train:
    print("\n" + "="*80)
    print("🔍 第9部分：模型测试和结果评估")
    print("="*80)
    
    # 加载预训练模型
    save_path = 'logs/model/Uet_TV_IMP_7labels_channel3.pth'
    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()
    print(f"✅ UNet模型加载完成: {save_path}")
    
    # 加载预训练的Forward网络（子波学习网络）
    forward_save_path = 'logs/model/forward_net_wavelet_learned.pth'
    try:
        forward_net.load_state_dict(torch.load(forward_save_path, map_location=device))
        forward_net.eval()
        print(f"✅ Forward网络加载完成: {forward_save_path}")
        use_learned_wavelet = True
    except FileNotFoundError:
        print(f"⚠️  Forward网络文件未找到: {forward_save_path}")
        print("   📝 解决方案：")
        print("      1. 设置 Train = True 并运行训练来生成Forward网络权重")
        print("      2. 或者当前将使用初始子波进行推理（性能可能下降）")
        print("   🔧 当前选择方案2：使用初始子波继续推理")
        use_learned_wavelet = False
    
    # 推理阶段：构建子波算子
    print("🔧 构建推理用子波算子...")
    size = S_obs.shape[0]
    nz = size
    epsI = 0.1
    
    # 构建差分算子
    S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(0.5 * torch.ones(nz - 1), diagonal=-1)
    S[0] = S[-1] = 0
    
    if use_learned_wavelet:
        print("   使用训练好的Forward网络获取学习的子波...")
        # 使用训练好的forward_net获取学习的子波
        with torch.no_grad():
            # 使用第一个测试样本获取学习的子波
            sample_S_obs = torch.tensor(S_obs_test_norm[..., None]).permute(2, 3, 0, 1).type(dtype)[:1]
            sample_Z_full = torch.tensor(Z_full_test_norm[..., None]).permute(2, 3, 0, 1).type(dtype)[:1]
            sample_reflection = DIFFZ(sample_Z_full)
            
            _, learned_wavelet = forward_net(
                sample_reflection, 
                torch.tensor(wav0[None, None, :, None], device=device)
            )
            wav_learned_np = learned_wavelet.detach().cpu().squeeze().numpy()
        
        # 子波后处理（高斯窗平滑）
        N = len(wav_learned_np)
        std = 25
        gaussian_window = gaussian(N, std)
        wav_final = gaussian_window * (wav_learned_np - wav_learned_np.mean())
        wav_final = wav_final / wav_final.max()  # 归一化
        
        print(f"   ✅ 使用学习的子波: 长度={len(wav_final)}")
    else:
        print("   使用初始子波...")
        wav_final = wav0 / wav0.max()
        print(f"   ⚠️  使用初始子波: 长度={len(wav_final)}")
    
    # 使用最终子波构建卷积算子
    WW = pylops.utils.signalprocessing.convmtx(wav_final, size, len(wav_final) // 2)[:size]
    WW = torch.tensor(WW, dtype=torch.float32, device=device)
    WW = WW @ S.to(device)
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device)
    
    print(f"✅ 推理子波算子构建完成:")
    print(f"   - 子波长度: {len(wav_final)}")
    print(f"   - 卷积算子形状: {WW.shape}")
    print(f"   - 子波类型: {'学习的子波' if use_learned_wavelet else '初始子波'}")
    
    print("🔍 开始测试...")
    # 新增：收集所有批次，拼成3D体
    all_pred = []
    all_true = []
    all_input = []
    all_back = []  # 收集低频背景阻抗
    all_sesimic=[]  ##收集观测地震数据
    with torch.no_grad():
        for batch_idx, (test_S_obs, test_Z_full, test_Z_back, test_Z_true) in enumerate(Test_loader):
            datarn = torch.matmul(WW.T, test_S_obs - torch.matmul(WW, test_Z_back))
            x, _, _, _ = torch.linalg.lstsq(PP[None, None], datarn)
            Z_init = x + test_Z_back
            Z_init = (Z_init - Z_init.min()) / (Z_init.max() - Z_init.min())
            Z_pred = net(torch.cat([Z_init, test_S_obs], dim=1)) + Z_init
            all_pred.append(Z_pred.cpu().numpy())
            all_true.append(test_Z_full.cpu().numpy())
            all_input.append(test_S_obs.cpu().numpy())
            all_back.append(test_Z_back.cpu().numpy())  # 新增
            all_sesimic.append(test_S_obs.cpu().numpy())
            print(f"   处理批次 {batch_idx + 1}/{len(Test_loader)}")
    print("✅ 测试完成")
    
    # 拼成3D体 [N, 1, time, space] -> [N, time, space]
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_input = np.concatenate(all_input, axis=0)
    all_back = np.concatenate(all_back, axis=0)  # 新增
    all_pred = np.squeeze(all_pred, axis=1)
    all_true = np.squeeze(all_true, axis=1)
    all_input = np.squeeze(all_input, axis=1)
    all_back = np.squeeze(all_back, axis=1)  # 新增
    all_sesimic = np.squeeze(all_sesimic, axis=1)  # 新增
    # 反归一化
    all_pred_imp = np.exp(all_pred * (logimpmax - logimpmin) + logimpmin)
    all_true_imp = np.exp(all_true * (logimpmax - logimpmin) + logimpmin)
    all_back_imp = np.exp(all_back * (logimpmax - logimpmin) + logimpmin)
    ##对地震数据单独归一化
    all_sesimic = (all_sesimic - all_sesimic.min()) / (all_sesimic.max() - all_sesimic.min())
    # 保存为3D体
    print(f"\n💾 保存推理结果3D数据...")
    np.save('logs/results/prediction_sample.npy', all_pred)
    np.save('logs/results/true_sample.npy', all_true)
    np.save('logs/results/input_sample.npy', all_input)
    np.save('logs/results/seismic_record.npy', all_sesimic)
    
    np.save('logs/results/prediction_impedance.npy', all_pred_imp)
    np.save('logs/results/true_impedance.npy', all_true_imp)
    np.save('logs/results/background_impedance.npy', all_back_imp)  # 新增
    print(f"   ✅ 推理3D数据已保存: logs/results/prediction_impedance.npy, logs/results/true_impedance.npy 等 shape: {all_pred_imp.shape}")
    print(f"   ✅ 低频背景阻抗已保存: logs/results/background_impedance.npy shape: {all_back.shape}")
    print("\n" + "="*80)
    print("🎉 程序执行完成")
    print("="*80)

