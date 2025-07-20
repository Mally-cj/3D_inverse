## 2024.06.64
## Hongling Chen
## Xi'an Jiaotong University
## multichannel seismic impedance inversion by semi-supervised manner for # 📚 演示用数据加载：在实际工程中不存在完整的真实阻抗
# 这里为了算法验证和训练目的，使用SEGY文件模拟"真实"阻抗
print("🗄️  Loading reference impedance (for algorithm validation only)...")
print("⚠️  注意：实际工程中不存在这样的完整真实阻抗数据")
segy=_read_segy("../data/yyf_smo_train_Volume_PP_IMP.sgy")

impedance_model = np.array([trace.data for trace in segy.traces])
impedance_model = impedance_model.reshape(251, 1189, 601).transpose(2, 1, 0)

# 根据设备配置调整数据大小
if not USE_FULL_DATA:
    print(f"📊 Reducing data size for CPU: {impedance_model.shape} -> (601, {MAX_SPATIAL_SLICES}, 251)")
    impedance_model = impedance_model[:, :MAX_SPATIAL_SLICES, :]

impedance_model = np.log(impedance_model)
print(f"✅ Reference impedance loaded: {impedance_model.shape}")
print(f"💾 Memory usage: {get_memory_usage():.1f} MB")cal application
## Test: 

"""
重要说明：实际工程应用与演示代码的区别
========================================

实际工程中的数据情况：
1. 测井数据：少数几口井位置的准确波阻抗数据（1D，高成本获取）
   - 井位坐标定义在 pos 变量中，包含8口井的位置
   - 这些位置的阻抗数据是通过测井直接获得的，准确度高
2. 插值阻抗：利用测井数据通过地质统计学或其他方法插值得到的完整3D阻抗模型（不够准确）
   - 在井位处：准确（等于测井数据）
   - 在井间：不准确（通过插值估算）
3. 低频背景：对插值阻抗进行低通滤波得到的低频趋势
4. 地震数据：通过地震勘探获得的反射波数据

数据获取的三步骤流程：
========================================
第1步：随机路径生成伪二维剖面
- 使用随机路径连接8口井，生成N_WELL_PROFILES条连井剖面
- 每个剖面高度固定：601个时间采样点
- 每个剖面长度不定：取决于井间连接路径
- 输出：N_WELL_PROFILES条不定长的伪二维剖面

第2步：数据提取
- 根据连井剖面坐标，从完整3D数据中提取对应的2D剖面数据
- 提取内容：低频背景、插值阻抗、真实阻抗、地震数据、井位掩码
- 保持剖面的原始尺寸：601×变长

第3步：统一裁剪
- 将不定长剖面裁剪成统一大小：601×PATCH_SIZE
- 使用滑窗方式进行数据增强，重叠步长为5个点
- 最终生成统一规格的训练数据集

井位约束机制：
- well_pos: 定义了8口井的空间位置 
- vWellMask: 生成井位掩码，标记哪些区域受井约束
- index*out, index*Cimp1: 在训练中只在井位处计算监督损失
- 这样确保网络在井位处输出与测井数据一致

本演示代码的简化：
- 为了训练和验证目的，使用同一个SEGY文件模拟不同类型的数据
- impedance_model：作为训练目标（在实际中不存在完整的真实阻抗）
- impedance_model_log：模拟测井插值结果（实际中来自井数据插值）
- mback：从插值阻抗提取的低频背景（符合实际工程做法）

在实际部署时需要：
- 将impedance_model_log替换为真实的测井插值数据
- 移除impedance_model相关的训练目标代码
- 调整为无监督或弱监督学习策略
""" 


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch.optim
from Model.net2D import UNet, forward_model  # unet
from Model.utils import *  # unet
from torch.utils import data
from Model.joint_well import *  # unet
import matplotlib.pyplot as plt
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
import data_tools as tools
from icecream import ic 
sys.path.append('../codes')
from cpp_to_py import generate_well_mask as generate_well_mask2
from cpp_to_py import get_wellline_and_mask as get_wellline_and_mask2
import psutil
import gc
from tqdm import tqdm

# 内存监控函数
def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


Train = True

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

#######################################子波、阻抗、低频阻抗数据以及地震数据的生成过程############################################
### wavelet  为了方便，调用了pylops中现成的子波函数
dt0 = 0.001
ntwav = 51 #half size
wav, twav, wavc = ricker(np.arange(ntwav)*dt0, 30)

# 在实际工程应用中，我们只有以下真实数据：
# 1. 测井插值得到的阻抗数据（不完整但是起点）
# 2. 观测的地震数据
# 3. 几口井位置的准确测井数据
print("�️  注意：实际工程中不存在完整的真实阻抗数据")
print("💾 Memory usage: {get_memory_usage():.1f} MB")

# 实际工程中的测井插值阻抗数据（模拟通过少数井插值得到的不完整阻抗）
# 在实际应用中，这应该来自测井数据的插值结果，而不是完整的真实数据
print("🔄 Loading well-interpolated impedance model...")
# 注意：在实际工程中，这应该是通过测井数据插值生成的，这里为了演示使用同一数据
# 在实际应用中应该替换为: segy = _read_segy("path_to_well_interpolated_impedance.sgy")
# 注意：在实际工程中，这应该是通过测井数据插值生成的，这里为了演示使用同一数据
# 在实际应用中应该替换为: segy = _read_segy("path_to_well_interpolated_impedance.sgy")
segy=_read_segy("../data/yyf_smo_train_Volume_PP_IMP.sgy")
# segy = _read_segy("/home/shendi_chl/BGP/seismic_Impedance_inversion_2D/datasets/intial_imp_m2_fortrain_7wells1.sgy") #field data
impedance_model_log = []
for i in range(0,len(segy.traces)):
    impedance_model_log.append(segy.traces[i].data)

impedance_model_log = np.array(impedance_model_log).reshape(251,len(impedance_model_log)//251,601).transpose(2,1,0)

# 根据设备配置调整数据大小
if not USE_FULL_DATA:
    impedance_model_log = impedance_model_log[:, :MAX_SPATIAL_SLICES, :]

impedance_model_log = np.log(impedance_model_log)
print(f"✅ Well-interpolated impedance loaded: {impedance_model_log.shape}")

# 从测井插值阻抗中提取低频背景模型（这是实际工程中的做法）
print("🌊 Generating low-frequency background from well-interpolated impedance...")
mback = []
for i in range(impedance_model_log.shape[2]):
    B, A = signal.butter(2, 0.012, 'low') # 2*cuttoff_fre/fs  低通滤波获取低频数据
    m_loww = signal.filtfilt(B, A, impedance_model_log[...,i].T).T
    nsmooth = 3
    m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_loww) #低通滤波后在时间切片上有高频噪音，可以稍微平滑下
    nsmooth = 3
    m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_low.T).T #低通滤波后在时间切片上有高频噪音，可以稍微平滑下
    mback.append(m_low[...,None])
mback =  np.concatenate(mback, axis = 2)

#synthetic data from ref  地震数据的合成,为了偷懒，同样用了pylop中现成的函数
print("⚡ Generating synthetic seismic data from well-interpolated impedance...")
dims = impedance_model_log.shape
PPop = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=dims[0], spatdims=dims[1:3])

# 实际工程中应该直接使用观测的地震数据，这里从测井插值阻抗合成地震数据作为演示
syn1 = PPop*impedance_model_log.flatten()
syn1 =  syn1.reshape(impedance_model_log.shape)  #从测井插值阻抗合成的地震数据（模拟观测数据）

# 从测井插值阻抗合成的地震数据，用于有监督训练约束
synlog = PPop*impedance_model_log.flatten()    
synlog =  synlog.reshape(impedance_model_log.shape)
print(f"✅ Synthetic data generated: {syn1.shape}")

# 实际工程中的数据流程应该是：
# 观测地震数据 -> 阻抗反演 -> 获得高分辨率阻抗
# 而不是：阻抗模型 -> 合成地震数据 -> 阻抗反演

#可以在数据中加入一定的噪音干扰，
# np.random.seed(42)
# syn1 = syn1 + np.random.normal(0, 2e-2, dims)
# # calculate the SNR
# SNR = 10*np.log10(np.linalg.norm(syn1)**2/(np.linalg.norm(np.random.normal(0, 2e-2, dims))**2)) # about 10dB
# print(SNR)

####################################### 井位数据定义 ####################################################################
nx, ny = syn1.shape[1:3]
basex=450
basey=212

# 已知测井位置的绝对坐标（这些位置的波阻抗数据是准确的，来自实际测井）
pos=[[594,295],[572,692],[591,996],[532,1053],[603,1212],[561,842],[504,846],[499,597]]
# 转换为相对网格坐标
well_pos=[[y-basey,x-basex] for [x,y] in pos ]
# well_pos=[[x-basex,y-basey] for [x,y] in pos ]

# 注释掉的其他井位配置选项
# well_pos=[[594-basex,]]
# well_pos = [[594-basex,84], [572-212,69], [81,144], [49,79], [109,144], [109,109], [29,29]]  #假设已知井的空间位置
train_well = well_pos

print(f"📍 井位信息:")
print(f"  - 测井位置数量: {len(well_pos)}")
print(f"  - 井位坐标 (网格): {well_pos}")
print(f"  - 这些位置的阻抗数据是准确的（来自实际测井）")
print(f"  - 其他位置的阻抗数据通过插值获得（不够准确）")

# plt.figure()
# plt.imshow(impedance_model[102,...].T,cmap='jet',)
# plt.scatter(np.array(train_well)[:,0],np.array(train_well)[:,1],c='b');
# plt.xlabel('Crossline', fontdict={'size': 12})
# plt.ylabel('Inline', fontdict={'size': 12})
# plt.xlim(1, nx)
# plt.ylim(1, ny)
# plt.show()

# 生成井位掩码：标记哪些位置有准确的测井数据
grid_shape = syn1.shape[1:3]
vWellMask = generate_well_mask2(well_pos, grid_shape, well_range=15, sigma=5)
# vCrd, vMask = get_wellline_and_mask2(well_pos, grid_shape, vWellMask)

print(f"🎯 井位掩码生成完成:")
print(f"  - 网格形状: {grid_shape}")
print(f"  - 井位影响范围: 15个网格点")
print(f"  - 掩码形状: {vWellMask.shape}")
print(f"  - 掩码用于标识哪些位置有准确的测井约束")


###################################### 数据获取流程：三步骤生成训练数据集 ####################################
"""
数据获取的三个步骤：
1. 随机路径生成：使用随机路径获取构成伪二维剖面的井口坐标，每个二维剖面高为601，长不定
2. 数据提取：利用获得的井口坐标获得等大小的低频、插值阻抗、地震数据集
3. 统一裁剪：再次裁剪成601×70大小，使得数据集统一大小
"""

# 步骤1: 随机路径生成连井剖面
print(f"🔗 步骤1: 生成{N_WELL_PROFILES}条随机连井剖面...")
print(f"  - 基于{len(well_pos)}口井的位置生成随机连接路径")
print(f"  - 每条剖面垂直高度: 601个采样点")
print(f"  - 水平长度: 不定（根据井间路径决定）")

train_well1 = add_labels(train_well)
extension_length = 10  # 延长部分的长度

# 存储各类数据的列表
implow_train = []    # 低频背景模型数据
implog_train = []    # 测井插值阻抗数据
syn_train = []       # 合成地震数据
synlog_train = []    # 从插值阻抗合成的地震数据
Masks = []           # 井位掩码
path_tem=[]

for i in range(N_WELL_PROFILES):
    # 生成第i条随机连井剖面的坐标点
    interpolated_points, vMask = get_wellline_and_mask2(well_pos, grid_shape, vWellMask)
    
    print(f"  剖面{i+1}: 包含{len(interpolated_points)}个空间点")
    
    # 扩展掩码到完整的时间-空间维度 (601×空间点数)
    vMask = np.tile(vMask, (601,1))
    Masks.append(vMask)
    
    # 步骤2: 根据连井剖面坐标提取对应的数据
    # 提取低频背景模型沿着连井剖面的数据 (601×剖面长度)
    implow_train.append(mback[:,interpolated_points[:,0], interpolated_points[:,1]])
    
    # 提取合成地震数据沿着连井剖面的数据（实际中应为观测地震数据）
    syn_train.append(syn1[:,interpolated_points[:,0], interpolated_points[:,1]])
    
    # 提取从插值阻抗合成的地震数据（用于有监督约束）
    synlog_train.append(synlog[:,interpolated_points[:,0], interpolated_points[:,1]])
    
    # 提取测井插值阻抗沿着连井剖面的数据（实际工程的起点数据）
    implog_train.append(impedance_model_log[:,interpolated_points[:,0], interpolated_points[:,1]])

print(f"✅ 步骤1&2完成: 生成了{N_WELL_PROFILES}条伪二维剖面，每条高度601点")
print(f"  - 每条剖面包含: 低频背景、插值阻抗、地震数据、井位掩码")
print(f"  - ⚠️  真实阻抗仅用于性能评估，不参与训练损失计算！")

# 步骤3: 数据增强 - 将不定长的连井剖面裁剪成统一大小的训练块
print(f"📦 步骤3: 数据增强 - 裁剪成统一大小 {PATCH_SIZE}×{PATCH_SIZE}...")
print(f"  - 输入: {N_WELL_PROFILES}条不定长连井剖面 (601×变长)")
print(f"  - 输出: 统一大小的训练块 (601×{PATCH_SIZE})")

patchsize = PATCH_SIZE
oversize = 5
print(f"  - 重叠步长: {oversize}个点，用于数据增强")

# 存储裁剪后的训练数据
implow_train_set = []     # 低频背景训练块
implog_train_set = []     # 测井插值阻抗训练块
syn_train_set = []        # 地震数据训练块
synlog_train_set = []     # 插值阻抗合成地震训练块
Masks_set = []            # 井位掩码训练块

for i in range(N_WELL_PROFILES):
    # 使用image2cols函数将每条连井剖面切分成多个(601, patchsize)的小块
    # (syn1.shape[0], patchsize) = (601, PATCH_SIZE) 指定切分块的大小
    # (1, oversize) = (1, 5) 指定重叠步长
    
    implow_train_set.append(torch.tensor(image2cols(implow_train[i],(syn1.shape[0],patchsize),(1,oversize))))
    syn_train_set.append(torch.tensor(image2cols(syn_train[i],(syn1.shape[0],patchsize),(1,oversize))))
    synlog_train_set.append(torch.tensor(image2cols(synlog_train[i],(syn1.shape[0],patchsize),(1,oversize))))
    Masks_set.append(torch.tensor(image2cols(Masks[i],(syn1.shape[0],patchsize),(1,oversize))))
    implog_train_set.append(torch.tensor(image2cols(implog_train[i],(syn1.shape[0],patchsize),(1,oversize))))

# 将所有训练块拼接成最终的训练数据集
# [batch, channel, height, width] = [N_samples, 1, 601, PATCH_SIZE]
implow_train_set = torch.cat(implow_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
implog_train_set = torch.cat(implog_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
syn_train_set = torch.cat(syn_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
synlog_train_set = torch.cat(synlog_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
Masks_set = torch.cat(Masks_set,0)[...,None].permute(0,3,1,2).type(dtype)

print(f"✅ 步骤3完成: 生成统一训练数据集")
print(f"  - 最终训练样本数: {len(syn_train_set)}")
print(f"  - 每个样本大小: {syn_train_set.shape[2]}×{syn_train_set.shape[3]} (时间×空间)")
print(f"  - 数据类型: 低频背景、插值阻抗、地震数据、井位掩码")

#下面是对训练数据集进行归一化
logimpmax = impedance_model_log.max()
logimpmin = impedance_model_log.min()
logimp_set1 = (implog_train_set - logimpmin)/(logimpmax - logimpmin)
syn1_set = 2*(syn_train_set - syn_train_set.min())/(syn_train_set.max() - syn_train_set.min())-1
synlog_set = 2*(synlog_train_set - synlog_train_set.min())/(synlog_train_set.max() - synlog_train_set.min())-1
mback_set = (implow_train_set  - logimpmin)/(logimpmax - logimpmin)

# 应用训练样本限制（仅限CPU模式）
if MAX_TRAIN_SAMPLES is not None:
    print(f"🔄 Limiting training samples to {MAX_TRAIN_SAMPLES} for CPU optimization...")
    total_samples = len(syn1_set)
    if total_samples > MAX_TRAIN_SAMPLES:
        # 随机选择样本
        indices = torch.randperm(total_samples)[:MAX_TRAIN_SAMPLES]
        syn1_set = syn1_set[indices]
        logimp_set1 = logimp_set1[indices]
        mback_set = mback_set[indices]
        Masks_set = Masks_set[indices]
        synlog_set = synlog_set[indices]
        print(f"📊 Reduced training samples from {total_samples} to {len(syn1_set)}")

#下面是对测试数据集的归一化
print("🔧 Normalizing test data...")
syn1_nor =  2*(syn1 -syn1.min())/(syn1.max()-syn1.min())-1
implow_nor = (mback - logimpmin)/(logimpmax - logimpmin)
implog_nor =  (impedance_model_log-logimpmin)/(logimpmax - logimpmin)

test_data = torch.tensor(syn1_nor[...,None]).permute(2,3,0,1).type(dtype)
test_low = torch.tensor(implow_nor[...,None]).permute(2,3,0,1).type(dtype)
test_implog = torch.tensor(implog_nor[...,None]).permute(2,3,0,1).type(dtype)
test_low = torch.tensor(implow_nor[...,None]).permute(2,3,0,1).type(dtype)
test_imp = torch.tensor(imp_nor[...,None]).permute(2,3,0,1).type(dtype)
test_implog = torch.tensor(implog_nor[...,None]).permute(2,3,0,1).type(dtype)

##训练数据集与测试数据集的 集成
print(f"📦 Creating data loaders (batch size: {BATCH_SIZE})...")
batch_size = BATCH_SIZE  # 使用配置的批大小
Train_loader = data.DataLoader(data.TensorDataset(syn1_set,logimp_set1,logimp_set1,mback_set,Masks_set), batch_size=batch_size, shuffle=True)
Train_loader_sup = data.DataLoader(data.TensorDataset(synlog_set,logimp_set1,mback_set), batch_size=batch_size, shuffle=True)
Test_loader = data.DataLoader(data.TensorDataset(test_data, test_implog, test_implog, test_low), batch_size=batch_size, shuffle=False, drop_last=False)

##为了保证子波模块的加快收敛加入的初始子波，进而生成子波卷积矩阵，初始子波可以从外面输入
wav0 = wavelet_init(syn1_set.cpu().type(torch.float32), 101).squeeze().numpy()
size = syn1.shape[0]
W = pylops.utils.signalprocessing.convmtx(wav0, size, len(wav0) // 2)[:size]
W = torch.tensor(W, dtype=torch.float32, device=device)

##
N = len(wav0)  # 窗的长度
fp=30
fs = 1000
std = int((fs/fp)/2)  # 标准差，决定窗的宽度
gaussian_window = gaussian(N, std)
gaus_win = torch.tensor(gaussian_window[None,None,:,None]).type(dtype)

# pdb.set_trace()

#######################################################################################################################
##阻抗正演过程的差分算子
def DIFFZ(z):
    DZ= torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]], device=device).type(dtype)
    DZ[...,:-1,:] = 0.5*(z[...,1:, :] - z[..., :-1, :])
    return DZ

#阻抗到地震数据的正演过程，pytorch写法
class ImpedanceOperator():
    def __init__(self, wav):
        self.wav = wav
    def DIFFZ(self, z): # nonlinear operator
        nz = z.shape[2]
        S= torch.diag(0.5 * torch.ones(nz-1), diagonal=1) - torch.diag(
                    0.5 * torch.ones(nz-1), diagonal=-1)
        S[0] = S[-1] = 0
        DZ = torch.matmul(S.to(device), z)
        return DZ

    def forward(self, z):
        WEIGHT = torch.tensor(self.wav.reshape(1, 1, self.wav.shape[0], 1), device=device).type(dtype)
        For_syn = F.conv2d(self.DIFFZ(z), WEIGHT, stride=1, padding='same')
        return For_syn

def tv_loss(x, alfa): # TV约束，可以在噪音干扰情况下改善反演结果
    """
    Isotropic TV loss similar to the one in (cf. [1])
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return alfa*torch.mean(2*dh[..., :-1, :] + dw[..., :, :-1]) #空间方程乘以2，是为了使空间平滑，根据测试情况，可以删除

#网络的定义
def get_network_and_input(input_depth=2, n_channels = 1):  # 'meshgrid'
    """ Getting the relevant network and network input (based on the image shape and input depth)
    """
    # net = inverse_model().type(dtype) 
    net = UNet(input_depth, n_channels, channels=(8, 16, 32, 64),  skip_channels=(0, 8, 16, 32),use_sigmoid=True, use_norm=False).to(device)
    return net




#网络的训练
def train(net, forward_net, clean_img=True, save_path="", admm_iter=100, admm_iter1=50, LR=0.0005, mu=0.001, yita=10, beta=0 ):

    # get optimizer and loss function:
    mse = torch.nn.MSELoss().type(dtype)  # using MSE loss
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # using ADAM opt
    optimizerF = torch.optim.Adam(forward_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=.9, step_size=1000)

    net.train()
    psnr_net_total = []
    total_lossF = []
    epsI = 0.1 #
    for i in range(admm_iter): #子波模块的训练部分，利用测井数据的合成数据与地震数据的匹配损失函数来进行网络更新
        print(i)
        for y,Cimp1,Cimp1_dup,impback,index in Train_loader: # Cimp1对应的测井插值生成的阻抗
            optimizerF.zero_grad()
            lossF =  mse(index * forward_net(DIFFZ(Cimp1), torch.tensor(wav0[None, None, :, None], device=device))[0], index * y)*y.shape[3]
            lossF.backward()
            optimizerF.step()
            total_lossF.append(lossF.detach().cpu().numpy())


    wav00  = forward_net(DIFFZ(Cimp1), torch.tensor(wav0[None, None, :, None], device=device))[1].detach().cpu().squeeze().numpy()
    N = len(wav00)  # 窗的长度
    std = 25  # 标准差，决定窗的宽度
    gaussian_window = gaussian(N, std)
    wav00 = gaussian_window*(wav00-wav00.mean()) #对子波进行一个窗口平滑，可以使子波估计结果更稳健，因为上述的子波模块会出现边界不平滑，可能还得仔细调调网络和损失函数
    #用新的子波生成卷积矩阵，构建正演算子
    nz = y.shape[2]
    S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(
        0.5 * torch.ones(nz - 1), diagonal=-1)
    S[0] = S[-1] = 0
    WW = pylops.utils.signalprocessing.convmtx(wav00/wav00.max(), size, len(wav00) // 2)[:size]
    WW = torch.tensor(WW, dtype=torch.float32, device=device)
    WW = WW@S.to(device)
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device)
    for i in range(admm_iter1): #unet模块的训练
        labeled_iter = iter(Train_loader_sup)
        for y, Cimp1, Cimp1_dup, impback, index in Train_loader:
            optimizer.zero_grad()

            datarn = torch.matmul(WW.T, y-torch.matmul(WW, impback))
            x, _, _, _ = torch.linalg.lstsq(PP[None,None],datarn)
            x = x + impback  #最小二乘解
            x = (x-x.min())/(x.max()-x.min())  #进行了一步归一化，避免最小二乘大小带来的影响
            # x = Cimp1.max()*x/x.max()

            out = net(torch.cat([x, y], dim=1)) + x  #网络的输出
            out_np = out.detach().cpu()

            if beta!=0: # 这部分代码可以加入有监督学习部分，但是如果想要充分利用有监督学习部分，不应该利用测井曲线插值生成简单的数据集，因为其中没有引入额外信息
                #loading unlabeled data
                try:
                    y_sup, imp_sup, mback_sup = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(Train_loader_sup)
                    y_sup, imp_sup, mback_sup = next(labeled_iter)

                datarn = torch.matmul(WW.T, y_sup - torch.matmul(WW, mback_sup))
                x, _, _, _ = torch.linalg.lstsq(PP[None, None], datarn)
                x = x + mback_sup
                x = (x - x.min()) / (x.max() - x.min())

                out_sup = net(torch.cat([x, y_sup], dim=1)) + x
                total_loss_sup = mse(out_sup,imp_sup)
            else:
                total_loss_sup=0

            #半监督学习损失函数定义
            # index*out: 只在测井位置处计算损失，index是掩码，标记哪些位置有准确的测井数据
            # Cimp1: 测井插值阻抗（在测井位置处是准确的）
            loss_sup = yita*mse(index*out, index*Cimp1)*Cimp1.shape[3]/3  # 测井位置约束损失
            loss_unsup = mse(forward_net(DIFFZ(out), torch.tensor(wav0[None, None, :, None], device=device))[0], y)  # 地震数据拟合损失

            total_loss = ( loss_unsup +  tv_loss(out, mu) + loss_sup ) + beta*total_loss_sup

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)  
            optimizer.step()  
            scheduler.step()
            
            #一些参数的输出，为了监测网络的训练
            if clean_img is not False:
                  # 注意：实际工程中没有完整的真实阻抗用于性能评估
                  # 只能监控损失函数的变化
                  print('\r',  '%04d/%04d Loss %f Loss_sup %f Loss_unsup %f total_loss_sup %f' % (i, admm_iter, total_loss.item(), loss_sup, loss_unsup, total_loss_sup), end='')        
            else:
                  print('\r', 'iteration %04d/%04d Loss %f' % (i, admm_iter, total_loss.item()), end='')
            
 
        torch.save(net.state_dict(),save_path+f"_{i:05d}")  #网络权重的保存
    return wav00, out_np



def test(net, forward_net,save_path="", beta=0.05, mu=0.1):  # LR_x needed only if method!=fixed_point
 
    ##load the net
    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()   
    
    # get optimizer and loss function:
    mse = torch.nn.MSELoss().type(dtype)  # using MSE loss
    
    predicted_impedance = []
    true_impedance = []
    true_mback = []
    test_loss = []

    wav00 = forward_net(torch.tensor(wav0[None, None, :, None], device=device), torch.tensor(wav0[None, None, :, None], device=device))[1].detach().cpu().squeeze().numpy()
    N = len(wav00)  # 窗的长度
    std = 25  # 标准差，决定窗的宽度
    gaussian_window = gaussian(N, std)
    wav00 = gaussian_window * (wav00 - wav00.mean())
    epsI = 0.1 # depends on the noise level

    nz = size
    S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(
        0.5 * torch.ones(nz - 1), diagonal=-1)
    S[0] = S[-1] = 0
    WW = pylops.utils.signalprocessing.convmtx(wav00/wav00.max(), size, len(wav00) // 2)[:size]
    WW = torch.tensor(WW, dtype=torch.float32, device=device)
    WW = WW@S.to(device)
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device)
    with torch.no_grad():    
        for y,Cimp1,Cimp1_dup,impback in Test_loader:

            datarn = torch.matmul(WW.T, y-torch.matmul(WW, impback))
            x, _, _, _ = torch.linalg.lstsq(PP[None,None],datarn)
            x = x + impback
            x = (x-x.min())/(x.max()-x.min())
            # x = Cimp.max()*x/x.max()

            out = net(torch.cat([x, y], dim=1)) + x
            out_np = out
            
            total_loss = mse(forward_net(DIFFZ(out), torch.tensor(wav0[None, None, :, None], device=device))[0], y)
            test_loss.append(total_loss.item())
            
            # 实际工程中只有插值阻抗，没有真实阻抗
            true_mback.append(impback)
            predicted_impedance.append(out_np)
            
    predicted_impedance = torch.cat(predicted_impedance, dim=0).detach().cpu().numpy()         
    true_mback = torch.cat(true_mback, dim=0).detach().cpu().numpy()
    return true_mback, predicted_impedance, test_loss



## 
print("🚀 Initializing networks...")
net= get_network_and_input()
forward_net = forward_model(nonlinearity="tanh").to(device)
Forward_ope = ImpedanceOperator(wav[::-1].copy())

if Train:
    print(f"🎯 Starting training with {ADMM_ITER} + {ADMM_ITER1} iterations...")
    INV_wavelet, INV_imp, PSNRT  = train(net, forward_net, 
                                        save_path='Uet_TV_IMP_7labels_channel3.pth',
                                        admm_iter=ADMM_ITER, 
                                        admm_iter1=ADMM_ITER1)
    print("✅ Training completed!")
else:
    print("🔍 Running inference...")
    name=49
    Ture_imp, True_mimp, INV_imp, loss11  = test(net, forward_net, save_path=f'Uet_TV_IMP_7labels_channel3.pth_{name:05d}')
    

    well_pos2 = np.array(well_pos)
    corr = np.corrcoef( INV_imp[:, 0, well_pos2[:, 1], well_pos2[:, 0]].flatten(),  Ture_imp[:, 0, well_pos2[:, 1], well_pos2[:, 0]].flatten())[0,1]
    print(f"Well correlation: {corr:.4f}")
    tools.single_imshow(INV_imp[150,0], vmin=0.2, vmax=1, cmap='jet', title=f'INV_impedance_{name}')
    tools.single_imshow(Ture_imp[150,0], vmin=0.2, vmax=1, cmap='jet', title=f'Ture_imp_{name}')
    tools.single_imshow(True_mimp[150,0], vmin=0.2, vmax=1, cmap='jet', title=f'dipin_{name}')

    # plt.figure();plt.imshow(INV_imp[150,0],vmin=0.2, vmax=1, cmap='jet');
    # plt.xlabel('Trace No.', fontdict={ 'size':12})
    # plt.ylabel('Time (ms)', fontdict={'size':12})
    # plt.show()
    # plt.figure();plt.imshow(Ture_imp[150,0],vmin=0.2, vmax=1, cmap='jet');
    # plt.xlabel('Trace No.', fontdict={ 'size':12})
    # plt.ylabel('Time (ms)', fontdict={'size':12})
    # plt.show()
    # plt.figure();plt.imshow(True_mimp[150,0], vmin=0.2, vmax=1, cmap='jet');
    # plt.xlabel('Trace No.', fontdict={ 'size':12})
    # plt.ylabel('Time (ms)', fontdict={'size':12})
    # plt.show()
    # plt.figure();plt.imshow(syn1, cmap='seismic');plt.show()
    plt.figure()
    plt.plot(INV_imp[0,0,:,31],label=f'inv_imp_{name}')
    plt.plot(Ture_imp[0,0,:,31],label=f'true_imp_{name}')
    plt.plot(True_mimp[0,0,:,31],label=f'dipin_{name}')
    plt.legend()
    plt.show()


    k = 120
    plt.figure();plt.imshow(INV_imp[:,0,k,:],vmin=0.2, vmax=1,cmap='jet', interpolation='bicubic');
    plt.xlabel('Crossline', fontdict={ 'size':12})
    plt.ylabel('Inline', fontdict={'size':12})
    plt.title(f"inv_imp_{name}")
    plt.show()
    plt.figure();plt.imshow(Ture_imp[:,0,k,:],vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Crossline', fontdict={ 'size':12})
    plt.ylabel('Inline', fontdict={'size':12})
    plt.title(f"ture_imp_{name}");plt.show()
    plt.figure();plt.imshow(True_mimp[:,0,k,:], vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Crossline', fontdict={ 'size':12})
    plt.ylabel('Inline', fontdict={'size':12})
    plt.title(f"dipin_{name}")
    plt.show()

    # plt.figure();plt.imshow(syn1[k,:,:].T, cmap='seismic'); plt.show()
    plt.figure();plt.imshow(implog_nor[k,:,:].T, cmap='jet')
    plt.title(f"true_imp_{name}");
    plt.show()






