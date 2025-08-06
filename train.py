"""
简化版地震阻抗反演主程序
使用独立的数据处理模块和缓存机制
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用物理第1张卡（第二张卡）
import torch
print(f"可见的GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"当前GPU索引: {torch.cuda.current_device()}")
else:
    print("CUDA不可用，将使用CPU")



import sys
import torch.optim
from Model.net2D import UNet, forward_model
from Model.utils import DIFFZ, tv_loss, wavelet_init, save_stage1_loss_data, save_stage2_loss_data, save_complete_training_loss
import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.utils.wavelets import ricker
from scipy.signal import filtfilt
from scipy import signal
import torch.nn.functional as F
from scipy.signal.windows import gaussian
import pdb
import sys
sys.path.append('..')
from icecream import ic 
sys.path.append('../codes')
sys.path.append('deep_learning_impedance_inversion_chl')
import psutil
import gc
from tqdm import tqdm
import json
from pathlib import Path
# 导入数据处理模块
from data_processor import SeismicDataProcessor
import run_test


PROJECT_DIR = Path(__file__).parent.resolve()
print(f"项目目录: {PROJECT_DIR}")    ##使用多线程，不用绝对目录会乱





##如果传入配置文件，则从配置文件中读取config
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"🚀 Using config: {config_file}")

    ##取config的名字为文件夹名
    save_dir = os.path.join(PROJECT_DIR,f'logs/'+config_file.split('/')[-1].split('.')[0]+'/')
else:
    ##如果没传入配置文件，则使用默认配置    
    config={
        'lr1': 1e-4,   ## 学习率
        'lr2': 1e-4,   ## Forward网络学习率
        'sup_coeff': 1,   
        'tv_coeff': 1,
        'unsup_coeff':1.0,
        'stage1_epoch_number': 3,
        'stage2_epoch_number': 4,
        'device': 'cuda:0',
        'inference_device': 'cuda:1',
        # 模型结构参数
        'unet_in_channels': 2,
        'unet_out_channels': 1,
        'unet_channels': [8, 16, 32, 64],
        'unet_skip_channels': [0, 8, 16, 32],
        'unet_use_sigmoid': True,
        'unet_use_norm': False,
        'forward_nonlinearity': "tanh",
        # 训练参数
        'scheduler_step_size': 30,
        'scheduler_gamma': 0.9,
        'max_grad_norm': 1.0,
        'wavelet_length': 101,
        'gaussian_std': 25,
        'epsI': 0.1,
        'tv_loss_weight': 1.0,
        'sup_loss_divisor': 3,
        # 打印和保存间隔
        'stage1_print_interval': 20,
        'stage2_print_interval': 10,
        'stage2_loss_save_interval': 5,
        'stage2_complete_loss_save_interval': 5,
        # 文件路径和命名
        'forward_model_filename': 'forward_net_wavelet_learned.pth',
        'unet_model_prefix': 'Uet_TV_IMP_7labels_channel3',
        'cache_dir': 'cache',
    }
    from datetime import datetime
    save_dir = os.path.join(PROJECT_DIR,f'logs/'+datetime.now().strftime("%Y%m%d-%H-%M-%S")+'/')


device = torch.device(config['device'])
print(f"🚀 Using device: {device}")
dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

model_save_dir= os.path.join(save_dir, 'model')
os.makedirs(model_save_dir, exist_ok=True)
##把config存成config.json到save_dir
config = {key: value for key, value in config.items() if not isinstance(value, torch.device)}
with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


#############################################################################################################
### 第1部分：数据处理 - 使用独立的数据处理模块
#############################################################################################################

print("\n" + "="*80) 
print("📂 第1部分：数据处理")
print("="*80)

# 一键处理所有数据
from data_processor import SeismicDataProcessor

processor = SeismicDataProcessor(cache_dir=config['cache_dir'], device=config['device'])

data_info = None  # 先初始化
train_loader, norm_params, data_info = processor.process_train_data()

# 提取归一化参数
logimpmax = norm_params['logimpmax']
logimpmin = norm_params['logimpmin']

print(f"📊 数据处理完成:")
print(f"   - 阻抗数据形状: {data_info['impedance_shape']}")
print(f"   - 地震数据形状: {data_info['seismic_shape']}")
print(f"   - 井位数量: {len(data_info['well_positions'])}")



#############################################################################################################
### 第3部分：网络初始化
#############################################################################################################

print("\n" + "="*80)
print("🤖 第3部分：网络初始化")
print("="*80)

# UNet阻抗反演网络
print("🏗️  初始化UNet反演网络...")
net = UNet(
    in_ch=config['unet_in_channels'],                 # 输入通道：[最小二乘初始解, 观测地震数据]
    out_ch=config['unet_out_channels'],                # 输出通道：阻抗残差
    channels=config['unet_channels'],
    skip_channels=config['unet_skip_channels'],
    use_sigmoid=config['unet_use_sigmoid'],        # 输出归一化到[0,1]
    use_norm=config['unet_use_norm']
).to(device)

# Forward建模网络（子波学习）
print("⚡ 初始化Forward建模网络...")
forward_net = forward_model(nonlinearity=config['forward_nonlinearity']).to(device)

print(f"✅ 网络初始化完成:")
print(f"   - UNet参数量: {sum(p.numel() for p in net.parameters()):,}")
print(f"   - Forward网络参数量: {sum(p.numel() for p in forward_net.parameters()):,}")
print(f"   - 设备: {device}")

#############################################################################################################
### 第4部分：训练算法 - 两阶段训练
#############################################################################################################

print("\n" + "="*80)
print("🚀 第4部分：两阶段训练算法")
print("="*80)


size = data_info['seismic_shape'][0]  # 从数据中动态获取，不能固定
##按照时间年月日时分秒命名文件夹


# 优化器
optimizerF = torch.optim.Adam(forward_net.parameters(), lr=config['lr1'])   ##子波矫正器的优化器
optimizer = torch.optim.Adam(net.parameters(), lr=config['lr2'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

# 损失函数
mse = torch.nn.MSELoss()

#########################################################################################################
### 阶段1：子波矫正器（ForwardNet）学习最优子波
#########################################################################################################

print("\n" + "-"*60)
print("🌊 阶段1：子波矫正器(ForwardNet)学习最优子波")
print("-"*60)
print("目标：利用井位高可信度区域，通过ForwardNet(子波矫正器)自适应调整初始子波，使其更贴合实际地震响应")
print("损失：L_wavelet = ||M ⊙ [ForwardNet(∇Z_full, w_0)]_synth - M ⊙ S_obs||²")

psnr_net_total = []
total_lossF = []
epsI = config['epsI']
wav0 = wavelet_init(config['wavelet_length']).squeeze().numpy()
print("开始子波矫正器训练...")
for i in range(config['stage1_epoch_number']):
    epoch_loss = 0
    batch_count = 0
    
    for S_obs_batch, Z_full_batch, Z_back_batch, M_mask_batch in train_loader:
        optimizerF.zero_grad()
        # 计算完整阻抗的反射系数
        reflection_coeff = DIFFZ(Z_full_batch)
        # ForwardNet(子波矫正器)：输入反射系数和初始子波，输出矫正后子波并合成地震
        synthetic_seismic, learned_wavelet = forward_net(
            reflection_coeff, 
            torch.tensor(wav0[None, None, :, None], device=device)
        )
        # 损失：合成地震与观测地震的掩码加权MSE
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
    if i % config['stage1_print_interval'] == 0:
        print(f"   Epoch {i:04d}/{config['stage1_epoch_number']:04d}, 子波矫正损失: {avg_loss:.6f}")
        print(f"      说明：损失越小，ForwardNet输出的矫正子波在高可信度区域拟合观测数据越好")

# 保存阶段1的loss数据
save_stage1_loss_data(save_dir, total_lossF)
# 提取矫正后的子波
print("🎯 提取ForwardNet矫正后的子波...")
with torch.no_grad():
    _, wav_learned = forward_net(
        DIFFZ(Z_full_batch), 
        torch.tensor(wav0[None, None, :, None], device=device)
    )
    wav_learned_np = wav_learned.detach().cpu().squeeze().numpy()
# 子波后处理（高斯窗平滑）
N = len(wav_learned_np) # 窗的长度
std = config['gaussian_std']  # 标准差，决定窗的宽度
gaussian_window = gaussian(N, std)
#对子波进行一个窗口平滑，可以使子波估计结果更稳健，因为上述的子波模块会出现边界不平滑，可能还得仔细调调网络和损失函数
wav_learned_smooth = gaussian_window * (wav_learned_np - wav_learned_np.mean())
print(f"✅ 阶段1完成：ForwardNet矫正子波学习")
print(f"   - 训练轮次: {config['stage1_epoch_number']}")
print(f"   - 最终损失: {total_lossF[-1]:.6f}")
print(f"   - 矫正后子波长度: {len(wav_learned_smooth)}")
#########################################################################################################
### 阶段2：UNet阻抗反演
#########################################################################################################
print("\n" + "-"*60)
print("🎯 阶段2：UNet阻抗反演")
print("-"*60)
print("目标：使用ForwardNet矫正后的子波进行高精度阻抗反演")
print("策略：最小二乘初始化 + UNet残差学习")
print("损失：L_total = L_unsup + L_sup + L_tv")
# 构建基于矫正子波的卷积算子
print("🔧 构建ForwardNet矫正后的子波的卷积算子...")
nz = S_obs_batch.shape[2]
S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(0.5 * torch.ones(nz - 1), diagonal=-1)
S[0] = S[-1] = 0
WW = pylops.utils.signalprocessing.convmtx(wav_learned_smooth/wav_learned_smooth.max(), size, len(wav_learned_smooth) // 2)[:size]
WW = torch.tensor(WW, dtype=torch.float32, device=device)
WW = WW @ S.to(device)
PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device) ##最小二乘解的Toplitz矩阵的装置
pdb.set_trace()
print(f"✅ 阶段2完成：UNet阻抗反演训练")
# 保存Forward网络（子波矫正器）
forward_save_path= os.path.join(model_save_dir, config['forward_model_filename'])
torch.save(forward_net.state_dict(), forward_save_path)

# 初始化阶段2的loss记录列表
stage2_total_loss = []
stage2_sup_loss = []
stage2_unsup_loss = []
stage2_tv_loss = []

threads_inference=[]  # 用于存储推理线程

for i in range(config['stage2_epoch_number']):
    epoch_loss = 0
    epoch_loss_sup = 0
    epoch_loss_unsup = 0
    epoch_loss_tv = 0
    batch_count = 0
    for S_obs_batch, Z_full_batch, Z_back_batch, M_mask_batch in train_loader:
        optimizer.zero_grad()
        # 步骤1：最小二乘初始化
        datarn = torch.matmul(WW.T, S_obs_batch - torch.matmul(WW, Z_back_batch))
        x, _, _, _ = torch.linalg.lstsq(PP[None, None], datarn)
        Z_init = x + Z_back_batch  # 加回低频背景
        Z_init = (Z_init - Z_init.min()) / (Z_init.max() - Z_init.min())  # 归一化
        # 步骤2：UNet残差学习
        Z_pred = net(torch.cat([Z_init, S_obs_batch], dim=1)) + Z_init
        # 三项损失函数计算
        # 1. 井约束损失（差异化监督）
        loss_sup = config['sup_coeff'] * mse(
            M_mask_batch * Z_pred, 
            M_mask_batch * Z_full_batch
        ) * Z_full_batch.shape[3] / config['sup_loss_divisor']
        # 2. 物理约束损失（正演一致性）
        pred_reflection = DIFFZ(Z_pred)
        pred_seismic, _ = forward_net(
            pred_reflection, 
            torch.tensor(wav0[None, None, :, None], device=device)  # 始终用初始子波做正演一致性
        )
        loss_unsup =  config['unsup_coeff']* mse(pred_seismic, S_obs_batch)
        # 3. 总变分正则化损失（空间平滑性）
        loss_tv = config['tv_coeff']* tv_loss(Z_pred, config['tv_loss_weight'])
        # 总损失
        total_loss = loss_unsup + loss_tv + loss_sup
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        batch_size=S_obs_batch.shape[0]
        epoch_loss += total_loss.item()
        epoch_loss_sup += loss_sup.item()
        epoch_loss_unsup += loss_unsup.item()
        epoch_loss_tv += loss_tv.item()
        batch_count += batch_size
    
    # 记录每个epoch的平均损失
    avg_total = epoch_loss / batch_count
    avg_sup = epoch_loss_sup / batch_count
    avg_unsup = epoch_loss_unsup / batch_count
    avg_tv = epoch_loss_tv / batch_count
    
    stage2_total_loss.append(avg_total)
    stage2_sup_loss.append(avg_sup)
    stage2_unsup_loss.append(avg_unsup)
    stage2_tv_loss.append(avg_tv)
    
    if i % config['stage2_print_interval'] == 0:
        print(f"   Epoch {i:04d}/{config['stage2_epoch_number']:04d}")
        print(f"      总损失: {avg_total:.6f}")
        print(f"      井约束损失: {avg_sup:.6f} (高可信度区域匹配)")
        print(f"      物理约束损失: {avg_unsup:.6f} (正演一致性)")
        print(f"      TV正则化损失: {avg_tv:.6f} (空间平滑性)")
        model_save_path = os.path.join(model_save_dir, f'{config["unet_model_prefix"]}_epoch={i}.pth')
        torch.save(net.state_dict(), model_save_path)
        print(f"💾 UNet模型已保存: {model_save_path}")
        test_save_dir= os.path.join(save_dir, 'test', f'test_epoch={i}')
        thread=run_test.inference(model_path1=forward_save_path, model_path2=model_save_path, folder_dir=test_save_dir,inference_device=config['inference_device'], config=config)
        threads_inference.append(thread)
    
    if i % config['stage2_loss_save_interval'] == 0:
        # 保存阶段2的loss数据
        save_stage2_loss_data(save_dir, stage2_total_loss, stage2_sup_loss, 
                                stage2_unsup_loss, stage2_tv_loss)
    
    # 保存完整训练过程loss对比图
    if i % config['stage2_complete_loss_save_interval'] == 0:
        save_complete_training_loss(save_dir, total_lossF, stage2_total_loss, 
                                    stage2_sup_loss, stage2_unsup_loss, stage2_tv_loss, 
                                    )


for thread in threads_inference:
    thread.join()  # 等待所有推理线程完成

