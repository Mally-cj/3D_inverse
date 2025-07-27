"""
简化版地震阻抗反演主程序
使用独立的数据处理模块和缓存机制
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch.optim
from Model.net2D import UNet, forward_model
from Model.utils import DIFFZ, tv_loss, wavelet_init
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

# 导入数据处理模块
from data_processor import SeismicDataProcessor

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
    ADMM_ITER = 100
    ADMM_ITER1 = 50
else:
    print("💻 CPU mode: Using optimized subset")
    dtype = torch.FloatTensor
    # CPU优化参数配置
    ADMM_ITER = 30
    ADMM_ITER1 = 15

print(f"📋 Configuration:")
print(f"  - Training iterations: {ADMM_ITER} + {ADMM_ITER1}")

#############################################################################################################
### 第1部分：数据处理 - 使用独立的数据处理模块
#############################################################################################################

print("\n" + "="*80) 
print("📂 第1部分：数据处理")
print("="*80)

# 一键处理所有数据
from data_processor import SeismicDataProcessor

processor = SeismicDataProcessor(cache_dir='cache', device='auto')

data_info = None  # 先初始化
if Train:
    train_loader, norm_params, data_info = processor.process_train_data()
    print(f"   - 训练批数: {len(train_loader)}")
else:
    # 用patch loader和索引
    test_loader, indices, shape3d, norm_params = processor.process_test_data()
    # 手动补充data_info（用于后续shape等信息）
    impedance_model_full = processor.load_impedance_data()
    S_obs = processor.load_seismic_data()
    well_pos, _, _ = processor.generate_well_mask(S_obs)
    data_info = {
        'impedance_shape': impedance_model_full.shape,
        'seismic_shape': S_obs.shape,
        'well_positions': well_pos
    }
    print(f"   - 测试批数: {len(test_loader)}")

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
### 第4部分：训练算法 - 两阶段训练
#############################################################################################################

if Train:
    print("\n" + "="*80)
    print("🚀 第4部分：两阶段训练算法")
    print("="*80)
    
    # 训练参数
    lr = 1e-3
    yita = 1e-1    # 井约束损失权重
    mu = 5e-4      # TV正则化权重
    beta = 0       # 额外监督损失权重
    size = data_info['seismic_shape'][0]
    
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
    ### 阶段1：子波矫正器（ForwardNet）学习最优子波
    #########################################################################################################
    
    print("\n" + "-"*60)
    print("🌊 阶段1：子波矫正器(ForwardNet)学习最优子波")
    print("-"*60)
    print("目标：利用井位高可信度区域，通过ForwardNet(子波矫正器)自适应调整初始子波，使其更贴合实际地震响应")
    print("损失：L_wavelet = ||M ⊙ [ForwardNet(∇Z_full, w_0)]_synth - M ⊙ S_obs||²")
    
    admm_iter = ADMM_ITER
    psnr_net_total = []
    total_lossF = []
    epsI = 0.1
    wav0 = wavelet_init(101).squeeze().numpy()
    print("开始子波矫正器训练...")
    for i in range(admm_iter):
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
        if i % 20 == 0:
            print(f"   Epoch {i:04d}/{admm_iter:04d}, 子波矫正损失: {avg_loss:.6f}")
            print(f"      说明：损失越小，ForwardNet输出的矫正子波在高可信度区域拟合观测数据越好")
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
    std = 25  # 标准差，决定窗的宽度
    gaussian_window = gaussian(N, std)
    #对子波进行一个窗口平滑，可以使子波估计结果更稳健，因为上述的子波模块会出现边界不平滑，可能还得仔细调调网络和损失函数
    wav_learned_smooth = gaussian_window * (wav_learned_np - wav_learned_np.mean())
    print(f"✅ 阶段1完成：ForwardNet矫正子波学习")
    print(f"   - 训练轮次: {admm_iter}")
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
    admm_iter1 = ADMM_ITER1
    print(f"开始UNet反演训练 (共{admm_iter1}轮)...")
    for i in range(admm_iter1):
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
            loss_sup = yita * mse(
                M_mask_batch * Z_pred, 
                M_mask_batch * Z_full_batch
            ) * Z_full_batch.shape[3] / 3
            # 2. 物理约束损失（正演一致性）
            pred_reflection = DIFFZ(Z_pred)
            pred_seismic, _ = forward_net(
                pred_reflection, 
                torch.tensor(wav0[None, None, :, None], device=device)  # 始终用初始子波做正演一致性
            )
            loss_unsup = mse(pred_seismic, S_obs_batch)
            # 3. 总变分正则化损失（空间平滑性）
            loss_tv = tv_loss(Z_pred, mu)
            # 总损失
            total_loss = loss_unsup + loss_tv + loss_sup
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            epoch_loss += total_loss.item()
            epoch_loss_sup += loss_sup.item()
            epoch_loss_unsup += loss_unsup.item()
            epoch_loss_tv += loss_tv.item()
            batch_count += 1
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
    # 保存Forward网络（子波矫正器）
    forward_save_path = 'logs/model/forward_net_wavelet_learned.pth'
    torch.save(forward_net.state_dict(), forward_save_path)
    print(f"💾 ForwardNet(子波矫正器)已保存: {forward_save_path}")
    print(f"   说明：ForwardNet包含训练时学习的矫正子波参数")

#############################################################################################################
### 第5部分：测试和结果评估
#############################################################################################################
elif not Train:
    print("\n" + "="*80)
    print("🔍 第5部分：模型测试和结果评估")
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
    
    # 推理阶段：构建子波算子
    print("🔧 构建推理用子波算子...")
    wav0 = wavelet_init(101).squeeze().numpy()
    wav00=torch.tensor(wav0[None, None, :, None],device=device)
    size = data_info['seismic_shape'][0]
    nz = size
    epsI = 0.1
    S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(0.5 * torch.ones(nz - 1), diagonal=-1)
    S=S.to(device)
    S[0] = S[-1] = 0

    
    wav_learned_np= forward_net(wav00, wav00)[1].detach().cpu().squeeze().numpy()
    N = len(wav_learned_np)
    std = 25
    gaussian_window = gaussian(N, std)
    wav_final = gaussian_window * (wav_learned_np - wav_learned_np.mean())
    wav_final = wav_final / wav_final.max()
    WW = pylops.utils.signalprocessing.convmtx(wav_final, size, len(wav_final) // 2)[:size]
    WW = torch.tensor(WW,device=device)
    WW = WW.float()
    S = S.float()
    WW = WW @ S
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device)
    print(f"✅ 推理子波算子构建完成:")
    print(f"   - 子波长度: {len(wav_final)}")
    print(f"   - 卷积算子形状: {WW.shape}")
    print(f"   - 子波类型: {'学习的子波' if use_learned_wavelet else '初始子波'}")
    print("🔍 开始测试patch推理...")
    # 1. 获取patch loader、索引、shape、归一化参数
    test_loader, indices, shape3d, norm_params = processor.process_test_data()
    
    # 2. 推理循环
    pred_patch_list = []
    true_patch_list = []
    input_patch_list = []
    back_patch_list = []
    seismic_patch_list = []
    logimpmax = norm_params['logimpmax']
    logimpmin = norm_params['logimpmin']
    with torch.no_grad():
        for i, (s_patch, imp_patch, zback_patch) in enumerate(test_loader):
            s_patch = s_patch.to(device)
            imp_patch = imp_patch.to(device)
            zback_patch = zback_patch.to(device)
            # 最小二乘初始化
            datarn = torch.matmul(WW.T, s_patch - torch.matmul(WW, zback_patch))
            x, _, _, _ = torch.linalg.lstsq(PP[None, None], datarn)
            Z_init = x + zback_patch
            Z_init = (Z_init - Z_init.min()) / (Z_init.max() - Z_init.min())
            # 网络推理
            Z_pred = net(torch.cat([Z_init, s_patch], dim=1)) + Z_init
            # 收集patch结果（适配batch>1）
            Z_pred_np = Z_pred.cpu().numpy()  # [batch, 1, time, patch_size]
            imp_patch_np = imp_patch.cpu().numpy()
            zback_patch_np = zback_patch.cpu().numpy()
            s_patch_np = s_patch.cpu().numpy()
            # squeeze掉通道维（axis=1），遍历batch
            Z_pred_np = np.squeeze(Z_pred_np, axis=1)  # [batch, time, patch_size]
            imp_patch_np = np.squeeze(imp_patch_np, axis=1)
            zback_patch_np = np.squeeze(zback_patch_np, axis=1)
            s_patch_np = np.squeeze(s_patch_np, axis=1)
            for b in range(Z_pred_np.shape[0]):
                pred_patch_list.append(Z_pred_np[b])
                true_patch_list.append(imp_patch_np[b])
                back_patch_list.append(zback_patch_np[b])
                seismic_patch_list.append(s_patch_np[b])
            current_patches = len(pred_patch_list)
            print(f"   处理patch {current_patches}/{len(indices)}")
    
    # 3. 拼回3D体
    print(f"   拼回3D体...") # [N, time, patch_size]
    pred_3d = processor.reconstruct_3d_from_patches(pred_patch_list, indices)
    true_3d = processor.reconstruct_3d_from_patches(true_patch_list, indices)
    back_3d = processor.reconstruct_3d_from_patches(back_patch_list, indices)
    seismic_3d = processor.reconstruct_3d_from_patches(seismic_patch_list, indices)
    
    # 4. 反归一化
    # pred_3d_imp = np.exp(pred_3d * (logimpmax - logimpmin) + logimpmin)
    # true_3d_imp = np.exp(true_3d * (logimpmax - logimpmin) + logimpmin)
    # back_3d_imp = np.exp(back_3d * (logimpmax - logimpmin) + logimpmin)
    pred_3d_imp = pred_3d
    true_3d_imp = true_3d
    back_3d_imp = back_3d
    
    # 5. 保存所有可视化需要的文件
    os.makedirs('logs/results', exist_ok=True)
    # np.save('logs/results/prediction_sample.npy', np.array(pred_patch_list))
    # np.save('logs/results/true_sample.npy', np.array(true_patch_list))
    # np.save('logs/results/input_sample.npy', np.array(seismic_patch_list))
    np.save('logs/results/seismic_record.npy', seismic_3d)
    np.save('logs/results/prediction_impedance.npy', pred_3d_imp)
    np.save('logs/results/true_impedance.npy', true_3d_imp)
    np.save('logs/results/background_impedance.npy', back_3d_imp)
    print(f"   ✅ 推理数据已保存: logs/results/prediction_impedance.npy 等 shape: {pred_3d_imp.shape}")
    print("\n" + "="*80)
    print("🎉 程序执行完成")
    print("="*80) 