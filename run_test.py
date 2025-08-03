from Model.net2D import UNet, forward_model
import torch
from Model.utils import DIFFZ, tv_loss, wavelet_init, save_stage1_loss_data, save_stage2_loss_data, save_complete_training_loss
from scipy.signal.windows import gaussian
import pylops
from data_processor import SeismicDataProcessor
import numpy as np
import os
from visual_results import plot_sections_with_wells,plot_well_curves_seisvis
import threading
from functools import wraps
import pdb
from data_tools import run_in_thread

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

##准备数据
# processor = SeismicDataProcessor(cache_dir='cache', device=device)
# test_loader, xy, shape3d, norm_params = processor.process_test_data(batch_size=100,patch_size=800)
# number_of_patches = len(xy)
# print(f"number_of_patches: {number_of_patches}")

@run_in_thread
def inference(model_path1=None,model_path2=None,folder_dir='logs/test',inference_device="cpu", config=None):
    print('新开线程执行推理...')
    device = torch.device(inference_device)
    # inference_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    processor = SeismicDataProcessor(cache_dir='cache', device=inference_device)
    test_loader, xy, shape3d, norm_params = processor.process_test_data(batch_size=500,patch_size=120)
    number_of_patches = len(xy)
    print(f"number_of_patches: {number_of_patches}")

    # 阶段一：Forward建模网络（子波学习）
    forward_net = forward_model(nonlinearity=config['forward_nonlinearity']).to(device)
    forward_net.load_state_dict(torch.load(model_path1, map_location=device))
    forward_net.eval()
    

    # 阶段二：加载预训练模型
    # save_path = 'logs/model/Uet_TV_IMP_7labels_channel3_epoch=40.pth'
    # 认为config一定不为None

    
    net = UNet(
        in_ch=config['unet_in_channels'],                 # 输入通道：[最小二乘初始解, 观测地震数据]
        out_ch=config['unet_out_channels'],                # 输出通道：阻抗残差
        channels=config['unet_channels'],
        skip_channels=config['unet_skip_channels'],
        use_sigmoid=config['unet_use_sigmoid'],        # 输出归一化到[0,1]
        use_norm=config['unet_use_norm']
    ).to(device)
    net.load_state_dict(torch.load(model_path2, map_location=device))
    net.eval()

    # 推理阶段：构建子波算子
    # print("🔧 构建推理用子波算子...")
    # 如果没有传入config，使用默认值
    if config is None:
        config = {
            'wavelet_length': 101,
            'epsI': 0.1,
            'gaussian_std': 25
        }
    
    wav0 = wavelet_init(config['wavelet_length']).squeeze().numpy()
    wav00=torch.tensor(wav0[None, None, :, None],device=device)
    # 获取第一个batch的数据来确定维度
    first_batch = next(iter(test_loader))
    s_patch_first = first_batch[0]
    nz = s_patch_first.shape[2]  # 时间维度，与训练时保持一致
    
    # 使用与训练时相同的size计算方法
    # 训练时: size = data_info['seismic_shape'][0]
    # 这里我们需要从测试数据中获取相同的维度
    size = s_patch_first.shape[3]  # patch_size，与训练时的patch_size一致
    
    print(f"🔍 维度信息:")
    print(f"   - s_patch_first.shape: {s_patch_first.shape}")
    print(f"   - shape3d: {shape3d}")
    print(f"   - nz: {nz}")
    print(f"   - size: {size}")
    
    
    # 重新创建test_loader，因为我们已经消耗了第一个batch
    test_loader, xy, shape3d, norm_params = processor.process_test_data(batch_size=500,patch_size=120)
    number_of_patches = len(xy)
    epsI = config['epsI']
    S = torch.diag(0.5 * torch.ones(nz - 1), diagonal=1) - torch.diag(0.5 * torch.ones(nz - 1), diagonal=-1)
    S=S.to(device)
    S[0] = S[-1] = 0

    
    wav_learned_np= forward_net(wav00, wav00)[1].detach().cpu().squeeze().numpy()
    N = len(wav_learned_np)
    std = config['gaussian_std']
    print("N",N)
    print("std",std)
    gaussian_window = gaussian(N, std)
    wav_final = gaussian_window * (wav_learned_np - wav_learned_np.mean())
    wav_final = wav_final / wav_final.max()
    WW = pylops.utils.signalprocessing.convmtx(wav_final, size, len(wav_final) // 2)[:size]
    WW = torch.tensor(WW,device=device)
    WW = WW.float()
    S = S.float()
    WW = WW @ S
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0], device=device)
    
    print(f"🔍 调试信息:")
    print(f"   - size: {size}")
    print(f"   - WW.shape: {WW.shape}")
    print(f"   - S.shape: {S.shape}")
    print(f"   - PP.shape: {PP.shape}")
    # print(f"✅ 推理子波算子构建完成:")
    # print(f"   - 子波长度: {len(wav_final)}")
    # print(f"   - 卷积算子形状: {WW.shape}")
    # print(f"   - 子波类型: {'学习的子波' if use_learned_wavelet else '初始子波'}")
    # print("🔍 开始测试patch推理...")
    # 1. 获取patch loader、索引、shape、归一化参数
    
    # 2. 推理循环
    pred_patch_list = []
    true_patch_list = []
    input_patch_list = []
    back_patch_list = []
    indices_list = []
    seismic_patch_list = []
    logimpmax = norm_params['logimpmax']
    logimpmin = norm_params['logimpmin']
    with torch.no_grad():
        for i, (s_patch, imp_patch, zback_patch,indice ) in enumerate(test_loader):
            s_patch = s_patch.to(device)
            imp_patch = imp_patch.to(device)
            zback_patch = zback_patch.to(device)
            
            if i == 0:  # 只打印第一个batch的信息
                print(f"🔍 数据形状:")
                print(f"   - s_patch.shape: {s_patch.shape}")
                print(f"   - zback_patch.shape: {zback_patch.shape}")
                print(f"   - imp_patch.shape: {imp_patch.shape}")
            
            # 最小二乘初始化
            datarn = torch.matmul(WW.T, s_patch - torch.matmul(WW, zback_patch))
            x, _, _, _ = torch.linalg.lstsq(PP[None, None], datarn)
            Z_init = x + zback_patch
            Z_init = (Z_init - Z_init.min()) / (Z_init.max() - Z_init.min())
            # 网络推理
            Z_pred = net(torch.cat([Z_init, s_patch], dim=1)) + Z_init
            # 收集patch结果（适配batch>1）
            # 直接squeeze和tolist后用extend批量添加，避免for循环
            Z_pred_np = np.squeeze(Z_pred.cpu().numpy(), axis=1)  # [batch, time, patch_size]
            imp_patch_np = np.squeeze(imp_patch.cpu().numpy(), axis=1)
            zback_patch_np = np.squeeze(zback_patch.cpu().numpy(), axis=1)
            s_patch_np = np.squeeze(s_patch.cpu().numpy(), axis=1)
            pred_patch_list.extend(list(Z_pred_np))
            true_patch_list.extend(list(imp_patch_np))
            back_patch_list.extend(list(zback_patch_np))
            seismic_patch_list.extend(list(s_patch_np))
            indices_list.extend(indice.tolist())
            current_patches = len(pred_patch_list)
            print(f"   处理patch {current_patches}/{number_of_patches}")
    # 3. 拼回3D体
    # print(f"   拼回3D体...") # [N, time, patch_size]
    pred_3d = processor.reconstruct_3d_from_patches(pred_patch_list, indices_list)
    true_3d = processor.reconstruct_3d_from_patches(true_patch_list, indices_list)
    back_3d = processor.reconstruct_3d_from_patches(back_patch_list, indices_list)
    seismic_3d = processor.reconstruct_3d_from_patches(seismic_patch_list, indices_list)
    
    # 4. 反归一化
    # pred_3d_imp = np.exp(pred_3d * (logimpmax - logimpmin) + logimpmin)
    # true_3d_imp = np.exp(true_3d * (logimpmax - logimpmin) + logimpmin)
    # back_3d_imp = np.exp(back_3d * (logimpmax - logimpmin) + logimpmin)
    pred_3d_imp = pred_3d
    true_3d_imp = true_3d
    back_3d_imp = back_3d
    
    # 5. 保存所有可视化需要的文件
    os.makedirs(f'{folder_dir}', exist_ok=True)
    # np.save('logs/results/prediction_sample.npy', np.array(pred_patch_list))
    # np.save('logs/results/true_sample.npy', np.array(true_patch_list))
    # np.save('logs/results/input_sample.npy', np.array(seismic_patch_list))
    # np.save(os.path.join(folder_dir, 'seismic_record.npy'), seismic_3d)
    # np.save(os.path.join(folder_dir, 'prediction_impedance.npy'), pred_3d_imp)
    # np.save(os.path.join(folder_dir, 'true_impedance.npy'), true_3d_imp)
    # np.save(os.path.join(folder_dir, 'background_impedance.npy'), back_3d_imp)

    print(f"✅ 推理数据已保存: {folder_dir}/prediction_impedance.npy 等 shape: {pred_3d_imp.shape}")
    # print("\n" + "="*80)
    # print("🎉 程序执行完成")
    # print("="*80) 
    
    plot_well_curves_seisvis(true_3d_imp, pred_3d_imp, well_pos=None, back_imp=back_3d_imp, save_dir=folder_dir)
    # plot_well_curves_seisvis(true_imp, pred_imp, well_positions, back_imp=back_imp, save_dir='results')
    plot_sections_with_wells(pred_3d_imp, true_3d_imp, back_3d_imp, seismic_3d, well_pos=None, section_type='inline', save_dir=folder_dir)
    plot_sections_with_wells(pred_3d_imp, true_3d_imp, back_3d_imp, seismic_3d, well_pos=None, section_type='xline', save_dir=folder_dir)
    
    # thread1.join()
    # thread2.join()


# model_path1 = '/home/shendi_gjh_cj/codes/3D_project/logs/20250803-17-38-42/model/forward_net_wavelet_learned.pth'  # Forward建模网络路径
# model_path2='/home/shendi_gjh_cj/codes/3D_project/logs/20250803-17-38-42/model/Uet_TV_IMP_7labels_channel3_epoch=20.pth'
# inference(model_path1,model_path2,folder_dir='logs/test')

# def visualize_thread(pred_3d_imp, true_3d_imp, back_3d_imp, seismic_3d, folder_dir):
#     # 可视化结果
#     plot_sections_with_wells(pred_3d_imp, true_3d_imp, back_3d_imp, seismic_3d, well_pos=None, section_type='inline', save_dir=folder_dir)
#     plot_sections_with_wells(pred_3d_imp, true_3d_imp, back_3d_imp, seismic_3d, well_pos=None, section_type='xline', save_dir=folder_dir)



# inference()