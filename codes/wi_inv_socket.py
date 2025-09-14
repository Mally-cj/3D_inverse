# cython:language_level=39
# -*- coding: utf-8 -*-
# from nt import mkdir
import os
import traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Specify which GPU to use
"""
Created on Wed Apr 10 10:55:26 2024
2D 反演 unet
@author: ET
"""
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/shendi_gjh_cj/codes/3D_project")
sys.path.append("/home/shendi_gjh_cj/codes/3D_project/codes")
import torch
import numpy as np
#import matplotlib.pyplot as plt
# import pylops
from scipy.signal.windows import tukey
from scipy import signal
import pdb
from tqdm import tqdm
# from et_socket import *
from wi_inv_model import UNet, DIPLoss,forward_model
# from Model.net2D import forward_model
import os
from utils import tv_loss
import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import data_tools as tools
from visual_results import plot_sections_with_wells,plot_well_curves_seisvis,plot_sections_with_wells_single


Config={
    # "cheng_norm":'mean_std',
    "cheng_norm":'min_max',
    "sup_coeff":2,
    "unsup_coeff":1,
    "dip_coeff":0.5,
    "dipin_coeff":0,
    # "data_norm":'mean_std',
    "data_norm":'min_max',
    'hha':'test'

}

if Config['data_norm'] == 'min_max' and Config['cheng_norm'] == 'mean_std':
    plan_id=1
elif Config['data_norm'] == 'mean_std' and Config['cheng_norm'] == 'mean_std':
    plan_id=2
elif Config['data_norm'] == 'mean_std' and Config['cheng_norm'] == 'min_max':
    plan_id=3
elif Config['data_norm'] == 'min_max' and Config['cheng_norm'] == 'min_max':
    plan_id=4


save_folder=f"/home/shendi_gjh_cj/codes/3D_project/logs/E12_{Config['sup_coeff']}_{Config['unsup_coeff']}_{Config['dip_coeff']}_{Config['dipin_coeff']}_{plan_id}"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

writer = tf.summary.create_file_writer(save_folder)
# tensorboard_callback = TensorBoard(log_dir=save_folder, histogram_freq=0)
print("save folder:",save_folder)

def custom_convmtx(h, n_samples, center):
    """
    替代pylops.utils.signalprocessing.convmtx

    参数:
        h (np.ndarray): 输入卷积核（一维数组）
        n_samples (int): 输出信号的样本数（矩阵的列数）
        center (int): 卷积核的中心位置（索引）

    返回:
        np.ndarray: 卷积矩阵（形状为 (n_samples, n_samples)）
    """
    h = np.asarray(h) / np.max(h)  # 归一化卷积核
    h_len = len(h)
    # 计算总行数（卷积后的理论长度）
    total_rows = h_len + n_samples - 1
    # 初始化全零矩阵
    matrix = np.zeros((n_samples, total_rows))

    # 填充托普利茨结构
    for i in range(total_rows):
        start_col = i - center  # 当前行的起始列位置
        for j in range(h_len):
            col_idx = start_col + j
            if 0 <= col_idx < n_samples:
                matrix[col_idx, i] = h[j]

    # 返回前 n_samples 行
    return matrix[:, :n_samples]

# from utils import wavelet_init
# wav = wavelet_init(257).squeeze().numpy()
# WW = custom_convmtx(wav, 257, 0)
# WW = WW.astype(np.float32)
# tools.single_imshow(WW[:,:],title="WW")

# import pylops
# nt0=601
# # WW2= pylops.utils.signalprocessing.convmtx(wav, nt0, len(wav)//2)[:601]
# WW2 = pylops.utils.signalprocessing.convmtx(wav, 257, 0)[:257]
# tools.single_imshow(WW2[:,:],title="WW2")


def custom_convmtx2(h, n_samples, center):
    """
    替代pylops.utils.signalprocessing.convmtx

    参数:
        h (np.ndarray): 输入卷积核（一维数组）
        n_samples (int): 输出信号的样本数（矩阵的列数）
        center (int): 卷积核的中心位置（索引）

    返回:
        np.ndarray: 卷积矩阵（形状为 (n_samples, n_samples)）
    """
    h_len = len(h)
    # 计算总行数（卷积后的理论长度）
    total_rows = h_len + n_samples - 1
    # 初始化全零矩阵
    matrix = np.zeros((n_samples, total_rows))

    # 填充托普利茨结构
    for i in range(total_rows):
        start_col = i - center  # 当前行的起始列位置
        for j in range(h_len):
            col_idx = start_col + j
            if 0 <= col_idx < n_samples:
                matrix[col_idx, i] = h[j]

    # 返回前 n_samples 行
    return matrix[:, :n_samples]

class WIInv:
    def __init__(self):
        self._device = None             #设备
        self._model = None              #阻抗网络
        self._modelF = None             #子波网络
        
        self._lossfun = None            #有监督损失函数
        self._lossfunF = None           #无监督损失函数
        self._diploss = None            #diploss损失函数
        self._optimizer = None          #阻抗网络优化器
        self._optimizerF = None         #子波网络优化器
        self._scheduler = None          #阻抗网络学习率迭代器

        self._epsI = 0.1

        self._fLoss = 10.0              #最优损失
        self._fLearningRate = 0.001    #学习率
        self._nCmp = 0                  #最大剖面道数
        self._nSample = 0               #剖面采样数
        self._fSampleStep = 0           #采样率s
        self._sSavePath = ""            #模型保存路径
        self._nTrainWavelet = 0         #子波网络训练次数
        self._nTrainImp = 0             #阻抗网络训练epoch
        self._nTrainBanch = 0           #batchsize

        self._wav = None                #初始子波
        self._nWav = 0                  #子波长
        self._gaussian_window = None    #子波边缘矫正窗函数
        self._LP_A = None               #初始阻抗低通滤波系数分母
        self._LP_B = None               #初始阻抗低通滤波系数分子
        self._avg_seis = 0              #地震归一化参数均值
        self._stddev_seis = 0           #地震归一化参数值域跨度
        self._avg_imp = 0               #log初始阻抗归一化参数均值
        self._stddev_imp = 0            #log初始阻抗归一化参数值域跨度

        self.vSeis = None               #训练地震剖面缓存
        self.vImp = None                #训练初始阻抗剖面缓存
        self.vImpLow = None             #训练初始阻抗低频剖面缓存
        self.vMask = None               #训练掩码剖面缓存

        self.WW = None                  #方阵化子波缓存
        self.PP = None                  #方阵化子波缓存
        self.mid_x   = 0                #最小二乘归一化参数均值
        self.range_x = 0                #最小二乘归一化参数值域跨度

        self.iloop = 0
         
    #差分
    def DIFFZ(self, z):
        DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]], dtype=torch.float32).to(self._device)
        DZ[...,:-1,:] = 0.5*(z[...,1:, :] - z[..., :-1, :])
        return DZ

    #初始化超参数
    #子波网络训练次数，阻抗网络训练epoch，batchsize，最大剖面道数，剖面采样数，采样率s，模型保存路径，子波数组，地震均值，地震值域跨度，初始阻抗均值，初始阻抗值域跨度
    def reset(self, nTrainWavelet, nTrainImp, nTrainBanch, nCmp, nSample, fSampleStep, sSavePath, vWavelet, avg_seis, 
    stddev_seis, avg_imp, stddev_imp, vSeis_min, vSeis_max, vImp_min, vImp_max,train_loader=None):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nTrainWavelet = nTrainWavelet
        self._nTrainImp = nTrainImp
        self._nTrainBanch = nTrainBanch
        self._nCmp = nCmp
        self._nSample = nSample
        self._fSampleStep = fSampleStep
        self._sSavePath = sSavePath
        self._nWav = vWavelet.size
        # self._wav = torch.tensor(vWavelet[None, None, :, None]).to(self._device)

        self._wav = torch.tensor((vWavelet/np.max(vWavelet))[None, None, :, None]).to(self._device)
        self._avg_seis = avg_seis
        self._stddev_seis = stddev_seis
        self._avg_imp = avg_imp
        self._stddev_imp = stddev_imp

        self._vSeis_min = vSeis_min
        self._vSeis_max = vSeis_max
        self._vImp_min = vImp_min
        self._vImp_max = vImp_max
        self.train_loader = train_loader

        #self._gaussian_window = gaussian(self._nWav, std = 1.0 / self._fSampleStep / 60)
        self._gaussian_window = tukey(self._nWav, 0.5)
        self._LP_B, self._LP_A = signal.butter(2, 12 * self._fSampleStep, 'low')

        print(self._nTrainWavelet,self._nTrainImp,self._nTrainBanch,self._nCmp,self._nSample,self._nWav,self._fSampleStep)

    #初始化网络
    #sOpenModel预测时加载最优模型路径
    def open(self, sOpenModel,folder=None):
        self._model      = UNet(2, 1, channels=(8, 16, 32, 64), skip_channels=(0, 8, 16, 32), use_sigmoid=True, use_norm=False).to(self._device)
        #self._model      = CG_net(2,8).to(self._device)
        self._modelF     = forward_model(nonlinearity="tanh").to(self._device)
        self._lossfun    = torch.nn.MSELoss()
        #self._lossfun    = MAELoss()
        self._lossfunF   = torch.nn.MSELoss()
        self._optimizer  = torch.optim.Adam(self._model.parameters(), lr=self._fLearningRate)  # using ADAM opt
        self._optimizerF = torch.optim.Adam(self._modelF.parameters(), lr=self._fLearningRate)
        self._scheduler  = torch.optim.lr_scheduler.StepLR(self._optimizer, gamma=.98, step_size=self._nTrainImp)
        self._diploss    = DIPLoss(self._device, int(0.02 / self._fSampleStep))

        if sOpenModel != "":
            try:
                self._model.load_state_dict(torch.load(sOpenModel)) 
                self._modelF.load_state_dict(torch.load(sOpenModel+"F")) 
                self._model.eval()
                self._modelF.eval()
                # print(QtCore.QCoreApplication.translate("etpyWIInv", "加载最优模型"))
            except:
                traceback.print_exc()
                # print(QtCore.QCoreApplication.translate("etpyWIInv", "加载最优模型失败"))
                return 0

        if folder is not None:
 


            WW=np.load(os.path.join(save_folder,"WW_PP.npz"))["WW"]
            PP=np.load(os.path.join(save_folder,"WW_PP.npz"))["PP"]
            self.WW=torch.tensor(WW).to(self._device)
            self.PP=torch.tensor(PP).to(self._device)
            self.mid_x = torch.tensor(np.load(os.path.join(save_folder,"WW_PP.npz"))["mid_x"]).to(self._device)
            self.range_x = torch.tensor(np.load(os.path.join(save_folder,"WW_PP.npz"))["range_x"]).to(self._device)
        return 1

    #预处理训练数据缓存
    #vSeis地震剖面，vImp原始阻抗，vMask一维掩码
    def setTrainData(self, vSeis, vImp, vMask):
        # vSeis = [torch.tensor(arr, dtype=torch.float32) for arr in vSeis]
        # vImp = [torch.tensor(arr, dtype=torch.float32) for arr in vImp]
        # vMask = [torch.tensor(arr, dtype=torch.float32) for arr in vMask]
        self.vSeis = []
        self.vImp = []
        self.vImpLow = []
        self.vMask = []

        # pdb.set_trace()

        for i in range(len(vSeis)):
            seis = vSeis[i]
            nanSeie = np.where(np.isnan(seis))
            imp  = np.where(np.isnan(vImp[i]), np.float32(1.0), vImp[i])
            imp  = np.where(imp < 1, np.float32(1.0), imp)
            # imp  = np.log(imp)  #阻抗对数化
            # nanImp = np.where(imp==0)


            # #合并地震和阻抗无效区域
            # union_indices = set(zip(*nanSeie)) | set(zip(*nanImp))
            # union_rows, union_cols = zip(*union_indices)
            # nanPred = (np.array(union_rows), np.array(union_cols))

            # seis[nanPred] = 0
            # seis = (seis - self._avg_seis) / self._stddev_seis
            seis = 2*(seis - self._vSeis_min) / (self._vSeis_max-self._vSeis_min) - 1
            self.vSeis.append(np.clip(seis, -5.0, 5.0))
            # imp[nanPred] = imp[np.where(imp!=0)].min()




            impLow = signal.filtfilt(self._LP_B, self._LP_A, imp.T) #滤低频阻抗
            impLow = signal.filtfilt(np.ones(3)/float(3), 1, impLow)
            impLow = signal.filtfilt(np.ones(3)/float(3), 1, impLow.T).astype(np.float32)
            # pdb.set_trace()
            imp = (imp-self._vImp_min) / (self._vImp_max-self._vImp_min)
            impLow = (impLow-self._vImp_min) / (self._vImp_max-self._vImp_min)
            # imp = (imp - self._avg_imp) / self._stddev_imp
            # impLow = (impLow - self._avg_imp) / self._stddev_imp
            mask = np.tile(vMask[i], [seis.shape[0], 1])  #掩码复制成二维
            # mask[nanPred] = 0

            self.vImp.append(imp)
            self.vImpLow.append(impLow)
            self.vMask.append(mask)

            #if i%10==0:
            #    plt.imsave("d:/test/vSeis/vSeisIN_%d.png"%(self.iloop), self.vSeis[-1], cmap='gray')
            #    plt.imsave("d:/test/vLabel/vImpIN_%d.png"%(self.iloop), self.vImp[-1], cmap='gray')
            #    self.iloop += 1

    #取一个训练数据
    def GetExample(self):
        vSeis = []
        vImp = []
        vImpLow = []
        vMask = []

        for _ in range(self._nTrainBanch):
            # pdb.set_trace()
            example_select = np.random.randint(0, len(self.vSeis))
            if self.vSeis[example_select].shape[1] > self._nCmp:    #超过最大剖面道数裁剪输入
                while 1:    #训练数据必需包含井
                    startcmp = np.random.randint(0, self.vSeis[example_select].shape[1] - self._nCmp)
                    if np.sum(self.vMask[example_select][None, :, startcmp:startcmp+self._nCmp]) > 0:
                        break
                vSeis.append(self.vSeis[example_select][None, :, startcmp:startcmp+self._nCmp])
                vImp.append(self.vImp[example_select][None, :, startcmp:startcmp+self._nCmp])
                vImpLow.append(self.vImpLow[example_select][None, :, startcmp:startcmp+self._nCmp])
                vMask.append(self.vMask[example_select][None, :, startcmp:startcmp+self._nCmp])
            else:
                vSeis.append(self.vSeis[example_select][None])
                vImp.append(self.vImp[example_select][None])
                vImpLow.append(self.vImpLow[example_select][None])
                vMask.append(self.vMask[example_select][None])


        max_length = max(arr.shape[2] for arr in vSeis)
        min_length = min(arr.shape[2] for arr in vSeis)
        if max_length != min_length:    #batch间数据道数不一致填充成一样
            for i in range(self._nTrainBanch):
                if max_length != vSeis[i].shape[2]:
                    vSeis[i] = np.pad(vSeis[i], ((0,0), (0,0), (0,max_length-vSeis[i].shape[2])), 'edge')
                    vImp[i] = np.pad(vImp[i], ((0,0), (0,0), (0,max_length-vImp[i].shape[2])), 'edge')
                    vImpLow[i] = np.pad(vImpLow[i], ((0,0), (0,0), (0,max_length-vImpLow[i].shape[2])), 'edge')
                    vMask[i] = np.pad(vMask[i], ((0,0), (0,0), (0,max_length-vMask[i].shape[2])), 'edge')

        vSeis = torch.tensor(np.asarray(vSeis)).to(self._device)
        vImp = torch.tensor(np.asarray(vImp)).to(self._device)
        vImpLow = torch.tensor(np.asarray(vImpLow)).to(self._device)
        vMask = torch.tensor(np.asarray(vMask)).to(self._device)
        vSeis = vSeis.float()
        vImp = vImp.float()
        vImpLow = vImpLow.float()
        vMask = vMask.float()
        return vSeis, vImp, vImpLow, vMask

    #训练子波网络，初始化子波相关参数
    def trainWavelet(self):
        lossw = 0
        lossF_list=[]
        # progressBar.progress(progressBar.START, self._nTrainWavelet, QtCore.QCoreApplication.translate("etpyWIInv", "训练子波"))
        # for iter in tqdm(range(self._nTrainWavelet),desc="Training Wavelet"):
        for iter in range(self._nTrainWavelet):
            for vSeis, vImp, _, vMask in self.train_loader:
                vSeis = vSeis.to(self._device)
                vImp = vImp.to(self._device)
                vMask = vMask.to(self._device)
                # vSeis, vImp, _, vMask = self.GetExample()
                # pdb.set_trace()
                self._optimizerF.zero_grad()
                # vImp=vImp.float()
                self._wav=self._wav.float()

                # pdb.set_trace()

                predF = self._modelF(self.DIFFZ(vImp), self._wav)
                # pdb.set_trace()
             
                # plt.plot(vSeis[0,0,:,10].cpu().detach().numpy(),label="vSeis")
                # plt.plot(predF[0][0,0,:,10].cpu().detach().numpy(),label="predF")
                # plt.legend()
                # plt.show()
                lossF = self._lossfunF(vMask * predF[0], vMask * vSeis)
                # pdb.set_trace()
                # loss
                # _nSample=601
                # vSeis.shape[3]=120
                # self._nTrainBanch 就是batchsize
                lossF= lossF*vSeis.shape[0]*vSeis.shape[3]*vSeis.shape[2]/torch.sum(vMask)
                # lossF*=self._nTrainBanch*self._nSample*vSeis.shape[3]/torch.sum(vMask)
                lossF.backward()
                self._optimizerF.step()
                lossw += lossF.item()
                lossF_list.append(lossF.item())
                # progressBar.progress(progressBar.STEP)
                # tqdm.set_postfix_str(f"Loss: {lossF.item():.4f}")
                print(f"Epoch: {iter:2d}, Loss: {lossF:.4f}", end='\r', flush=True)
                # print(f"Epoch: {iter:2d}, Loss: {lossF:.4f}")
        print('train wavelet ->final_loss:',lossF)


        plt.plot(lossF_list)
        # plt.title("第一阶段损失函数")
        plt.title("the loss of stage 1")

        plt.show()
        # re_sesimic,_ = self._modelF(self.DIFFZ(tensor_true_3d), self._wav)
        # well_true=re_sesimic[0,0,:,140].cpu().detach().numpy()
        # plt.plot(well_true,label="re_sesimic")
        # plt.plot(seismic_3d[:,140,0],label="true_sesimic")
        # plt.legend()
        # plt.show()

        lossw /= self._nTrainWavelet
        print('total loss:',lossw)

        #print("LastWaveletLoss:",self._nTrainBanch*self._nSample*vSeis.shape[3]/torch.sum(vMask),lossF.item())
        # print(QtCore.QCoreApplication.translate("etpyWIInv", "子波损失") + ":",lossw)
        # progressBar.progress(progressBar.STOP)

        try:
            torch.save(self._modelF.state_dict(),self._sSavePath+"F")
            # print(QtCore.QCoreApplication.translate("etpyWIInv", "保存子波模型"))
        except:
            traceback.print_exc()
            # print(QtCore.QCoreApplication.translate("etpyWIInv", "保存子波模型失败"))
            return 0

        self._modelF.eval()
        # _, vImp, _, _ = self.GetExample()

        # wav = wavelet_init(257).squeeze().numpy()
        WW = custom_convmtx(self._wav.cpu().detach().squeeze().numpy(), 601, len(self._wav.cpu().detach().squeeze().numpy())//2)
        WW = WW.astype(np.float32)
        # tools.single_imshow(WW[:,:],title="WW")     ##*601
        
        S= np.diag(0.5 * np.ones(601-1, dtype='float32'), k=1) - np.diag(
                0.5 * np.ones(601-1, dtype='float32'), -1)
        S[0] = S[-1] = 0
        WS=np.einsum('ij,jk->ik', WW, S)    ##601*601q
        for vSeis, vImp, _, vMask in self.train_loader:
            vSeis = vSeis.to(self._device)
            vImp = vImp.to(self._device)
            vMask = vMask.to(self._device)
            break
        
        # pdb.set_trace()
        well_loc=-1
        well_loc_batch=-1
        for batch in range(vMask.shape[0]):
            for a in range(vMask.shape[-1]):
                # print("max=",vMask[batch,0,0,a].max())
                if vMask[batch,0,0,a].max() ==1:
                    well_loc=a
                    well_loc_batch=batch
                    break
        print(f'well_loc={well_loc},well_loc_batch={well_loc_batch}')
        # pdb.set_trace()

        re_sesimic=WS@vImp[well_loc_batch,0].cpu().detach().numpy()  ##601*1189

        sei, wav0  = self._modelF(self.DIFFZ(vImp), self._wav)
        # tools.single_imshow(re_sesimic,vmin=-0.2,vmax=0.2,title="WS@vImp")
        # tools.single_imshow(vSeis[well_loc_batch,0].cpu().detach().numpy(),vmin=-0.2,vmax=0.2,title="true_vSeis")
        # tools.single_imshow(sei[well_loc_batch,0].cpu().detach().numpy(),vmin=-0.2,vmax=0.2,title="_modelF(self.DIFFZ(vImp)")
        # tools.single_imshow((vSeis[well_loc_batch,0]*vMask[well_loc_batch,0]).cpu().detach().numpy(),vmin=-0.2,vmax=0.2,title="mask_true_vSeis")
        # tools.single_imshow((sei[well_loc_batch,0]*vMask[well_loc_batch,0]).cpu().detach().numpy(),vmin=-0.2,vmax=0.2,title="mask_vMask")

        # plt.plot(re_sesimic[:,well_loc],label="WS@DIFFZ(vImp)")
        # plt.plot(vSeis[well_loc_batch,0,:,well_loc].cpu().detach().numpy(),label="trueSeis")
        # plt.plot(sei[well_loc_batch,0,:,well_loc].cpu().detach().numpy(),label="re_sei")
        # plt.legend()
        # plt.show()



        
        wav0=wav0.detach().cpu().squeeze().numpy()
        wav00 = self._gaussian_window * (wav0 - wav0.mean())
        # pdb.set_trace()
        plt.plot(wav0,label="model(wav)")
        plt.plot(self._wav.detach().cpu().squeeze().numpy(),label="origin wav")
        plt.plot(wav00,label="gaussian wav")
        
        plt.legend()
        plt.show()

        S = torch.diag(0.5 * torch.ones(self._nSample - 1), diagonal=1) - torch.diag(0.5 * torch.ones(self._nSample - 1), diagonal=-1)
        S[0] = S[-1] = 0
        # WW = pylops.utils.signalprocessing.convmtx(wav00/wav00.max(), self._nSample, self._nWav // 2)[:self._nSample]
        WW = custom_convmtx(wav00, self._nSample, self._nWav // 2)
        WW = torch.tensor(WW.astype(np.float32)).to(self._device)
        self.WW = WW @ S.to(self._device)
        self.PP = (torch.matmul(self.WW.T, self.WW) + self._epsI * (torch.eye(self.WW.shape[0])).to(self._device))[None,None]

        # pdb.set_trace()

        ## 把WW和PP保存为npz

        # 统计最小二乘归一化参数
        # self.mid_x =0
        # self.range_x =0
        # for _ in range(10):
        #     vSeis, _, vImpLow, _ = self.GetExample()
        #     datarn = torch.matmul(self.WW.T, vSeis - torch.matmul(self.WW, vImpLow))
        #     x, _, _, _ = torch.linalg.lstsq(self.PP, datarn)
        #     x = x + vImpLow
        #     self.mid_x += (x.max() + x.min()) / 2
        #     self.range_x += (x.max() - x.min()) / 2
        # self.mid_x /= 10
        # self.range_x /= 10
        
        self.mid_x =0
        self.range_x =0
        cnt=0
        for vSeis, vImp, vImpLow, vMask in self.train_loader:
            vSeis = vSeis.to(self._device)
            vImpLow = vImpLow.to(self._device)
   
            datarn = torch.matmul(self.WW.T, vSeis - torch.matmul(self.WW, vImpLow))
            x, _, _, _ = torch.linalg.lstsq(self.PP, datarn)

            x = x + vImpLow
            # tools.single_imshow(x,title="x")
            # # ic(x.shape)
            # pdb.set_trace()
            # plt.plot(x[0,0,:,10],label="x")
            # plt.plot(vImpLow[0,0,:,10],label="vImpLow")
            # plt.legend()
            # plt.show()

            self.mid_x += (x.max() + x.min()) / 2
            self.range_x += (x.max() - x.min()) / 2
            # break

            # for j in range(x.shape[0]):
            #     self.mid_x += (x[j].max() + x[j].min()) / 2
            #     self.range_x += (x[j].max() - x[j].min()) / 2
            #     cnt+=1
        self.mid_x /= len(self.train_loader)
        self.range_x /= len(self.train_loader)

        np.savez(os.path.join(save_folder,"WW_PP.npz"),WW=self.WW.cpu().detach().numpy(),
        PP=self.PP.cpu().detach().numpy(),mid_x=self.mid_x.cpu(),range_x=self.range_x.cpu())
        

        return lossw

    #训练阻抗网络
    def train(self):
        self._model.train()
        loss = 0
        #losss=0
        #lossd=0
        #lossu=0
        vmin=float('inf')
        vmax=float('-inf')
        if Config['cheng_norm'] == 'min_max':
            for vSeis, vImp, vImpLow, vMask in self.train_loader:
                vSeis = vSeis.to(self._device)
                vImp = vImp.to(self._device)
                vImpLow = vImpLow.to(self._device)
                datarn = torch.matmul(self.WW.T, vSeis - torch.matmul(self.WW, vImpLow)) #减去低频背景的影响，借鉴的pylops
                x, _, _, _ = torch.linalg.lstsq(self.PP, datarn)
                x = x + vImpLow
                batch_vmin = x.min().item()
                batch_vmax = x.max().item()
                vmin = min(vmin, batch_vmin)
                vmax = max(vmax, batch_vmax)
            print(f"训练时,vmin: {vmin}, vmax: {vmax}")

        print(f"sup_coeff: {Config['sup_coeff']}, unsup_coeff: {Config['unsup_coeff']}, dip_coeff: {Config['dip_coeff']}")
        iter=0
        best_total_loss=float('inf')
        for i in range(self._nTrainImp):
            for vSeis, vImp, vImpLow, vMask in self.train_loader:
                vSeis = vSeis.to(self._device)
                vImp = vImp.to(self._device)
                vImpLow = vImpLow.to(self._device)
                vMask = vMask.to(self._device)
                # vSeis, vImp, vImpLow, vMask = self.GetExample()
                # pdb.set_trace()
                self._optimizer.zero_grad()
                datarn = torch.matmul(self.WW.T, vSeis - torch.matmul(self.WW, vImpLow)) #减去低频背景的影响，借鉴的pylops
                x, _, _, _ = torch.linalg.lstsq(self.PP, datarn)
                x = x + vImpLow  #最小二乘解
                # pdb.set_trace()
                # tools.single_imshow(x[0,0],title="x")

                if Config['cheng_norm'] == 'mean_std':
                    x_n = (x - self.mid_x) / self.range_x #(x-x.min())/(x.max()-x.min())  #进行了一步归一化，避免最小二乘大小带来的影响
                else:
                    x_n = (x - vmin) / (vmax - vmin)
                # pdb.set_trace()
                # plt.plot(x[0,0,:,10],label="origin")
                # plt.plot(x[0,0,:,10],label="impedance")
                # plt.plot(vImpLow[0,0,:,10],label="low")
                # plt.legend()
                # plt.show()
                # break

                pred = self._model(torch.cat([x_n, vSeis], dim=1)) + x_n
                predF = self._modelF(self.DIFFZ(pred), self._wav)
                loss_sup   = self._lossfun(vMask * pred, vMask * vImp) * self._nTrainBanch*self._nSample*vSeis.shape[3]/torch.sum(vMask)
                loss_unsup = self._lossfunF(predF[0], vSeis)

                # plt.plot(predF[0,0,:,10],label="pred")
                loss_tv   = tv_loss(pred,1.0)
                loss_dip,_ = self._diploss(vSeis, pred)
                loss_low = self._lossfun(pred, vImpLow)
                total_loss = Config['sup_coeff']*loss_sup + Config['unsup_coeff']*loss_unsup + Config['dip_coeff']*loss_dip + Config['dipin_coeff']*loss_low
                loss += total_loss.item()
                #losss += loss_sup.item()
                #lossd += loss_tv.item()
                #lossu += loss_unsup.item()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1)  
                self._optimizer.step()  
                self._scheduler.step()
                loss_info={
                    "loss_sup":loss_sup.item(),
                    "loss_unsup":loss_unsup.item(),
                    "loss_tv":loss_tv.item(),
                    "loss_dip":loss_dip.item(),
                    "loss_low":loss_low.item(),
                    "total_loss":total_loss.item()
                }
                iter+=1
                with writer.as_default():
                    tf.summary.scalar('loss_sup', loss_sup.item(), step=iter)
                    tf.summary.scalar('loss_unsup', loss_unsup.item(), step=iter)
                    tf.summary.scalar('loss_tv', loss_tv.item(), step=iter)
                    tf.summary.scalar('loss_dip', loss_dip.item(), step=iter)
                    tf.summary.scalar('loss_low', loss_low.item(), step=iter)
                    tf.summary.scalar('total_loss', total_loss.item(), step=iter)
                # print(f"loss_sup={loss_sup.item()}, loss_unsup={loss_unsup.item()}, loss_tv={loss_tv.item()}, loss_dip={loss_dip.item()}, loss_low={loss_low.item()}, total_loss={total_loss.item()}")
                print(f"Epoch: {i:2d}, Loss: {total_loss:.4f}", end='\r', flush=True)
                if iter> 50 and total_loss.item() < best_total_loss:
                    best_total_loss = total_loss.item()
                    self.save()
                    print(f"best total loss: {best_total_loss}")
                if iter %10 ==0:
                    print(f"total loss: {total_loss.item()}")
                    # print(f"end epoch{i}!")
            

            #if i%50==0:
            #    for ib in range(len(vSeis)):
            #        plt.imsave("d:/test/vSeis/vSeis_%d.png"%(self.iloop), vSeis[ib,0].cpu().detach().numpy(), cmap='gray')
            #        plt.imsave("d:/test/vLabel/vImp_%d.png"%(self.iloop), vImp[ib,0].cpu().detach().numpy(), cmap='gray')
            #        #vImpLow.cpu().detach().numpy().tofile("d:/test/vLabel/vImpLow_%d.bin"%(self.iloop))
            #        #x.cpu().detach().numpy().tofile("d:/test/vMask/vMask_%d.bin"%(self.iloop))
            #        plt.imsave("d:/test/vRet/vPred_%d.png"%(self.iloop), pred[ib,0].cpu().detach().numpy(), cmap='gray')
            #        #predF[0].cpu().detach().numpy().tofile("d:/test/vRet/vPredF_%d.bin"%(self.iloop))
            #        self.iloop += 1
        print("loss:",loss)
        loss /= self._nTrainImp
        #losss /= self._nTrainImp
        #lossd /= self._nTrainImp
        #lossu /= self._nTrainImp
        #print("sup:",losss)
        #print("unsup:",lossu)
        #print("dip:",lossd)
        # pdb.set_trace()
        if loss < self._fLoss:
            self._fLoss = loss
            res = self.save()
            if res == 0:
                return -1;
        else:
            self._fLoss = self._fLoss * 1.005
        self.iloop += 1
        return loss

    #保存模型
    def save(self):
        if self._sSavePath == "":
            return 1
        try:
            # pdb.set_trace()
            torch.save(self._model.state_dict(), self._sSavePath)
            # print(QtCore.QCoreApplication.translate("etpyWIInv", "成功保存模型于") + ':',self.iloop)
        except:
            # print(QtCore.QCoreApplication.translate("etpyWIInv", "保存模型错误"))
            return 0
        return 1
    def pred_init(self):
        WW=np.load(os.path.join(save_folder,"WW_PP.npz"))["WW"]
        PP=np.load(os.path.join(save_folder,"WW_PP.npz"))["PP"]
        self.WW=torch.tensor(WW).to(self._device)
        self.PP=torch.tensor(PP).to(self._device)
        pass
    #预测一个剖面
    
    def pred(self, vSeis=None, vImp=None):
        # nanSeie = np.where(np.isnan(vSeis))
        # vImp = np.where(np.isnan(vImp), np.float32(1.0), vImp)
        # vImp = np.where(vImp < 1, np.float32(1.0), vImp)
        # # vImp = np.log(vImp)
        # # nanImp = np.where(vImp==0)
        # # union_indices = set(zip(*nanSeie)) | set(zip(*nanImp))
        # # union_rows, union_cols = zip(*union_indices)
        # # nanPred = (np.array(union_rows), np.array(union_cols))


        # # vSeis[nanPred] = 0
        # # vSeis = (vSeis - self._avg_seis) / self._stddev_seis
        # vSeis = 2*(vSeis - self._vSeis_min) / (self._vSeis_max-self._vSeis_min) - 1
        # vSeis = np.clip(vSeis, -5.0, 5.0)
        
        # # vImp[nanPred] = vImp[np.where(vImp!=0)].min()

        # vImpLow = signal.filtfilt(self._LP_B, self._LP_A, vImp.T)
        # vImpLow = signal.filtfilt(np.ones(3)/float(3), 1, vImpLow)
        # vImpLow = signal.filtfilt(np.ones(3)/float(3), 1, vImpLow.T).astype(np.float32)

        # vImpLow = (vImpLow - self._vImp_min) / (self._vImp_max-self._vImp_min) 
        # # vImpLow = (vImpLow - self._avg_imp) / self._stddev_imp
        # vSeis = torch.tensor(vSeis[None,None]).float().to(self._device)
        # vImpLow = torch.tensor(vImpLow[None,None]).float().to(self._device)
        ##读取self.WW和self.PP
        # WW=np.load(os.path.join(save_folder,"WW_PP.npz"))["WW"]
        # PP=np.load(os.path.join(save_folder,"WW_PP.npz"))["PP"]
        # self.WW=torch.tensor(WW).to(self._device)
        # self.PP=torch.tensor(PP).to(self._device)


        if Config['cheng_norm'] == 'min_max':
            vmin=float('inf')
            vmax=float('-inf')
            with torch.no_grad():
                for i, (vSeis, vImpLow, _, _) in enumerate(self.test_loader):
                    vSeis = vSeis.to(self._device)
                    vImpLow = vImpLow.to(self._device)
                    datarn = torch.matmul(self.WW.T, vSeis - torch.matmul(self.WW, vImpLow))
                    x, _, _, _ = torch.linalg.lstsq(self.PP, datarn)
                    x = x + vImpLow
                    batch_vmin = x.min().item()
                    batch_vmax = x.max().item()
                    vmin = min(vmin, batch_vmin)
                    vmax = max(vmax, batch_vmax)
            print(f"推理时,vmin: {vmin}, vmax: {vmax}")

        else:
            print(f"推理时,mid_x: {self.mid_x}, range_x: {self.range_x}")


        pred_patch_list = []
        true_patch_list = []
        seismic_patch_list = []
        indices_list = []
        implow_patch_list = []
        re_seismic_patch_list = []

        with torch.no_grad():
            for i, (vSeis, vImp,vImpLow, indice) in enumerate(self.test_loader):
                if i>150:break
                vSeis = vSeis.to(self._device)
                vImpLow = vImpLow.to(self._device)
                # vImp = vImp.to(self._device)
                # pdb.set_trace()
                datarn = torch.matmul(self.WW.T, vSeis - torch.matmul(self.WW, vImpLow))
                x, _, _, _ = torch.linalg.lstsq(self.PP, datarn)
                x = x + vImpLow
                if Config['cheng_norm'] == 'mean_std':
                    x_n = (x - self.mid_x) / self.range_x #(x-x.min())/(x.max()-x.min()) 
                else:
                    x_n = (x - vmin) / (vmax - vmin)
          
                vPred = self._model(torch.cat([x_n, vSeis], dim=1)) + x_n
                re_seismic,_ = self._modelF(self.DIFFZ(vPred), self._wav)


                assert torch.isnan(vPred).any() == False
             
                # pdb.set_trace()
                pred_patch_list.extend(list(np.squeeze(vPred.cpu().numpy(), axis=1)))
                true_patch_list.extend(list(np.squeeze(vImp.cpu().numpy(), axis=1)))
                seismic_patch_list.extend(list(np.squeeze(vSeis.cpu().numpy(), axis=1)))
                implow_patch_list.extend(list(np.squeeze(vImpLow.cpu().numpy(), axis=1)))
                re_seismic_patch_list.extend(list(np.squeeze(re_seismic.cpu().numpy(), axis=1)))
                # pred_patch_list.extend(list(vPred.cpu().detach().squeeze().numpy()))
                indices_list.extend(indice.tolist())
                # break


    
        # pred_3d = self.processor.reconstruct_3d_from_patches(pred_patch_list, indices_list)
        # true_3d = self.processor.reconstruct_3d_from_patches(true_patch_list, indices_list)
        # seismic_3d = self.processor.reconstruct_3d_from_patches(seismic_patch_list, indices_list)
        # implow_3d = self.processor.reconstruct_3d_from_patches(implow_patch_list, indices_list)
        # re_seismic_3d = self.processor.reconstruct_3d_from_patches(re_seismic_patch_list, indices_list)

        # tensor_true_3d=torch.tensor(true_3d[:,:,0].reshape(1,1,601,1189))
        # re_sesimic,_ = self._modelF(self.DIFFZ(tensor_true_3d), self._wav)


        # well_true=re_sesimic[0,0,:,140].cpu().detach().numpy()
        # np.save(os.path.join(save_folder, 're_seismic_record.npy'), re_seismic_3d)
        # np.save(os.path.join(save_folder, 'prediction_impedance.npy'), pred_3d)
        # np.save(os.path.join(save_folder, 'true_impedance.npy'), true_3d)
        # np.save(os.path.join(save_folder, 'implow_impedance.npy'), implow_3d)
        # np.save(os.path.join(save_folder, 'seismic_record.npy'), seismic_3d)
        # print("save to ",save_folder)
        # pdb.set_trace()

      
        #vSeis.cpu().detach().numpy().tofile("d:/test/vSeis/vSeis_%d.bin"%(self.iloop))
        #vImp.tofile("d:/test/vLabel/vImp_%d.bin"%(self.iloop))
        #vImpLow.cpu().detach().numpy().tofile("d:/test/vLabel/vImpLow_%d.bin"%(self.iloop))
        #x.cpu().detach().numpy().tofile("d:/test/vMask/vMask_%d.bin"%(self.iloop))
        #vPred.cpu().detach().numpy().tofile("d:/test/vRet/vPred_%d.bin"%(self.iloop))
        #self.iloop += 1

        # vPred = vPred.cpu().detach().squeeze().numpy()
        # vPred = np.exp(vPred * self._stddev_imp + self._avg_imp)
        # vPred[nanPred] = np.nan

        return vPred

    def getCmp(self):
        return self._nCmp

    def getSample(self):
        return self._nSample

predict = WIInv()


def process_socket(vInputBuf):
    streamInput = QtCore.QDataStream(vInputBuf)
    nFunc = streamInput.readUInt8()
    vBuf = QtCore.QByteArray()
    buffer = QtCore.QBuffer(vBuf)
    buffer.open(QtCore.QIODevice.WriteOnly)
    streamOutput = QtCore.QDataStream()
    streamOutput.setDevice(buffer)

    if nFunc == 1:
        nTrainImp = streamInput.readUInt32()
        nTrainBanch = streamInput.readUInt32()
        nCmp = streamInput.readUInt32()
        nSample = streamInput.readUInt32()
        fSampleStep = streamInput.readFloat()
        sSavePath = streamInput.readQString()
        nWavelet = streamInput.readUInt32()
        vWavelet = DecodeFloatArray(streamInput, nWavelet)
        avg_seis = streamInput.readFloat()
        stddev_seis = streamInput.readFloat()
        avg_imp = streamInput.readFloat()
        stddev_imp = streamInput.readFloat()
        predict.reset(int(3000/nTrainBanch), nTrainImp, nTrainBanch, nCmp, nSample, fSampleStep, sSavePath, vWavelet, avg_seis, stddev_seis, avg_imp, stddev_imp)
        streamOutput.writeUInt8(1)

    elif nFunc == 2:
        nSample = predict.getSample()
        vSeis = []
        vImp = []
        vMask = []
        while(1):
            nCrd = streamInput.readUInt32()
            if nCrd == 0:
                break
            data = DecodeFloatArray(streamInput, (nCrd, nSample))
            vSeis.append(np.transpose(data,(1,0)))
            data = DecodeFloatArray(streamInput, (nCrd, nSample))
            vImp.append(np.transpose(data,(1,0)))
            data = DecodeFloatArray(streamInput, nCrd)
            vMask.append(data)
        predict.setTrainData(vSeis, vImp, vMask)
        streamOutput.writeUInt8(1)

    elif nFunc == 3:
        sPath = streamInput.readQString()
        bRet = predict.open(sPath)
        streamOutput.writeUInt8(bRet)

    elif nFunc == 4:
        loss = predict.train()
        streamOutput.writeFloat(loss)

    elif nFunc == 5:
        lossw = predict.trainWavelet()
        streamOutput.writeFloat(lossw)
        
    elif nFunc == 6:
        nSample = predict.getSample()
        nCrd = streamInput.readUInt32()
        vSeis = DecodeFloatArray(streamInput, (nCrd, nSample))
        vImp = DecodeFloatArray(streamInput, (nCrd, nSample))
        vPred = predict.pred(np.transpose(vSeis,(1,0)), np.transpose(vImp,(1,0)))
        EncodeFloatArray(streamOutput, np.transpose(vPred,(1,0)))

    return vBuf


def run():
    StartSocket(process_socket)


# #这个地方需要根据实际的训练数据进行修改，bgp在训练的时候用的是把yyf_smo_train_Volume_PP_IMP当作模型数据，生成地震数据来对网络进行训练
if __name__ == '__main__':
    import sys
    from data_processor import SeismicDataProcessor
    processor = SeismicDataProcessor(cache_dir='cache',device='cpu',train_batch_size=60,train_patch_size=120,norm_method=Config['data_norm'])
    
    S_obs = processor.load_seismic_data()      
    shape_3d=S_obs.shape    
    well_pos, M_well_mask, M_well_mask_dict = processor.generate_well_mask(shape_3d)
    training_data = processor.build_training_profiles(
        well_pos, M_well_mask_dict
    )
 
    avg_imp=training_data['Z_full_train_set'].mean()
    stddev_imp=training_data['Z_full_train_set'].std()

    vImp_min=training_data['Z_full_train_set'].min()
    vImp_max=training_data['Z_full_train_set'].max()

    avg_seis=training_data['S_obs_train_set'].mean()
    stddev_seis=training_data['S_obs_train_set'].std()
    vSeis_min=training_data['S_obs_train_set'].min()
    vSeis_max=training_data['S_obs_train_set'].max()
    

    vSeis = []
    vImp = []
    vMask = []
    print("making dataset")
    for i in tqdm(range(100),desc="Making dataset"):
        vSeis.append(training_data['S_obs_train_set'][i,0])
        vImp.append(training_data['Z_full_train_set'][i,0])
        vMask.append(training_data['M_mask_train_set'][i,0,0])
    # pdb.set_trace()
    train_loader, _, _ = processor.process_train_data2()

    
    nTrainWavelet=100
    nTrainGroup=2
    nTrainImp=200
    nTrainBanch=70
    nCmp=1000
    nSample=601
    fSampleStep=0.001
    # vWavelet=np.random.random(257
    from utils import wavelet_init
    vWavelet = wavelet_init(257).squeeze().numpy()


    print("dataset made, start training wavelet")
    predict.reset(nTrainWavelet, nTrainImp, nTrainBanch, nCmp, nSample, fSampleStep, 
    sSavePath=os.path.join(save_folder, "test.pth"), 
    vWavelet=vWavelet, 
    avg_seis=avg_seis, 
    stddev_seis=stddev_seis, 
    avg_imp=avg_imp, 
    stddev_imp=stddev_imp,
    vSeis_min=vSeis_min,
    vSeis_max=vSeis_max,
    vImp_min=vImp_min,
    vImp_max=vImp_max,
    train_loader=train_loader
    )



    predict.processor = processor


    predict.setTrainData(vSeis, vImp, vMask)
    predict.open("")
    print("training wavelet")
    predict.trainWavelet()

#     # for iter in tqdm(range(nTrainGroup),desc="Training"):
#         # print("iter:", iter)
#         # vSeis = []
#         # vImp = []
#         # vMask = []
#         # for _ in range(10):
#         #     cmp=np.random.randint(800,1500)
#         #     pdb.set_trace()
#         #     vSeis.append(np.random.random([nSample,cmp]))
#         #     vImp.append(np.random.random([nSample,cmp])+1) #(500,1077)
#         #     vMask.append(np.random.randint(0,2,cmp)) #(1077,)
#         # predict.setTrainData(vSeis, vImp, vMask)
    predict.train()

    predict.open(os.path.join(save_folder, "test.pth"),folder=save_folder)
    predict.test_loader, _,_,_ = predict.processor.process_test_data2(
        batch_size=50,
        patch_size=1400
    )
    predict.test_axis=0
    predict.processor = SeismicDataProcessor(cache_dir='cache', device='cpu',test_axis=predict.test_axis,norm_method=Config['data_norm'])

#     # # vSeis=np.random.random([nSample,nCmp])
#     # # vImp=np.random.random([nSample,nCmp])
#     # vPred=predict.pred(vSeis[0],vImp[0])
    vPred=predict.pred()

    
    # pdb.set_trace()
    writer.close()
    
    