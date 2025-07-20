## 2024.06.64
## Hongling Chen
## Xi'an Jiaotong University
## multichannel seismic impedance inversion by semi-supervised manner for practical application
## Test: 


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# sys.path.append("/home/shendi_gjh_cj/codes/3D_project/deep_learning_impedance_inversion_chl")
# sys.path.append("/home/shendi_gjh_cj/codes/3D_project/codes")
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
# import MINE.data_tools  as tools
import sys
sys.path.append('..')
import data_tools as tools
from icecream import ic 
sys.path.append('../codes')
from cpp_to_py import generate_well_mask as generate_well_mask2
from cpp_to_py import get_wellline_and_mask as get_wellline_and_mask2
import pickle
import numpy as np

Train = True

# 检测设备并设置数据类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

# 数据缓存文件路径
CACHE_FILE = 'processed_data_cache.pkl'

def load_or_process_data():
    """
    加载已处理的数据，如果不存在则处理并保存
    """
    if os.path.exists(CACHE_FILE):
        print("Loading processed data from cache...")
        with open(CACHE_FILE, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict
    else:
        print("Processing data for the first time...")
        data_dict = process_raw_data()
        print("Saving processed data to cache...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data_dict, f)
        return data_dict

def process_raw_data():
    """
    处理原始数据的函数
    """
    print("Step 1: Loading wavelet...")
    #######################################子波、阻抗、低频阻抗数据以及地震数据的生成过程############################################
    ### wavelet  为了方便，调用了pylops中现成的子波函数
    dt0 = 0.001
    ntwav = 51 #half size
    wav, twav, wavc = ricker(np.arange(ntwav)*dt0, 30)

    print("Step 2: Loading impedance model...")
    #impedance model   导入真实阻抗模型数据
    segy=_read_segy("../data/yyf_smo_train_Volume_PP_IMP.sgy")
    
    impedance_model = np.array([trace.data for trace in segy.traces])
    impedance_model = impedance_model.reshape(251, 1189, 601).transpose(2, 1, 0)
    impedance_model = np.log(impedance_model)

    print("Step 3: Processing low-frequency model...")
    # low-frequency model 导入测井曲线插值的阻抗数据
    segy=_read_segy("../data/yyf_smo_train_Volume_PP_IMP.sgy")
    
    impedance_model_log = []
    for i in range(0,len(segy.traces)):
        impedance_model_log.append(segy.traces[i].data)
    
    impedance_model_log = np.array(impedance_model_log).reshape(251,len(impedance_model_log)//251,601).transpose(2,1,0)
    impedance_model_log = np.log(impedance_model_log)

    print("Step 4: Creating low-frequency background...")
    #低频是从测井曲线插值的阻抗剖面中平滑而来
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

    print("Step 5: Generating synthetic seismic data...")
    #synthetic data from ref  地震数据的合成,为了偷懒，同样用了pylop中现成的函数
    dims = impedance_model.shape
    PPop = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=dims[0], spatdims=dims[1:3])
    syn1 = PPop*impedance_model.flatten()
    syn1 =  syn1.reshape(impedance_model.shape)  #用真实阻抗数据合成的地震数据

    synlog = PPop*impedance_model_log.flatten()    #用测井曲线插值的阻抗数据合成的地震数据，可以用来引入有监督训练数据
    synlog =  synlog.reshape(impedance_model_log.shape)

    print("Step 6: Processing well positions...")
    ####################################### 数据的生成过程 ####################################################################
    nx, ny = syn1.shape[1:3]
    basex=450
    basey=212

    pos=[[594,295],[572,692],[591,996],[532,1053],[603,1212],[561,842],[504,846],[499,597]]
    well_pos=[[y-basey,x-basex] for [x,y] in pos ]
    train_well = well_pos

    grid_shape = syn1.shape[1:3]
    vWellMask = generate_well_mask2(well_pos, grid_shape, well_range=15, sigma=5)

    print("Step 7: Generating training data...")
    #下面是经过空间中几口井随机产生的连井剖面，代码还是初级过程，可能会有点小问题，bgp以前应该有更稳健的方法
    train_well1 = add_labels(train_well)
    extension_length = 10  # 延长部分的长度
    implow_train = []
    imp_train = [] # true impedance
    implog_train = []
    syn_train = []
    synlog_train = []
    Masks = []
    path_tem=[]
    for i in range(30):
        interpolated_points, vMask  = get_wellline_and_mask2(well_pos, grid_shape, vWellMask)
        vMask = np.tile(vMask, (601,1))
        Masks.append(vMask)
        implow_train.append(mback[:,interpolated_points[:,0], interpolated_points[:,1]])
        imp_train.append(impedance_model[:,interpolated_points[:,0], interpolated_points[:,1]])
        syn_train.append(syn1[:,interpolated_points[:,0], interpolated_points[:,1]])
        synlog_train.append(synlog[:,interpolated_points[:,0], interpolated_points[:,1]])
        implog_train.append(impedance_model_log[:,interpolated_points[:,0], interpolated_points[:,1]])

    print("Step 8: Creating patches and tensor datasets...")
    # 下面的代码是对上面生成的多个连井剖面进行空间方向分割，为了增加数据集的，也可以利用深度学习中的数据增广方式进一步增加数据集
    patchsize = 70
    oversize = 5
    implow_train_set = []
    imp_train_set = [] # true impedance
    implog_train_set = [] # impedance form well logs
    syn_train_set = []
    synlog_train_set = []
    Masks_set = []
    for i in range(30):
        implow_train_set.append(torch.tensor(image2cols(implow_train[i],(syn1.shape[0],patchsize),(1,oversize))))
        imp_train_set.append(torch.tensor(image2cols(imp_train[i],(syn1.shape[0],patchsize),(1,oversize))))
        syn_train_set.append(torch.tensor(image2cols(syn_train[i],(syn1.shape[0],patchsize),(1,oversize))))
        synlog_train_set.append(torch.tensor(image2cols(synlog_train[i],(syn1.shape[0],patchsize),(1,oversize))))
        Masks_set.append(torch.tensor(image2cols(Masks[i],(syn1.shape[0],patchsize),(1,oversize))))
        implog_train_set.append(torch.tensor(image2cols(implog_train[i],(syn1.shape[0],patchsize),(1,oversize))))
    
    implow_train_set = torch.cat(implow_train_set,0)[...,None].permute(0,3,1,2)
    imp_train_set = torch.cat(imp_train_set,0)[...,None].permute(0,3,1,2)
    implog_train_set = torch.cat(implog_train_set,0)[...,None].permute(0,3,1,2)
    syn_train_set = torch.cat(syn_train_set,0)[...,None].permute(0,3,1,2)
    synlog_train_set = torch.cat(synlog_train_set,0)[...,None].permute(0,3,1,2)
    Masks_set = torch.cat(Masks_set,0)[...,None].permute(0,3,1,2)

    print("Step 9: Normalizing data...")
    #下面是对训练数据集进行归一化
    logimpmax = impedance_model_log.max()
    logimpmin = impedance_model_log.min()
    logimp_set = (imp_train_set - logimpmin)/(logimpmax - logimpmin)
    logimp_set1 = (implog_train_set - logimpmin)/(logimpmax - logimpmin)
    syn1_set = 2*(syn_train_set - syn_train_set.min())/(syn_train_set.max() - syn_train_set.min())-1
    synlog_set = 2*(synlog_train_set - synlog_train_set.min())/(synlog_train_set.max() - synlog_train_set.min())-1
    mback_set = (implow_train_set  - logimpmin)/(logimpmax - logimpmin)

    #下面是对测试数据集的归一化
    syn1_nor =  2*(syn1 -syn1.min())/(syn1.max()-syn1.min())-1
    implow_nor = (mback - logimpmin)/(logimpmax - logimpmin)
    imp_nor = (impedance_model-logimpmin)/(logimpmax - logimpmin)
    implog_nor =  (impedance_model_log-logimpmin)/(logimpmax - logimpmin)

    print("Step 10: Generating wavelet...")
    wav0 = wavelet_init(syn1_set.cpu().type(torch.float32), 101).squeeze().numpy()
    
    # 将所有处理好的数据保存到字典中
    data_dict = {
        'wav': wav,
        'impedance_model': impedance_model,
        'impedance_model_log': impedance_model_log,
        'mback': mback,
        'syn1': syn1,
        'synlog': synlog,
        'well_pos': well_pos,
        'syn1_set': syn1_set,
        'logimp_set': logimp_set,
        'logimp_set1': logimp_set1,
        'mback_set': mback_set,
        'Masks_set': Masks_set,
        'synlog_set': synlog_set,
        'syn1_nor': syn1_nor,
        'implow_nor': implow_nor,
        'imp_nor': imp_nor,
        'implog_nor': implog_nor,
        'logimpmax': logimpmax,
        'logimpmin': logimpmin,
        'wav0': wav0
    }
    
    print("Data processing completed!")
    return data_dict

# 加载或处理数据
print("=== Loading/Processing Data ===")
data_dict = load_or_process_data()

# 从字典中提取数据
wav = data_dict['wav']
impedance_model = data_dict['impedance_model']
impedance_model_log = data_dict['impedance_model_log']
mback = data_dict['mback']
syn1 = data_dict['syn1']
synlog = data_dict['synlog']
well_pos = data_dict['well_pos']
syn1_set = data_dict['syn1_set'].type(dtype)
logimp_set = data_dict['logimp_set'].type(dtype)
logimp_set1 = data_dict['logimp_set1'].type(dtype)
mback_set = data_dict['mback_set'].type(dtype)
Masks_set = data_dict['Masks_set'].type(dtype)
synlog_set = data_dict['synlog_set'].type(dtype)
syn1_nor = data_dict['syn1_nor']
implow_nor = data_dict['implow_nor']
imp_nor = data_dict['imp_nor']
implog_nor = data_dict['implog_nor']
logimpmax = data_dict['logimpmax']
logimpmin = data_dict['logimpmin']
wav0 = data_dict['wav0']

print("=== Creating Data Loaders ===")
# 创建测试数据
test_data = torch.tensor(syn1_nor[...,None]).permute(2,3,0,1).type(dtype)
test_low = torch.tensor(implow_nor[...,None]).permute(2,3,0,1).type(dtype)
test_imp = torch.tensor(imp_nor[...,None]).permute(2,3,0,1).type(dtype)
test_implog = torch.tensor(implog_nor[...,None]).permute(2,3,0,1).type(dtype)

##训练数据集与测试数据集的 集成
batch_size = 10
Train_loader = data.DataLoader(data.TensorDataset(syn1_set,logimp_set,logimp_set1,mback_set,Masks_set), batch_size=batch_size, shuffle=True)
Train_loader_sup = data.DataLoader(data.TensorDataset(synlog_set,logimp_set1,mback_set), batch_size=batch_size, shuffle=True)
Test_loader = data.DataLoader(data.TensorDataset(test_data, test_imp, test_implog, test_low), batch_size=batch_size, shuffle=False, drop_last=False)

print("=== Setting up additional parameters ===")
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

print("=== Data loading completed! ===")

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
        for y,Cimp,Cimp1,impback,index in Train_loader: # Cimp1对应的测井插值生成的阻抗
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
        for y, Cimp, Cimp1, impback, index in Train_loader:
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
            loss_sup = yita*mse(index*out, index*Cimp1)*Cimp1.shape[3]/3
            loss_unsup = mse(forward_net(DIFFZ(out), torch.tensor(wav0[None, None, :, None], device=device))[0], y)

            total_loss = ( loss_unsup +  tv_loss(out, mu) + loss_sup ) + beta*total_loss_sup

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)  
            optimizer.step()  
            scheduler.step()
            
            #一些参数的输出，为了监测网络的训练
            if clean_img is not False:
                  psnr_net = np.corrcoef(Cimp.detach().cpu().flatten(), out_np.detach().cpu().flatten())[0,1]
                  psnr_net_total.append(psnr_net) 
                  print('\r',  '%04d/%04d Loss %f Loss_sup %f Loss_unsup %f total_loss_sup %f' % (i, admm_iter, total_loss.item(), loss_sup, loss_unsup, total_loss_sup),
                  'psnrs: net: %.4f' % (psnr_net), end='')        
            else:
                  print('\r', 'iteration %04d/%04d Loss %f' % (i, admm_iter, total_loss.item()), end='')
            
 
        torch.save(net.state_dict(),save_path+f"_{i:05d}")  #网络权重的保存
    return wav00, out_np, psnr_net_total

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
        for y,Cimp,Cimp1,impback in Test_loader:

            datarn = torch.matmul(WW.T, y-torch.matmul(WW, impback))
            x, _, _, _ = torch.linalg.lstsq(PP[None,None],datarn)
            x = x + impback
            x = (x-x.min())/(x.max()-x.min())
            # x = Cimp.max()*x/x.max()

            out = net(torch.cat([x, y], dim=1)) + x
            out_np = out
            
            total_loss = mse(forward_net(DIFFZ(out), torch.tensor(wav0[None, None, :, None], device=device))[0], y)
            test_loss.append(total_loss.item())
            
            true_impedance.append(Cimp)
            true_mback.append(impback)
            predicted_impedance.append(out_np)
            
    predicted_impedance = torch.cat(predicted_impedance, dim=0).detach().cpu().numpy()
    true_impedance = torch.cat(true_impedance, dim=0).detach().cpu().numpy()           
    true_mback = torch.cat(true_mback, dim=0).detach().cpu().numpy()
    return true_impedance, true_mback, predicted_impedance, test_loss

print("=== Creating Networks ===")
## 
net= get_network_and_input()
forward_net = forward_model(nonlinearity="tanh").to(device)
Forward_ope = ImpedanceOperator(wav[::-1].copy())

print("=== Starting Training/Testing ===")
if Train:
    INV_wavelet, INV_imp, PSNRT  = train(net, forward_net, save_path='Uet_TV_IMP_7labels_channel3.pth')
else:
    name=49
    Ture_imp, True_mimp, INV_imp, loss11  = test(net, forward_net, save_path=f'Uet_TV_IMP_7labels_channel3.pth_{name:05d}')
    

    well_pos2 = np.array(well_pos)
    corr = np.corrcoef( INV_imp[:, 0, well_pos2[:, 1], well_pos2[:, 0]].flatten(),  Ture_imp[:, 0, well_pos2[:, 1], well_pos2[:, 0]].flatten())[0,1]
    print(corr)
    tools.single_imshow(INV_imp[150,0], vmin=0.2, vmax=1, cmap='jet', title=f'INV_impedance_{name}')
    tools.single_imshow(Ture_imp[150,0], vmin=0.2, vmax=1, cmap='jet', title=f'Ture_imp_{name}')
    tools.single_imshow(True_mimp[150,0], vmin=0.2, vmax=1, cmap='jet', title=f'dipin_{name}')

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
