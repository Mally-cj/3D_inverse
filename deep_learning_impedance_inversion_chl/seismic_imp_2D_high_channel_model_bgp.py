## 2024.06.64
## Hongling Chen
## Xi'an Jiaotong University
## multichannel seismic impedance inversion by semi-supervised manner for practical application
## Test: 

import os
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

Train = True

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
dtype = torch.cuda.FloatTensor

#######################################子波、阻抗、低频阻抗数据以及地震数据的生成过程############################################
### wavelet  为了方便，调用了pylops中现成的子波函数
dt0 = 0.001
ntwav = 51 #half size
wav, twav, wavc = ricker(np.arange(ntwav)*dt0, 30)

#impedance model   导入真实阻抗模型数据
segy = _read_segy("/home/shendi_chl/BGP/seismic_Impedance_inversion_2D/datasets/yp_imp_m2_gen.sgy") #field data
impedance_model = []
for i in range(0,len(segy.traces)):
    impedance_model.append(segy.traces[i].data)
impedance_model = np.array(impedance_model).reshape(200, 131, 198).transpose(2,1,0)
impedance_model = np.log(impedance_model)

# low-frequency model 导入测井曲线插值的阻抗数据
segy = _read_segy("/home/shendi_chl/BGP/seismic_Impedance_inversion_2D/datasets/intial_imp_m2_fortrain_7wells1.sgy") #field data
impedance_model_log = []
for i in range(0,len(segy.traces)):
    impedance_model_log.append(segy.traces[i].data)
impedance_model_log = np.array(impedance_model_log).reshape(200, 131, 198).transpose(2,1,0)
impedance_model_log = np.log(impedance_model_log)

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

#synthetic data from ref  地震数据的合成,为了偷懒，同样用了pylop中现成的函数
dims = impedance_model.shape
PPop = pylops.avo.poststack.PoststackLinearModelling(wav, nt0=dims[0], spatdims=dims[1:3])
syn1 = PPop*impedance_model.flatten()
syn1 =  syn1.reshape(impedance_model.shape)  #用真实阻抗数据合成的地震数据

synlog = PPop*impedance_model_log.flatten()    #用测井曲线插值的阻抗数据合成的地震数据，可以用来引入有监督训练数据
synlog =  synlog.reshape(impedance_model_log.shape)

#可以在数据中加入一定的噪音干扰，
# np.random.seed(42)
# syn1 = syn1 + np.random.normal(0, 2e-2, dims)
# # calculate the SNR
# SNR = 10*np.log10(np.linalg.norm(syn1)**2/(np.linalg.norm(np.random.normal(0, 2e-2, dims))**2)) # about 10dB
# print(SNR)

####################################### 数据的生成过程 ####################################################################
nx, ny = syn1.shape[1:3]
well_pos = [[81,29], [29,144], [81,144], [49,79], [109,144], [109,109], [29,29]]  #假设已知井的空间位置
train_well = well_pos
# plt.figure()
# plt.imshow(impedance_model[102,...].T,cmap='jet',)
# plt.scatter(np.array(train_well)[:,0],np.array(train_well)[:,1],c='b');
# plt.xlabel('Crossline', fontdict={'size': 12})
# plt.ylabel('Inline', fontdict={'size': 12})
# plt.xlim(1, nx)
# plt.ylim(1, ny)
# plt.show()


#下面是经过空间中几口井随机产生的连井剖面，代码还是初级过程，可能会有点小问题，bgp以前应该有更稳健的方法
train_well1 = add_labels(train_well)
extension_length = 10  # 延长部分的长度
implow_train = []
imp_train = [] # true impedance
implog_train = []
syn_train = []
synlog_train = []
Masks = []
for i in range(30):
    path = generate_random_path(train_well1, extension_length)
    interpolated_points = interpolate_path_points(path)
    points = [(x, y) for x, y, _ in path]
    # plt.plot(np.array(points)[:,0], np.array(points)[:,1], 'y')
    # plt.show()

    points = points[1:-1]
    indexes = find_points_indices(interpolated_points, points)
    sigma = 5  # Standard deviation for Gaussian function
    Mask = generate_gaussian_weighted_matrix(syn1.shape[0], len(interpolated_points), indexes, sigma)
    Masks.append(Mask)

    implow_train.append(mback[:,np.array(interpolated_points)[:,0], np.array(interpolated_points)[:,1]])
    imp_train.append(impedance_model[:,np.array(interpolated_points)[:,0], np.array(interpolated_points)[:,1]])
    syn_train.append(syn1[:,np.array(interpolated_points)[:,0], np.array(interpolated_points)[:,1]])
    synlog_train.append(synlog[:,np.array(interpolated_points)[:,0], np.array(interpolated_points)[:,1]])
    implog_train.append(impedance_model_log[:,np.array(interpolated_points)[:,0], np.array(interpolated_points)[:,1]])


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
implow_train_set = torch.cat(implow_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
imp_train_set = torch.cat(imp_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
implog_train_set = torch.cat(implog_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
syn_train_set = torch.cat(syn_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
synlog_train_set = torch.cat(synlog_train_set,0)[...,None].permute(0,3,1,2).type(dtype)
Masks_set = torch.cat(Masks_set,0)[...,None].permute(0,3,1,2).type(dtype)

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

test_data = torch.tensor(syn1_nor[...,None]).permute(2,3,0,1).type(dtype)
test_low = torch.tensor(implow_nor[...,None]).permute(2,3,0,1).type(dtype)
test_imp = torch.tensor(imp_nor[...,None]).permute(2,3,0,1).type(dtype)
test_implog = torch.tensor(implog_nor[...,None]).permute(2,3,0,1).type(dtype)

##训练数据集与测试数据集的 集成
batch_size = 10
Train_loader = data.DataLoader(data.TensorDataset(syn1_set,logimp_set,logimp_set1,mback_set,Masks_set), batch_size=batch_size, shuffle=True)
Train_loader_sup = data.DataLoader(data.TensorDataset(synlog_set,logimp_set1,mback_set), batch_size=batch_size, shuffle=True)
Test_loader = data.DataLoader(data.TensorDataset(test_data, test_imp, test_implog, test_low), batch_size=batch_size, shuffle=False, drop_last=False)

##为了保证子波模块的加快收敛加入的初始子波，进而生成子波卷积矩阵，初始子波可以从外面输入
wav0 = wavelet_init(syn1_set.cpu().type(torch.float32), 101).squeeze().numpy()
size = syn1.shape[0]
W = pylops.utils.signalprocessing.convmtx(wav0, size, len(wav0) // 2)[:size]
W = torch.tensor(W, dtype=torch.float32, device='cuda')

##
N = len(wav0)  # 窗的长度
fp=30
fs = 1000
std = int((fs/fp)/2)  # 标准差，决定窗的宽度
gaussian_window = gaussian(N, std)
gaus_win = torch.tensor(gaussian_window[None,None,:,None]).type(dtype)


#######################################################################################################################
##阻抗正演过程的差分算子
def DIFFZ(z):
    DZ= torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]]).type(dtype)
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
        DZ = torch.matmul(S.to('cuda'), z)
        return DZ

    def forward(self, z):
        WEIGHT = torch.tensor(self.wav.reshape(1, 1, self.wav.shape[0], 1)).type(torch.cuda.FloatTensor)
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
    net = UNet(input_depth, n_channels, channels=(8, 16, 32, 64),  skip_channels=(0, 8, 16, 32),use_sigmoid=True, use_norm=False).cuda()
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
            lossF =  mse(index * forward_net(DIFFZ(Cimp1), torch.tensor(wav0[None, None, :, None]).cuda())[0], index * y)*y.shape[3]
            lossF.backward()
            optimizerF.step()
            total_lossF.append(lossF.detach().cpu().numpy())


    wav00  = forward_net(DIFFZ(Cimp1), torch.tensor(wav0[None, None, :, None]).cuda())[1].detach().cpu().squeeze().numpy()
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
    WW = torch.tensor(WW, dtype=torch.float32, device='cuda')
    WW = WW@S.cuda()
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0]).to('cuda')
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
            loss_unsup = mse(forward_net(DIFFZ(out), torch.tensor(wav0[None, None, :, None]).cuda() )[0], y)

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
            
 
    torch.save(net.state_dict(),save_path)  #网络权重的保存
    return wav00, out_np, psnr_net_total



def test(net, forward_net,save_path="", beta=0.05, mu=0.1):  # LR_x needed only if method!=fixed_point
 
    ##load the net
    net.load_state_dict(torch.load(save_path))
    net.eval()   
    
    # get optimizer and loss function:
    mse = torch.nn.MSELoss().type(dtype)  # using MSE loss
    
    predicted_impedance = []
    true_impedance = []
    true_mback = []
    test_loss = []

    wav00 = forward_net(torch.tensor(wav0[None, None, :, None]).cuda(), torch.tensor(wav0[None, None, :, None]).cuda())[1].detach().cpu().squeeze().numpy()
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
    WW = torch.tensor(WW, dtype=torch.float32, device='cuda')
    WW = WW@S.cuda()
    PP = torch.matmul(WW.T, WW) + epsI * torch.eye(WW.shape[0]).to('cuda')
    with torch.no_grad():    
        for y,Cimp,Cimp1,impback in Test_loader:

            datarn = torch.matmul(WW.T, y-torch.matmul(WW, impback))
            x, _, _, _ = torch.linalg.lstsq(PP[None,None],datarn)
            x = x + impback
            x = (x-x.min())/(x.max()-x.min())
            # x = Cimp.max()*x/x.max()

            out = net(torch.cat([x, y], dim=1)) + x
            out_np = out
            
            total_loss = mse(forward_net(DIFFZ(out), torch.tensor(wav0[None, None, :, None]).cuda())[0], y)
            test_loss.append(total_loss.item())
            
            true_impedance.append(Cimp)
            true_mback.append(impback)
            predicted_impedance.append(out_np)
            
    predicted_impedance = torch.cat(predicted_impedance, dim=0).detach().cpu().numpy()
    true_impedance = torch.cat(true_impedance, dim=0).detach().cpu().numpy()           
    true_mback = torch.cat(true_mback, dim=0).detach().cpu().numpy()
    return true_impedance, true_mback, predicted_impedance, test_loss



## 
net= get_network_and_input()
forward_net = forward_model(nonlinearity="tanh").cuda()
Forward_ope = ImpedanceOperator(wav[::-1].copy())
if Train:
    INV_wavelet, INV_imp, PSNRT  = train(net, forward_net, save_path='Weights/Uet_TV_IMP_7labels_channel3.pth')
else:
    Ture_imp, True_mimp, INV_imp, loss11  = test(net, forward_net, save_path='Weights/Uet_TV_IMP_7labels_channel3.pth')
    corr = np.corrcoef( INV_imp.flatten(),  Ture_imp.flatten())[0,1]
    print(corr)
    plt.figure();plt.imshow(INV_imp[150,0],vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Trace No.', fontdict={ 'size':12})
    plt.ylabel('Time (ms)', fontdict={'size':12})
    plt.show()
    plt.figure();plt.imshow(Ture_imp[150,0],vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Trace No.', fontdict={ 'size':12})
    plt.ylabel('Time (ms)', fontdict={'size':12})
    plt.show()
    plt.figure();plt.imshow(True_mimp[150,0], vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Trace No.', fontdict={ 'size':12})
    plt.ylabel('Time (ms)', fontdict={'size':12})
    plt.show()
    # plt.figure();plt.imshow(syn1, cmap='seismic');plt.show()
    plt.figure()
    plt.plot(INV_imp[0,0,:,31] )
    plt.plot(Ture_imp[0,0,:,31])
    plt.plot(True_mimp[0,0,:,31])
    plt.show()

    k = 120
    plt.figure();plt.imshow(INV_imp[:,0,k,:],vmin=0.2, vmax=1,cmap='jet', interpolation='bicubic');
    plt.xlabel('Crossline', fontdict={ 'size':12})
    plt.ylabel('Inline', fontdict={'size':12})
    plt.show()
    plt.figure();plt.imshow(Ture_imp[:,0,k,:],vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Crossline', fontdict={ 'size':12})
    plt.ylabel('Inline', fontdict={'size':12})
    plt.show()
    plt.figure();plt.imshow(True_mimp[:,0,k,:], vmin=0.2, vmax=1, cmap='jet');
    plt.xlabel('Crossline', fontdict={ 'size':12})
    plt.ylabel('Inline', fontdict={'size':12})
    plt.show()
    # plt.figure();plt.imshow(syn1[k,:,:].T, cmap='seismic'); plt.show()
    plt.figure();plt.imshow(implog_nor[k,:,:].T, cmap='jet'); plt.show()






