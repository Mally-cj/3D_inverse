import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import pdb
import threading
from data_tools import run_in_thread

# 全局matplotlib锁，防止多线程冲突
matplotlib_lock = threading.Lock()
def make_yushi_wavelet(nYushiFreqLow, nYushiFreqHigh, nWaveletSample, dt):       
    #p_par yushi 子波积分频率下界
    #q_par  子波积分频率上界
    #nsample：输出子波长度
    #s_interval:采样间隔
    p_par = nYushiFreqLow;
    q_par = nYushiFreqHigh;
    nsample = nWaveletSample;

    t =dt*np.linspace(-(nsample//2),nsample//2,nsample);
    y=1.0/(q_par-p_par)*(q_par*np.exp(-np.power(3.1415926*q_par*t,2))-p_par*np.exp(-np.power(3.1415926*p_par*t,2)));
    taper=np.ones([1,nsample],dtype='float32');
    taper[0,0:nsample//3]=0.5*(1-np.cos(3.1415926*np.arange(0,nsample//3, 1)/(nsample//3-1)))
    taper[0,nsample-nsample//3:nsample]=0.5*(1+np.cos(3.1415926*np.arange(0,nsample//3, 1)/(nsample//3-1)))
    y_taper=np.multiply(y,taper);
    w = np.reshape(y_taper, (-1));
    return w;


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None,None, :]


def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0][0]



def image2cols(image,patch_size,stride):

        """       
        image:需要切分为图像块的图像       
        patch_size:图像块的尺寸，如:(10,10)        
        stride:切分图像块时移动过得步长，如:5        
        """

        if len(image.shape) == 2:        
        # 灰度图像        
            imhigh,imwidth = image.shape       
        if len(image.shape) == 3:        
        # RGB图像       
            imhigh,imwidth,imch = image.shape
        
        ## 构建图像块的索引     
        if imhigh == patch_size[0]:
            range_y = [0]
        else:
           range_y = np.arange(0,imhigh - patch_size[0],stride[0])        
        range_x = np.arange(0,imwidth - patch_size[1],stride[1])
        
        if range_y[-1] != imhigh - patch_size[0]:        
            range_y = np.append(range_y,imhigh - patch_size[0])        
        if range_x[-1] != imwidth - patch_size[1]:        
            range_x = np.append(range_x,imwidth - patch_size[1])
        
        sz = len(range_y) * len(range_x) ## 图像块的数量
        
        if len(image.shape) == 2:        
        ## 初始化灰度图像        
            res = np.zeros((sz,patch_size[0],patch_size[1]))        
        if len(image.shape) == 3:        
        ## 初始化RGB图像       
            res = np.zeros((sz,patch_size[0],patch_size[1],imch))
        
        index = 0        
        for y in range_y:        
           for x in range_x:        
               patch = image[y:y+patch_size[0],x:x+patch_size[1]]        
               res[index] = patch        
               index = index + 1
        
        return res


def col2image(coldata,imsize,stride):

        """       
        coldata: 使用image2cols得到的数据        
        imsize:原始图像的宽和高，如(321, 481)       
        stride:图像切分时的步长，如10        
        """
        
        patch_size = coldata.shape[1:3]      
        if len(coldata.shape) == 3:        
        ## 初始化灰度图像       
            res = np.zeros((imsize[0],imsize[1]))        
            w = np.zeros(((imsize[0],imsize[1])))        
        if len(coldata.shape) == 4:        
        ## 初始化RGB图像        
            res = np.zeros((imsize[0],imsize[1],3))        
            w = np.zeros(((imsize[0],imsize[1],3)))

        if imsize[0] == patch_size[0]:
            range_y = [0]
        else:        
           range_y = np.arange(0,imsize[0] - patch_size[0],stride[0])       
        range_x = np.arange(0,imsize[1] - patch_size[1],stride[1])
        
        if range_y[-1] != imsize[0] - patch_size[0]:        
            range_y = np.append(range_y,imsize[0] - patch_size[0])        
        if range_x[-1] != imsize[1] - patch_size[1]:       
            range_x = np.append(range_x,imsize[1] - patch_size[1])
            
        index = 0        
        for y in range_y:       
          for x in range_x:        
            res[y:y+patch_size[0],x:x+patch_size[1]] = res[y:y+patch_size[0],x:x+patch_size[1]] + coldata[index]            
            w[y:y+patch_size[0],x:x+patch_size[1]] = w[y:y+patch_size[0],x:x+patch_size[1]] + 1            
            index = index + 1
        
        return res / w


def generate_gaussian_weighted_matrix(m, n, ones_columns, sigma):
    """
    Generate a matrix of shape (m, n) with ones in the specified columns and
    other values based on Gaussian distance to the nearest column with a one.

    :param m: Number of rows
    :param n: Number of columns
    :param ones_columns: List of column indices where the values should be 1
    :param sigma: Standard deviation for the Gaussian function
    :return: Generated matrix
    """
    # Initialize the matrix with zeros
    matrix = np.zeros((m, n))

    # Set the specified columns to 1
    for col in ones_columns:
        matrix[:, col] = 1

    # Calculate values for other columns based on Gaussian distance to nearest '1' column
    for i in range(m):
        for j in range(n):
            if j not in ones_columns:
                min_distance = min([abs(j - col) for col in ones_columns])
                matrix[i, j] = np.exp(-min_distance ** 2 / (2 * sigma ** 2))

    return matrix


def DIFFZ(z, device=None, dtype=None):
    """
    计算阻抗的空间梯度，得到反射系数
    输入: z - 阻抗数据 [batch, channel, time, space]
    输出: DZ - 反射系数 [batch, channel, time, space]
    """
    if device is None:
        device = z.device
    if dtype is None:
        dtype = z.dtype
    DZ = torch.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]], device=device, dtype=dtype)
    DZ[..., :-1, :] = 0.5 * (z[..., 1:, :] - z[..., :-1, :])
    return DZ

def tv_loss(x, alfa=1.0):
    """
    总变分正则化损失，保持空间连续性
    输入: x - 预测阻抗 [batch, channel, time, space]
          alfa - 正则化权重
    输出: TV损失值
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])    # 水平梯度
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])    # 垂直梯度
    return alfa * torch.mean(2*dh[..., :-1, :] + dw[..., :, :-1])

def wavelet_init(wavelet_length):
    """
    从地震数据估计初始子波
    输入: seismic_data - 地震数据
          wavelet_length - 子波长度
    输出: 估计的初始子波
    """

    dt = 0.001
    t0=wavelet_length/2*dt
    t = np.arange(wavelet_length) * dt 
    f0 = 30  # 主频30Hz
    wav = (1 - 2*np.pi**2*f0**2*(t-t0)**2) * np.exp(-np.pi**2*f0**2*(t-t0)**2)

    # plt.plot(wav)
    # plt.legend()
    # plt.show()
    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# temp_wav = wavelet_init(257).squeeze().numpy()

def average_smoothing(signal, kernel_size):
    smoothed_signal = []
    for i in range(1):
        kernel = torch.ones(1, 1, 1, kernel_size)
        smoothed_signal.append(F.conv2d(signal[:, i:i + 1, ...], kernel, padding='same'))
    smoothed_signal = torch.cat(smoothed_signal, dim=1)

    return smoothed_signal


def save_stage1_loss_data(save_dir, total_lossF):
    """
    保存阶段1（子波矫正器）的loss数据并生成可视化图表
    
    Args:
        save_dir: 保存目录路径
        total_lossF: 阶段1的loss列表
    """
    # 保存loss数据
    stage1_loss_data = {
        'epochs': list(range(len(total_lossF))),
        'wavelet_loss': total_lossF
    }
    np.save(os.path.join(save_dir, 'stage1_loss_data.npy'), stage1_loss_data)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(total_lossF)), total_lossF, 'b-', linewidth=2, label='Wavelet Correction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Stage 1: Wavelet Correction Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stage1_wavelet_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Stage 1 loss data saved: {save_dir}/stage1_loss_data.npy")
    print(f"📊 Stage 1 loss plot saved: {save_dir}/stage1_wavelet_loss.png")


def save_stage2_loss_data(save_dir, stage2_total_loss, stage2_sup_loss, 
                         stage2_unsup_loss, stage2_tv_loss,stage2_imp_loss):
    """
    保存阶段2（UNet阻抗反演）的loss数据并生成可视化图表
    
    Args:
        save_dir: 保存目录路径
        stage2_total_loss: 总损失列表
        stage2_sup_loss: 井约束损失列表
        stage2_unsup_loss: 物理约束损失列表
        stage2_tv_loss: TV正则化损失列表
        admm_iter1: 训练轮次
    """
    # 保存loss数据
    epoch_number=len(stage2_total_loss)
    stage2_loss_data = {
        'epochs': list(range(epoch_number)),
        'total_loss': stage2_total_loss,
        'supervised_loss': stage2_sup_loss,
        'unsupervised_loss': stage2_unsup_loss,
        'tv_loss': stage2_tv_loss,
        'imp_loss': stage2_imp_loss
    }
    np.save(os.path.join(save_dir, 'stage2_loss_data.npy'), stage2_loss_data)
    # pdb.set_trace()
    # 绘制阶段2的详细loss曲线
    plt.figure(figsize=(15, 10))
    
    # 子图1：总损失
    plt.subplot(2, 2, 1)
    plt.plot(range(epoch_number), stage2_total_loss, 'r-', linewidth=2, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Stage 2: UNet Inversion Total Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：各项损失对比
    plt.subplot(2, 2, 2)
    plt.plot(range(epoch_number), stage2_sup_loss, 'g-', linewidth=2, label='Supervised Loss')
    plt.plot(range(epoch_number), stage2_unsup_loss, 'b-', linewidth=2, label='Unsupervised Loss')
    plt.plot(range(epoch_number), stage2_tv_loss, 'orange', linewidth=2, label='TV Regularization Loss')
    plt.plot(range(epoch_number), stage2_imp_loss, 'purple', linewidth=2, label='Impedance(fake) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Stage 2: Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：井约束损失
    plt.subplot(2, 2, 3)
    plt.plot(range(epoch_number), stage2_sup_loss, 'g-', linewidth=2, label='Supervised Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Supervised Loss (Well Constraint)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4：物理约束损失
    plt.subplot(2, 2, 4)
    plt.plot(range(epoch_number), stage2_unsup_loss, 'b-', linewidth=2, label='Unsupervised Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Unsupervised Loss (Forward Consistency)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stage2_unet_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 阶段2损失数据已保存: {save_dir}/stage2_loss_data.npy")
    print(f"📊 阶段2损失曲线已保存: {save_dir}/stage2_unet_loss.png")

def save_complete_training_loss(save_dir, total_lossF, stage2_total_loss, 
                               stage2_sup_loss, stage2_unsup_loss, stage2_tv_loss,stage2_imp_loss):
    """
    保存完整训练过程（阶段1+阶段2）的loss对比图
    
    Args:
        save_dir: 保存目录路径
        total_lossF: 阶段1的loss列表
        stage2_total_loss: 阶段2总损失列表
        stage2_sup_loss: 阶段2井约束损失列表
        stage2_unsup_loss: 阶段2物理约束损失列表
        stage2_tv_loss: 阶段2TV正则化损失列表
        admm_iter: 阶段1训练轮次
        admm_iter1: 阶段2训练轮次
    """
    plt.figure(figsize=(15, 6))
    
    admm_iter=len(total_lossF)
    admm_iter1=len(stage2_total_loss)
    # 阶段1损失
    plt.subplot(1, 2, 1)
    plt.plot(range(admm_iter), total_lossF, 'b-', linewidth=2, label='Wavelet Correction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Stage 1: Wavelet Correction Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 阶段2损失
    plt.subplot(1, 2, 2)
    plt.plot(range(admm_iter1), stage2_total_loss, 'r-', linewidth=2, label='Total Loss')
    plt.plot(range(admm_iter1), stage2_sup_loss, 'g-', linewidth=2, label='Supervised Loss')
    plt.plot(range(admm_iter1), stage2_unsup_loss, 'b-', linewidth=2, label='Unsupervised Loss')
    plt.plot(range(admm_iter1), stage2_tv_loss, 'orange', linewidth=2, label='TV Regularization Loss')
    plt.plot(range(admm_iter1), stage2_imp_loss, 'p-', linewidth=2, label='Fake imp Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Stage 2: UNet Inversion Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complete_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Complete training loss plot saved: {save_dir}/complete_training_loss.png")

def plot_loss_comparison(loss_data, save_dir, title="Training Loss Comparison Analysis"):
    if not loss_data:
        print("⚠️  没有可用的loss数据")
        return
    plt.figure(figsize=(15, 10))
    n_plots = len(loss_data)
    if n_plots == 1:
        cols = 1
        rows = 1
    elif n_plots == 2:
        cols = 2
        rows = 1
    else:
        cols = 2
        rows = (n_plots + 1) // 2
    plot_idx = 1
    if 'stage1' in loss_data:
        plt.subplot(rows, cols, plot_idx)
        stage1_data = loss_data['stage1']
        plt.plot(stage1_data['epochs'], stage1_data['wavelet_loss'], 'b-', linewidth=2, label='Wavelet Correction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Stage 1: Wavelet Correction Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_idx += 1
    if 'stage2' in loss_data:
        stage2_data = loss_data['stage2']
        
        # 总损失
        plt.subplot(rows, cols, plot_idx)
        plt.plot(stage2_data['epochs'], stage2_data['total_loss'], 'r-', linewidth=2, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Stage 2: UNet Inversion Total Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_idx += 1
        if plot_idx <= rows * cols:
            plt.subplot(rows, cols, plot_idx)
            plt.plot(stage2_data['epochs'], stage2_data['supervised_loss'], 'g-', linewidth=2, label='Supervised Loss')
            plt.plot(stage2_data['epochs'], stage2_data['unsupervised_loss'], 'b-', linewidth=2, label='Unsupervised Loss')
            plt.plot(stage2_data['epochs'], stage2_data['tv_loss'], 'orange', linewidth=2, label='TV Regularization Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.title('Stage 2: Loss Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Loss comparison analysis plot saved: {save_dir}/loss_comparison_analysis.png")



