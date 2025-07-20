import numpy as np
import torch
from torch.nn import functional as F

def make_yushi_wavelet(nYushiFreqLow, nYushiFreqHigh, nWaveletSample, dt):       
    #p_par yushi 子波积分频率下界
    #q_par  子波积分频率上界
    #nsample：输出子波长度
    #s_interval:采样间隔
    p_par = nYushiFreqLow;
    q_par = nYushiFreqHigh;
    nsample = nWaveletSample;

    s_int = dt
    t = s_int*np.linspace(-(nsample//2),nsample//2,nsample);
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


def wavelet_init(syn, nw):
    dataf = torch.fft.fft(syn.permute(0, 1, 3, 2) - syn.mean(), dim=3)
    dataf_mean = torch.mean(torch.abs(dataf), dim=-2)[:, :, None, :]
    dataf_mean = average_smoothing(dataf_mean, 3)
    dataf_mean = dataf_mean / torch.max(dataf_mean, dim=3, keepdim=True)[0]
    dataif = torch.fft.ifft(dataf_mean, dim=3)
    dataif = torch.real(torch.fft.ifftshift(dataif, dim=3))
    wave = dataif / torch.max(dataif, dim=3, keepdim=True)[0]
    wave = torch.mean(wave, dim=0, keepdim=True)[..., None].permute(0, 4, 3, 2, 1)

    wavelet =  wave[:, :, int(wave.shape[2] / 2) - nw // 2:int(wave.shape[2] / 2) + nw // 2 + 1, :]

    return wavelet


def average_smoothing(signal, kernel_size):
    smoothed_signal = []
    for i in range(1):
        kernel = torch.ones(1, 1, 1, kernel_size)
        smoothed_signal.append(F.conv2d(signal[:, i:i + 1, ...], kernel, padding='same'))
    smoothed_signal = torch.cat(smoothed_signal, dim=1)

    return smoothed_signal



