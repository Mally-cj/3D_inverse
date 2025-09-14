from lianjin import *
from data_tools import single_imshow
import numpy as np
    
base_line = 450   # Line起始值
base_cmp = 212    # CMP起始值
# 原始井位坐标 (Line, CMP)
pos = [  [532,1053], [561,842],[594,295],  [603,1212],[572,692], [591,996],
[504,846], [499,597]]
# 转换为相对坐标 (inline_idx, xline_idx)
well_positions = [[line-base_line-1, cmp-base_cmp-1] for [line, cmp] in pos]
# 生成折线轨迹
polyline_points,origin_pos = generate_polyline_points(well_positions)

##可视化
visual_trajectory(polyline_points,origin_points=pos)

# folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11_0.25_0.25_1_0.1_4"
folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11_1_1_0.5_0_4"

name=folder.split('/')[-1]
pred_3d=np.load(os.path.join(folder,"prediction_impedance.npy"))  ##601*1189*251
true_3d=np.load(os.path.join(folder,"true_impedance.npy"))
seis_3d =np.load(os.path.join(folder,"seismic_record.npy"))
low_3d =np.load(os.path.join(folder,"implow_impedance.npy"))


x_coords = [p[0] for p in polyline_points]
y_coords = [p[1] for p in polyline_points]

imp = true_3d[:, y_coords, x_coords]
seis=seis_3d[:, y_coords, x_coords]
low=low_3d[:, y_coords, x_coords]

mask=np.zeros_like(imp)
# mask[:,::50]=1
for idx,v in enumerate(origin_pos):
    if v!=0:
        mask[:,idx]=1

indices=[]

def caijian(img):
    patchszie=200
    length=img.shape[1]//200
    img=img[:,:length*200]
    img=img.reshape(601,length,1,200).transpose(1,2,0,3)
    return img


for i in range(0,imp.shape[1]-200,200):
    indices.append((i, i+200))




# from scipy.signal import filtfilt
# from scipy import signal
# B, A = signal.butter(2, 0.012, 'low')  # 截止频率约12Hz
# m_loww = signal.filtfilt(B, A, imp.T).T
# nsmooth = 3
# m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_loww)  # 时间方向平滑
# nsmooth = 3
# m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_low.T).T  # 空间方向平滑


imp_np=caijian(imp)
seis_np=caijian(seis)
mask_np=caijian(mask)
low_np=caijian(low)
print(imp_np.shape)
print(seis_np.shape)
print(mask_np.shape)


# np.save(os.path.join(folder,"imp_np.npy"),imp_np)

np.savez('/home/shendi_gjh_cj/codes/3D_project/data/new_traindata0914.npz',imp_np=imp_np,seis_np=seis_np,mask_np=mask_np,low_np=low_np,indices=indices)

# single_imshow(img[2,0])
# print(img.shape)