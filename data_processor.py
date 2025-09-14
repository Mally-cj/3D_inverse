"""
地震阻抗反演数据处理模块
支持数据加载、预处理、缓存机制和训练数据构建
"""

import os
import pickle
import pdb
import hashlib
import numpy as np
import torch
from torch.utils import data
from scipy.signal import filtfilt
from scipy import signal
from obspy.io.segy.segy import _read_segy
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('deep_learning_impedance_inversion_chl')
from cpp_to_py import generate_well_mask as generate_well_mask2
from cpp_to_py import get_wellline_and_mask as get_wellline_and_mask2
from utils import image2cols
from Model.joint_well import add_labels
import data_tools as tools
import pdb
from icecream import ic

class SeismicDataProcessor:
    """
    地震数据处理类
    支持数据加载、预处理、缓存和训练数据构建
    """

    def __init__(self, cache_dir='cache', device=None,type='train',train_batch_size=60,train_patch_size=120,
    N_WELL_PROFILES=60,test_axis=0,norm_method='mean_std'):
        """
        初始化数据处理器

        Args:
            train_batch_size: 训练批量大小
            train_patch_size: 训练patch大小
            N_WELL_PROFILES :生成的连井剖面个数,再根据patch_size切分
            cache_dir: 缓存目录
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.test_axis=test_axis
        self.norm_method=norm_method

        if device is not None:
            self.set_device(device)
        

        if type == 'train':
            self.config = {
                'BATCH_SIZE': train_batch_size,
                'PATCH_SIZE': train_patch_size,
                'N_WELL_PROFILES': N_WELL_PROFILES
            }


        # 数据缓存
        self._data_cache = {}
        self._normalization_params = {}

        print(f"🚀 数据处理器初始化完成:")
        print(f"   - 设备: {self.device}")
        print(f"   - 缓存目录: {cache_dir}")
        print(f"   - 配置: {self.config}")
    def get_device(self):
        ##如果存在self.device，则返回self.device，否则提示要set_device
        if hasattr(self, 'device'):
            return self.device
        else:
            raise ValueError("请先设置设备 self.set_device()")

    def set_device(self,device):
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def _get_cache_key(self, data_type, **kwargs):
        """生成缓存键"""
        key_parts = [data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{v}")
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key):
        """从缓存加载数据"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            print(f"📦 从缓存{cache_file}加载: {cache_key}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key, data):
        """保存数据到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        print(f"💾 保存到缓存: {cache_key}")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_impedance_data(self, file_path="data/yyf_smo_train_Volume_PP_IMP.sgy"):
        """
        加载阻抗数据

        Args:
            file_path: 阻抗数据文件路径

        Returns:
            impedance_model_full: 完整阻抗数据
        """
        cache_key = self._get_cache_key("impedance",
                                      full_data=True,
                                      max_slices=251)

        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        print(f"🔄 加载阻抗数据: {file_path}")
        ##如果找不到file_path，就加绝对路径前缀
        if not os.path.exists(file_path):
            file_path = os.path.join("/home/shendi_gjh_cj/codes/3D_project", file_path)
        segy = _read_segy(file_path)
        impedance_model_full = []

        for i in range(0, len(segy.traces)):
            impedance_model_full.append(segy.traces[i].data)

        impedance_model_full = np.array(impedance_model_full).reshape(
            251, len(impedance_model_full)//251, 601
        ).transpose(2, 1, 0)

        # impedance_model_full=np.clip(impedance_model_full,6000,15000)
        # pdb.set_trace()

        # 根据设备配置调整数据大小
        impedance_model_full = impedance_model_full
        # pdb.set_trace()
        
        impedance_model_full = np.log(impedance_model_full)

        # 保存到缓存
        self._save_to_cache(cache_key, impedance_model_full)

        print(f"✅ 阻抗数据加载完成: {impedance_model_full.shape}")
        return impedance_model_full

    def generate_low_frequency_background(self, impedance_model_full):
        """
        从完整阻抗数据生成低频背景

        Args:
            impedance_model_full: 完整阻抗数据

        Returns:
            Z_back: 低频背景阻抗
        """
        cache_key = self._get_cache_key("low_freq_background",
                                      shape=impedance_model_full.shape)

        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        print("🌊 生成低频背景阻抗...")
        Z_back = []

        for i in range(impedance_model_full.shape[2]):
            B, A = signal.butter(2, 0.012, 'low')  # 截止频率约12Hz
            m_loww = signal.filtfilt(B, A, impedance_model_full[..., i].T).T
            nsmooth = 3
            m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_loww)  # 时间方向平滑
            nsmooth = 3
            m_low = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, m_low.T).T  # 空间方向平滑
            Z_back.append(m_low[..., None])

        Z_back = np.concatenate(Z_back, axis=2)

        # 保存到缓存
        self._save_to_cache(cache_key, Z_back)

        print(f"✅ 低频背景阻抗生成完成: {Z_back.shape}")
        return Z_back

    def load_seismic_data(self, file_path="data/PSTM_resample1_lf_extension2.sgy"):
        """
        加载地震观测数据

        Args:
            file_path: 地震数据文件路径

        Returns:
            S_obs: 观测地震数据
        """
        cache_key = self._get_cache_key("seismic",
                                      full_data=True,
                                      max_slices=251)

        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        print(f"🌊 加载地震观测数据: {file_path}")
        if not os.path.exists(file_path):
            file_path = os.path.join("/home/shendi_gjh_cj/codes/3D_project", file_path)
        segy_seismic = _read_segy(file_path)
        S_obs = []

        for i in range(0, len(segy_seismic.traces)):
            S_obs.append(segy_seismic.traces[i].data)

        S_obs = np.array(S_obs).reshape(251, len(S_obs)//251, 601).transpose(2, 1, 0)

        # 根据设备配置调整数据大小
        S_obs = S_obs

        # 保存到缓存
        self._save_to_cache(cache_key, S_obs)

        print(f"✅ 地震观测数据加载完成: {S_obs.shape}")
        return S_obs

    def generate_well_mask(self, shape_3d):
        """
        生成井位掩码

        Args:
            shape_3d: 3D数据形状

        Returns:
            well_pos: 井位坐标
            M_well_mask: 井位掩码
            M_well_mask_dict: 井位掩码字典
        """
        cache_key = self._get_cache_key("well_mask",
                                      shape=shape_3d[1:3],
                                      full_data=True)

        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        print("🎯 生成井位掩码...")

        # 网格参数
        nx, ny = shape_3d[1:3]
        basex = 450
        basey = 212

        # # 定义井位
        # pos = [[594,295], [572,692], [591,996], [532,1053],
        #        [603,1212], [561,842], [504,846], [499,597]]
        pos = [[594,295], [572,692], [591,996], [532,1053],
               [603,1212], [561,842]]
        well_pos = [[y-basey, x-basex] for [x, y] in pos]

        # 生成井位掩码
        grid_shape = shape_3d[1:3]
        M_well_mask_dict = generate_well_mask2(well_pos, grid_shape, well_range=15, sigma=5)

        # 转换为2D数组格式
        M_well_mask = np.zeros(grid_shape)
        for (line, cmp), weight in M_well_mask_dict.items():
            M_well_mask[line, cmp] = weight
        
        ##保存M_well_mask为npy文件
        np.save("/home/shendi_gjh_cj/codes/3D_project/mask_grid.npy",M_well_mask)

        result = (well_pos, M_well_mask, M_well_mask_dict)

        # 保存到缓存
        self._save_to_cache(cache_key, result)

        print(f"✅ 井位掩码生成完成:")
        print(f"   - 井位数量: {len(well_pos)}")
        print(f"   - 掩码形状: {M_well_mask.shape}")

        return result

    def build_training_profiles(self,well_pos, M_well_mask_dict):
        """
        构建训练剖面数据

        Args:
            Z_back: 低频背景阻抗
            impedance_model_full: 完整阻抗数据
            S_obs: 观测地震数据
            well_pos: 井位坐标
            M_well_mask_dict: 井位掩码字典

        Returns:
            training_data: 训练数据字典
        """
        cache_key = self._get_cache_key("training_profiles",
                                      n_profiles=self.config['N_WELL_PROFILES'],
                                      patch_size=self.config['PATCH_SIZE'])

        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data


        print("📦 构建训练剖面数据...")

        # 1. 加载3D阻抗数据 601*1189*251
        impedance_model_full = self.load_impedance_data()           
        # 2. 生成3D低频背景 601*1189*251
        
        Z_back = self.generate_low_frequency_background(impedance_model_full)   
        # 3. 加载3D地震数据 601*1189*251
        S_obs = self.load_seismic_data()

        # 训练井位
        train_well = add_labels(well_pos)
        grid_shape = S_obs.shape[1:3]

        # 存储各类剖面数据
        Z_back_profiles = []
        Z_full_profiles = []
        S_obs_profiles = []
        M_mask_profiles = []
        path_coords = []

        print("   正在生成连井剖面...")
        for i in tqdm(range(self.config['N_WELL_PROFILES']), desc="生成剖面"):
            # 生成随机连井剖面
            interpolated_points, vMask = get_wellline_and_mask2(
                well_pos, grid_shape, M_well_mask_dict
            )

            path_coords.append(interpolated_points)

            # 扩展掩码到时间维度
            vMask_time_extended = np.tile(vMask, (601, 1))
            M_mask_profiles.append(vMask_time_extended)

            # 提取沿剖面的数据
            Z_back_profiles.append(Z_back[:, interpolated_points[:, 0], interpolated_points[:, 1]])
            Z_full_profiles.append(impedance_model_full[:, interpolated_points[:, 0], interpolated_points[:, 1]])
            S_obs_profiles.append(S_obs[:, interpolated_points[:, 0], interpolated_points[:, 1]])

        # 滑窗切分统一尺寸
        patchsize = self.config['PATCH_SIZE']
        oversize = patchsize-10

        Z_back_patches = []
        Z_full_patches = []
        S_obs_patches = []
        M_mask_patches = []
        pdb.set_trace()


        print("   正在切分训练块...")
        for i in tqdm(range(self.config['N_WELL_PROFILES']), desc="切分数据"):
            Z_back_patches.append(torch.tensor(image2cols(
                Z_back_profiles[i], (S_obs.shape[0], patchsize), (1, oversize)
            )))
            Z_full_patches.append(torch.tensor(image2cols(
                Z_full_profiles[i], (S_obs.shape[0], patchsize), (1, oversize)
            )))
            S_obs_patches.append(torch.tensor(image2cols(
                S_obs_profiles[i], (S_obs.shape[0], patchsize), (1, oversize)
            )))
            M_mask_patches.append(torch.tensor(image2cols(
                M_mask_profiles[i], (S_obs.shape[0], patchsize), (1, oversize)
            )))


        # pdb.set_trace()
        # 把列表内容拼接成numpy，列表中的内容是163*601*120*120，拼接后是5553*1*601*120
        Z_back_patches_np = np.concatenate(Z_back_patches, axis=0)[..., None].transpose(0, 3, 1, 2)
        Z_full_patches_np = np.concatenate(Z_full_patches, axis=0)[..., None].transpose(0, 3, 1, 2)
        S_obs_patches_np = np.concatenate(S_obs_patches, axis=0)[..., None].transpose(0, 3, 1, 2)
        M_mask_patches_np = np.concatenate(M_mask_patches, axis=0)[..., None].transpose(0, 3, 1, 2)





        # pdb.set_trace()
        ##剔除掉没有井位置的        
        mask = np.where(M_mask_patches_np.max(axis=(-2, -1))[:, 0] !=0)[0]
        Z_back_patches_np = Z_back_patches_np[mask]
        Z_full_patches_np = Z_full_patches_np[mask]
        S_obs_patches_np = S_obs_patches_np[mask]
        M_mask_patches_np = M_mask_patches_np[mask]





        training_data = {
            'Z_back_train_set': Z_back_patches_np,
            'Z_full_train_set': Z_full_patches_np,
            'S_obs_train_set': S_obs_patches_np,
            'M_mask_train_set': M_mask_patches_np,
            '3D_shape':impedance_model_full.shape
        }

        # 保存到缓存
        self._save_to_cache(cache_key, training_data)

        print(f"✅ 训练剖面数据构建完成:")
        print(f"   - 3D阻抗数据形状: {training_data['3D_shape']}")
        print(f"构建{len(Z_back_profiles)}个剖面，每个剖面形状为{Z_back_profiles[0].shape}")
        print(f"构建后的全部patch,整体形状为{Z_back_patches_np.shape}")

        return training_data


    def find_4d_outliers(self,arr: np.ndarray) -> tuple:
        """
        检测4维numpy数组中的异常值（3σ法则）
        
        参数:
            arr: 形状为(270, 1, 601, 120)的numpy数组
            
        返回:
            异常值的坐标和对应的值
        """
        # 计算整体均值和标准差
        mu, sigma = arr.mean(), arr.std()
        
        # 找到所有异常值的位置
        outlier_mask = np.abs(arr - mu) > 3 * sigma
        from icecream import ic
        # ic(outlier_mask.shape)
        ic(mu,sigma)
        ic(mu+3*sigma,mu-3*sigma)
        ic(arr.max(),arr.min())
        
        # 获取异常值的坐标和值
        outlier_coords = np.where(outlier_mask)
        outlier_values = arr[outlier_mask]
        
        return outlier_coords, outlier_values
    def process_train_data(self):
        """
        只处理训练数据
        Returns:
            train_loader: 训练数据加载器
            normalization_params: 归一化参数
            data_info: 数据信息字典
        """
        print("\n" + "="*80)
        print("🚀 开始训练数据处理流程")
        print("="*80)

        # 1. 加载3D地震数据 601*1189*251
        S_obs = self.load_seismic_data()      ##其实这里加载只是为了获取大小信息
        shape_3d=S_obs.shape

        # 4. 生成井位掩码
        well_pos, M_well_mask, M_well_mask_dict = self.generate_well_mask(shape_3d)
        # 5. 构建训练剖面数据
        training_data = self.build_training_profiles(
         well_pos, M_well_mask_dict
        )
        # coords, values = self.find_4d_outliers(training_data['S_obs_train_set'])
        # print(f"检测到 {len(values)} 个异常值:")
        # cnt=0
        # for coord, value in zip(zip(*coords), values):
            # print(f"位置: {coord}, 值: {value:.4f}")
            # cnt+=1
        # pdb.set_trace()
        # tools.single_imshow(S_obs[:,10])


        if self.norm_method == 'mean_std':
            logmean=training_data['Z_full_train_set'].mean()
            logstd=training_data['Z_full_train_set'].std()
            S_obs_mean=training_data['S_obs_train_set'].mean()
            S_obs_std=training_data['S_obs_train_set'].std()
            Z_full_norm = (training_data['Z_full_train_set'] - logmean) / logstd
            S_obs_norm=(training_data['S_obs_train_set'] - S_obs_mean) / S_obs_std
            Z_back_norm = (training_data['Z_back_train_set'] - logmean) / logstd
            normalization_params = {
            'logimpmean': logmean,
            'logimpstd': logstd,
            'S_obs_mean': S_obs_mean,
            'S_obs_std': S_obs_std
            }
        else:
            # 6. 数据归一化（直接写在此处）
            logimpmax = training_data['Z_full_train_set'].max()
            logimpmin = training_data['Z_full_train_set'].min()
            S_obs_min = training_data['S_obs_train_set'].min()
            S_obs_max = training_data['S_obs_train_set'].max()
            Z_full_norm = (training_data['Z_full_train_set'] - logimpmin) / (logimpmax - logimpmin)
            S_obs_norm=training_data['S_obs_train_set']/82000
            Z_back_norm = (training_data['Z_back_train_set'] - logimpmin) / (logimpmax - logimpmin)
            normalization_params = {
            'logimpmax': logimpmax,
            'logimpmin': logimpmin,
            'S_obs_min': S_obs_min,
            'S_obs_max': S_obs_max
            }
        print(normalization_params)

        # def DIFFZ(z):
        #     DZ=np.zeros([z.shape[0], z.shape[1], z.shape[2], z.shape[3]], dtype=np.float32)
        #     DZ[...,:-1,:] = 0.5*(z[...,1:, :] - z[..., :-1, :])
        #     return DZ
        # from codes.wi_inv_socket import custom_convmtx
        # from utils import wavelet_init
        # pdb.set_trace()
        # print(Z_full_norm.shape)
        # wav = wavelet_init(257).squeeze().numpy()
        # WW = custom_convmtx(wav, 601, 300)
        # WW = WW.astype(np.float32)
        # re_sesimic=WW@DIFFZ(Z_full_norm)
        # plt.plot(re_sesimic[0,0,:,141],label="re_sesimic")
        # plt.plot(S_obs_norm[0,0,:,10],label="S_obs_norm")
        # plt.legend()
        # plt.show()
        


        pdb.set_trace()
        # 7. 创建训练数据加载器
        # 将数据移动到指定设备
        train_loader = data.DataLoader(
            data.TensorDataset(
                torch.tensor(S_obs_norm, dtype=torch.float32, device=self.device),
                torch.tensor(Z_full_norm, dtype=torch.float32, device=self.device),
                torch.tensor(Z_back_norm, dtype=torch.float32, device=self.device),
                torch.tensor(training_data['M_mask_train_set'], dtype=torch.float32, device=self.device)
            ),
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False
        )
        # 数据信息
        data_info = {
            '3D_shape': shape_3d,
            'well_positions': well_pos,
            'config': self.config,
            'batch_shape':S_obs_norm.shape
        }

        print("\n" + "="*80)
        print("✅ 训练数据处理流程完成")
        print(f"数据集大小为{S_obs_norm.shape}")
        print("="*80)
        return train_loader, normalization_params, data_info


    def process_train_data2(self):
        data_load = np.load('/home/shendi_gjh_cj/codes/3D_project/data/new_traindata0914.npz')
        S_obs_norm = data_load['seis_np']
        Z_full_norm = data_load['imp_np']
        Z_back_norm = data_load['low_np']
        mask_np = data_load['mask_np']
        # pdb.set_trace()
        
        train_loader = data.DataLoader(
            data.TensorDataset(
                torch.tensor(S_obs_norm, dtype=torch.float32, device=self.device),
                torch.tensor(Z_full_norm, dtype=torch.float32, device=self.device),
                torch.tensor(Z_back_norm, dtype=torch.float32, device=self.device),
                torch.tensor(mask_np, dtype=torch.float32, device=self.device)
            ),
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False
        )

        print("\n" + "="*80)
        print("✅ 训练数据处理流程完成")
        print(f"数据集大小为{S_obs_norm.shape}")
        print("="*80)
        normalization_params={}
        data_info={}
        return train_loader, normalization_params, data_info

    def build_test_patches_regular(self, S_obs, Z_back, impedance_model_full, patch_size, oversize=70, axis=0):
        """
        沿axis方向滑窗切patch，生成2D剖面与训练数据维度一致。
        axis=0: x方向滑窗（inline），axis=1: y方向滑窗（xline）
        返回: patches, zback_patches, imp_patches, indices, shape3d
        """
        n_time, n_x, n_y = S_obs.shape
        spatial_shape = [n_x, n_y]
        patches, zback_patches, imp_patches, indices = [], [], [], []
        slide_len = spatial_shape[axis]
        keep_len = spatial_shape[1-axis]

        patch_size = min(patch_size, slide_len)
        # 生成滑窗位置
        start_begin = list(range(0, slide_len - patch_size + 1, oversize))
        start_begin += [slide_len - patch_size]
        
        for start in start_begin:
            # 沿另一个方向生成多个2D剖面
            for j in range(keep_len):
                if axis == 0:
                    # x方向滑窗，y=j，生成2D剖面 [601, patch_size]
                    patch = S_obs[:, start:start+patch_size, j]
                    zback_patch = Z_back[:, start:start+patch_size, j]
                    imp_patch = impedance_model_full[:, start:start+patch_size, j]
                    indices.append((start, j))
                else:
                    # y方向滑窗，x=j，生成2D剖面 [601, patch_size]
                    patch = S_obs[:, j, start:start+patch_size]
                    zback_patch = Z_back[:, j, start:start+patch_size]
                    imp_patch = impedance_model_full[:, j, start:start+patch_size]
                    indices.append((j, start))
                
                patches.append(torch.tensor(patch, dtype=torch.float32))
                zback_patches.append(torch.tensor(zback_patch, dtype=torch.float32))
                imp_patches.append(torch.tensor(imp_patch, dtype=torch.float32))
        
        # 将patch列表堆叠成张量，与训练数据维度一致 [num_patches, 1, time, patch_size]
        patches = torch.stack(patches).unsqueeze(1)
        zback_patches = torch.stack(zback_patches).unsqueeze(1)
        imp_patches = torch.stack(imp_patches).unsqueeze(1)
        
        shape3d = (n_time, n_x, n_y)
        return patches, zback_patches, imp_patches, indices, shape3d
    
    def process_test_data(self, batch_size=500, patch_size=70, test_number=None):
        """
        返回测试patch loader、patch索引、shape3d、归一化参数，支持方向选择
        axis: 0(x方向滑窗/inline) 或 1(y方向滑窗/xline)
        """
        self.test_number = test_number
        impedance_model_full = self.load_impedance_data()
        tools.single_imshow(impedance_model_full[0])

        Z_back = self.generate_low_frequency_background(impedance_model_full)
        S_obs = self.load_seismic_data()


        if self.norm_method == 'mean_std':
            logimpmean = impedance_model_full.mean()
            logimpstd = impedance_model_full.std()
            Z_back_norm = (Z_back - logimpmean) / logimpstd
            Z_full_norm = (impedance_model_full - logimpmean) / logimpstd
            seisc_mean=S_obs.mean()
            seisc_std=S_obs.std()
            S_obs_norm = (S_obs - seisc_mean) / seisc_std
            normalization_params = {
                'logimpmean': logimpmean,
                'logimpstd': logimpstd,
                'S_obs_mean': seisc_mean,
                'S_obs_std': seisc_std
            }
        else:
            logimpmax = impedance_model_full.max()
            logimpmin = impedance_model_full.min()
            S_obs_norm=S_obs/82000
            Z_back_norm = (Z_back - logimpmin) / (logimpmax - logimpmin)
            Z_full_norm = (impedance_model_full - logimpmin) / (logimpmax - logimpmin)
            normalization_params = {
                'logimpmax': logimpmax, 
                'logimpmin': logimpmin,
                'S_obs_min': S_obs.min(),
                'S_obs_max': S_obs.max()
            }
        print(normalization_params)
        print("S_obs.max()",S_obs.max())
        print("S_obs.min()",S_obs.min())
        # Z_full_norm=impedance_model_full
        # pdb.set_trace()
        # def DIFFZ(z):
        #     DZ=np.zeros(z.shape, dtype=np.float32)
        #     DZ[:-1,:,:] = 0.5*(z[1:,:,:] - z[:-1,:,:])
        #     return DZ
        # from codes.wi_inv_socket import custom_convmtx
        # from utils import wavelet_init

        # wav = wavelet_init(257).squeeze().numpy()
        # WW = custom_convmtx(wav, 601, len(wav)//2)
        # WW = WW.astype(np.float32)
        # tools.single_imshow(WW[:,:],title="WW")     ##*601
        
        # S= np.diag(0.5 * np.ones(601-1, dtype='float32'), k=1) - np.diag(
        #         0.5 * np.ones(601-1, dtype='float32'), -1)
        # S[0] = S[-1] = 0
        # WS=np.einsum('ij,jk->ik', WW, S)    ##601*601
        # re_sesimic=WS@Z_full_norm[:,:,82]  ##601*1189


        # inline=480
        # xline=122
        # inline,xline=480,122
        # inline,xline=83,144
        # inline,xline=784,141


        # re_sesimic=np.einsum('ij,jk->ik', WW, Z_full_norm[:,:,xline])  ##601*601 601*1189 
        # pdb.set_trace()
        # tools.single_imshow(re_sesimic)
        # tools.single_imshow(Z_full_norm[:,:,xline])
        # print(Z_full_norm[:,inline,xline])
        # print()

        # plt.plot(wav,label="wav")
        # plt.plot(Z_full_norm[:,inline,xline],label="Z_full_norm")
        # plt.title("well: inline="+str(inline)+",xline="+str(xline))
        # plt.show()


        # tem=(S@Z_full_norm[:,:,xline])[:,inline]
        # plt.plot(tem,label="S*Z_full_norm")
        # plt.title("inline="+str(inline)+",xline="+str(xline))
        # # pdb.set_trace()
        # plt.legend()
        # plt.show()
        # plt.legend()
        # plt.plot(re_sesimic[:,xline],label="re_sesimic")
        # plt.plot(S_obs_norm[:,inline,xline],label="S_obs_norm")
        # plt.legend()
        # plt.title("inline="+str(inline)+",xline="+str(xline))
        # plt.show()

        # print("S_obs.min()",S_obs.min())
        # print("S_obs.max()",S_obs.max())
        # print("re_sesimic.min()",re_sesimic.min())
        # print("re_sesimic.max()",re_sesimic.max())
        
        # 使用传入的axis参数
        patches, zback_patches, imp_patches, indices, shape3d = self.build_test_patches_regular(
            S_obs_norm, Z_back_norm, Z_full_norm, patch_size, patch_size-10, axis=self.test_axis
        )
        
        if test_number is None:
            test_number = len(patches)

        indices_tensor = torch.tensor(indices[:test_number], dtype=torch.int)
        test_loader = data.DataLoader(
            data.TensorDataset(patches[:test_number], imp_patches[:test_number], zback_patches[:test_number], indices_tensor[:test_number]),
            batch_size=batch_size, shuffle=False
        )
        return test_loader, indices, shape3d, normalization_params
        

    def process_test_data2(self, batch_size=500, patch_size=70, test_number=None):
        """
        返回测试patch loader、patch索引、shape3d、归一化参数，支持方向选择
        axis: 0(x方向滑窗/inline) 或 1(y方向滑窗/xline)
        """
        self.test_number = test_number

        data_loader=np.load('/home/shendi_gjh_cj/codes/3D_project/data/new_traindata0914.npz')
        S_obs_norm = data_loader['seis_np']
        Z_full_norm = data_loader['imp_np']
        Z_back_norm = data_loader['low_np']
        mask_np = data_loader['mask_np']
        indices = data_loader['indices']
    


        test_loader = data.DataLoader(
            data.TensorDataset(S_obs_norm, Z_full_norm, Z_back_norm, mask_np),
            batch_size=batch_size, shuffle=False
        )
        
        # 完整的归一化参数


        return test_loader, [], {}, {}

    def reconstruct_3d_from_patches(self, pred_patches, indices):
        """
        从2D剖面重建3D数据
        pred_patches: patch列表，每个patch形状为 [time, patch_size]
        indices: [(i, j)] 位置索引
        """
        
        assert len(pred_patches) == len(indices), "pred_patches 和 indices 长度不匹配"
        assert len(pred_patches[0].shape) == 2, "pred_patches 形状不正确"
        
        # 获取patch尺寸
        # print("pred_patches[0].shape", pred_patches[0].shape)
        n_time = pred_patches[0].shape[0]  # 时间维度
        patch_size = pred_patches[0].shape[1]  # 空间维度
        
        # print(f"🔍 重建信息:")
        # print(f"   - pred_patches 数量: {len(pred_patches)}")
        # print(f"   - indices 数量: {len(indices)}")
        # print(f"   - patch 形状: {pred_patches[0].shape}")
        # print(f"   - test_axis: {self.test_axis}")
        
        # 根据indices推断空间尺寸
        if self.test_axis == 0:
            max_i = max(idx[0] for idx in indices) + patch_size
            max_j = max(idx[1] for idx in indices) + 1
        else:
            max_i = max(idx[0] for idx in indices) + 1
            max_j = max(idx[1] for idx in indices) + patch_size
        n_x, n_y = max_i, max_j

        volume = np.zeros((n_time, n_x, n_y))
        count = np.zeros((n_time, n_x, n_y))

        for idx, (i, j) in enumerate(indices):
            patch = pred_patches[idx]  # [time, patch_size]
            
            if self.test_axis == 0:
                # x方向滑窗，y=j
                volume[:, i:i+patch_size, j] += patch
                count[:, i:i+patch_size, j] += 1
            else:
                # y方向滑窗，x=j
                volume[:, i, j:j+patch_size] += patch
                count[:, i, j:j+patch_size] += 1
                
        volume /= np.maximum(count, 1)
        return volume

if __name__ == "__main__":
    """测试数据处理模块"""
    # 创建数据处理器
    processor = SeismicDataProcessor(cache_dir='cache',device='cpu',train_batch_size=60,train_patch_size=120,norm_method='min_max')
    # train_loader, normalization_params, data_info = processor.process_train_data()
    test_loader, indices, shape3d, norm_params = processor.process_test_data()
    # S_obs = processor.load_seismic_data()      ##其实这里加载只是为了获取大小信息
    # shape_3d=S_obs.shape
    # print(shape_3d)
    
    # # 4. 生成井位掩码
    # well_pos, M_well_mask, M_well_mask_dict = processor.generate_well_mask(shape_3d)
    # # 5. 构建训练剖面数据
    # training_data = processor.build_training_profiles(
    #     well_pos, M_well_mask_dict
    # )
    # pdb.set_trace()
    ##读取train_loader的第1个数据
    
    # for idx, batch in enumerate(train_loader):
    #     print(idx)
    #     if idx ==3:
    #         S_obs_batch, Z_full_batch, Z_back_batch, M_mask_batch=batch
    #         break


    # for S_obs_batch, Z_full_batch, Z_back_batch, M_mask_batch in train_loader:
    #     print(S_obs_batch.shape)
    #     print(Z_full_batch.shape)
    #     print(Z_back_batch.shape)
    #     print(M_mask_batch.shape)
    #     break


    # # 假设原始 3D 数据
    # S_obs = np.random.rand(601, 1189, 251).astype(np.float32)
    # Z_back = np.random.rand(601, 1189, 251).astype(np.float32)
    # impedance_model_full = np.random.rand(601, 1189, 251).astype(np.float32)

    # # 切分
    # patches, zback_patches, imp_patches, indices, shape3d = processor.build_test_patches_regular(
    #     S_obs, Z_back, impedance_model_full, patch_size=500, oversize=70, axis=0
    # )
    # # 拼接
    # reconstructed = processor.reconstruct_3d_from_patches(patches)

    # # 验证
    # assert reconstructed.shape == S_obs.shape, f"拼接后的形状 {reconstructed.shape} 与原始形状 {S_obs.shape} 不一致"
    # print("切分和拼接逻辑匹配，数据一致！")

    # impedance_model_full = processor.load_impedance_data()
    # Z_back = processor.generate_low_frequency_background(impedance_model_full)
    # S_obs = processor.load_seismic_data()
    # logimpmax = impedance_model_full.max()
    # logimpmin = impedance_model_full.min()
    # S_obs_norm = 2 * (S_obs - S_obs.min()) / (S_obs.max() - S_obs.min()) - 1
    # Z_back_norm = (Z_back - logimpmin) / (logimpmax - logimpmin)
    # Z_full_norm = (impedance_model_full - logimpmin) / (logimpmax - logimpmin)
    # patch_size = processor.config['PATCH_SIZE']
    # processor.test_axis=0
    # patches, zback_patches, imp_patches, indices, shape3d = processor.build_test_patches_regular(
    #     S_obs_norm, Z_back_norm, Z_full_norm, patch_size=70, axis=0
    # )
    # %%
    # tools.single_imshow(impedance_model_full[:,:,0])
    # tools.single_imshow(Z_back[:,:,0])
    # tools.single_imshow(S_obs[:,:,0])
    # %%
    ##打印数据信息
    # print(f"\n📊 数据处理结果:")
    # print(f"   - 阻抗数据形状: {data_info['impedance_shape']}")
    # print(f"   - 地震数据形状: {data_info['seismic_shape']}")
    # print(f"   - 井位数量: {len(data_info['well_positions'])}")
    # print(f"   - 训练批数: {len(train_loader)}")
    # print(f"   - 测试批数: {len(test_loader)}")
    # print(f"   - 归一化参数: {norm_params}")