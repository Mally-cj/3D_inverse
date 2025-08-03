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
import sys
sys.path.append('deep_learning_impedance_inversion_chl')
from cpp_to_py import generate_well_mask as generate_well_mask2
from cpp_to_py import get_wellline_and_mask as get_wellline_and_mask2
from Model.utils import image2cols
from Model.joint_well import add_labels
import data_tools as tools

class SeismicDataProcessor:
    """
    地震数据处理类
    支持数据加载、预处理、缓存和训练数据构建
    """

    def __init__(self, cache_dir='cache', device='auto'):
        """
        初始化数据处理器

        Args:
            cache_dir: 缓存目录
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 设备配置
        if device == 'auto':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 根据设备自动调整参数
        # if self.device.type == 'cuda':
        self.dtype = torch.cuda.FloatTensor
        self.config = {
            'BATCH_SIZE': 60,
            'PATCH_SIZE': 120,
            'N_WELL_PROFILES': 30
        }
        # else:
        #     self.dtype = torch.FloatTensor
        #     self.config = {
        #         'BATCH_SIZE': 1,
        #         'PATCH_SIZE': 48,
        #         'N_WELL_PROFILES': 10
        #     }

        # 数据缓存
        self._data_cache = {}
        self._normalization_params = {}

        print(f"🚀 数据处理器初始化完成:")
        print(f"   - 设备: {self.device}")
        print(f"   - 缓存目录: {cache_dir}")
        print(f"   - 配置: {self.config}")

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
            print(f"📦 从缓存加载: {cache_key}")
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
        segy = _read_segy(file_path)
        impedance_model_full = []

        for i in range(0, len(segy.traces)):
            impedance_model_full.append(segy.traces[i].data)

        impedance_model_full = np.array(impedance_model_full).reshape(
            251, len(impedance_model_full)//251, 601
        ).transpose(2, 1, 0)

        # 根据设备配置调整数据大小
        impedance_model_full = impedance_model_full

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

    def generate_well_mask(self, S_obs):
        """
        生成井位掩码

        Args:
            S_obs: 观测地震数据

        Returns:
            well_pos: 井位坐标
            M_well_mask: 井位掩码
            M_well_mask_dict: 井位掩码字典
        """
        cache_key = self._get_cache_key("well_mask",
                                      shape=S_obs.shape[1:3],
                                      full_data=True)

        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        print("🎯 生成井位掩码...")

        # 网格参数
        nx, ny = S_obs.shape[1:3]
        basex = 450
        basey = 212

        # 定义井位
        pos = [[594,295], [572,692], [591,996], [532,1053],
               [603,1212], [561,842], [504,846], [499,597]]
        well_pos = [[y-basey, x-basex] for [x, y] in pos]

        # 生成井位掩码
        grid_shape = S_obs.shape[1:3]
        M_well_mask_dict = generate_well_mask2(well_pos, grid_shape, well_range=15, sigma=5)

        # 转换为2D数组格式
        M_well_mask = np.zeros(grid_shape)
        for (line, cmp), weight in M_well_mask_dict.items():
            M_well_mask[line, cmp] = weight

        result = (well_pos, M_well_mask, M_well_mask_dict)

        # 保存到缓存
        self._save_to_cache(cache_key, result)

        print(f"✅ 井位掩码生成完成:")
        print(f"   - 井位数量: {len(well_pos)}")
        print(f"   - 掩码形状: {M_well_mask.shape}")

        return result

    def build_training_profiles(self, Z_back, impedance_model_full, S_obs,
                              well_pos, M_well_mask_dict):
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
        oversize = 5

        Z_back_patches = []
        Z_full_patches = []
        S_obs_patches = []
        M_mask_patches = []

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

        # 拼接所有训练块
        Z_back_train_set = torch.cat(Z_back_patches, 0)[..., None].permute(0, 3, 1, 2).type(self.dtype)
        Z_full_train_set = torch.cat(Z_full_patches, 0)[..., None].permute(0, 3, 1, 2).type(self.dtype)
        S_obs_train_set = torch.cat(S_obs_patches, 0)[..., None].permute(0, 3, 1, 2).type(self.dtype)
        M_mask_train_set = torch.cat(M_mask_patches, 0)[..., None].permute(0, 3, 1, 2).type(self.dtype)

        training_data = {
            'Z_back_train_set': Z_back_train_set,
            'Z_full_train_set': Z_full_train_set,
            'S_obs_train_set': S_obs_train_set,
            'M_mask_train_set': M_mask_train_set
        }

        # 保存到缓存
        self._save_to_cache(cache_key, training_data)

        print(f"✅ 训练剖面数据构建完成:")
        print(f"   - 训练样本总数: {len(S_obs_train_set)}")
        print(f"   - 每个样本大小: {S_obs_train_set.shape[2]}×{S_obs_train_set.shape[3]}")

        return training_data



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
        # 1. 加载阻抗数据
        impedance_model_full = self.load_impedance_data()
        # 2. 生成低频背景
        Z_back = self.generate_low_frequency_background(impedance_model_full)
        # 3. 加载地震数据
        S_obs = self.load_seismic_data()
        # 4. 生成井位掩码
        well_pos, M_well_mask, M_well_mask_dict = self.generate_well_mask(S_obs)
        # 5. 构建训练剖面数据
        training_data = self.build_training_profiles(
            Z_back, impedance_model_full, S_obs, well_pos, M_well_mask_dict
        )
        # 6. 数据归一化（直接写在此处）
        logimpmax = impedance_model_full.max()
        logimpmin = impedance_model_full.min()
        S_obs_min = S_obs.min()
        S_obs_max = S_obs.max()
        Z_full_norm = (training_data['Z_full_train_set'] - logimpmin) / (logimpmax - logimpmin)
        S_obs_norm = 2 * (training_data['S_obs_train_set'] - S_obs_min) / (S_obs_max - S_obs_min) - 1
        Z_back_norm = (training_data['Z_back_train_set'] - logimpmin) / (logimpmax - logimpmin)

        # 7. 创建训练数据加载器
        train_loader = data.DataLoader(
            data.TensorDataset(
                S_obs_norm,
                Z_full_norm,
                Z_back_norm,
                training_data['M_mask_train_set']
            ),
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True
        )
        # 数据信息
        data_info = {
            'impedance_shape': impedance_model_full.shape,
            'seismic_shape': S_obs.shape,
            'well_positions': well_pos,
            'config': self.config
        }
        normalization_params = {
            'logimpmax': logimpmax,
            'logimpmin': logimpmin,
            'S_obs_min': S_obs_min,
            'S_obs_max': S_obs_max
        }
        print("\n" + "="*80)
        print("✅ 训练数据处理流程完成")
        print("="*80)
        return train_loader, normalization_params, data_info



    def build_test_patches_regular(self, S_obs, Z_back, impedance_model_full, patch_size, oversize=70, axis=0):
        """
        沿axis方向滑窗切patch，另一个空间轴全保留。
        axis=0: x方向滑窗（inline），axis=1: y方向滑窗（xline）
        返回: patches, zback_patches, imp_patches, indices, shape3d
        """
        n_time, n_x, n_y = S_obs.shape
        spatial_shape = [n_x, n_y]
        patches, zback_patches, imp_patches, indices = [], [], [], []
        slide_len = spatial_shape[axis]
        keep_len = spatial_shape[1-axis]

        patch_size=min(patch_size,slide_len)
        # 生成滑窗位置和对应的indices
        start_begin=list(range(0, slide_len - patch_size + 1, oversize))
        start_begin+=[slide_len-patch_size]
        
        for start in start_begin:
            end = start + patch_size
            slc = [slice(None)] * 3
            slc[axis+1] = slice(start, end)
            patch = S_obs[tuple(slc)]
            zback_patch = Z_back[tuple(slc)]
            imp_patch = impedance_model_full[tuple(slc)]
            patches.append(torch.tensor(patch, dtype=torch.float32))
            zback_patches.append(torch.tensor(zback_patch, dtype=torch.float32))
            imp_patches.append(torch.tensor(imp_patch, dtype=torch.float32))

            # 为每个patch记录实际位置
            for j in range(keep_len):
                if axis == 0:
                    indices.append((start, j))  # x方向滑窗，y=j
                else:
                    indices.append((j, start))  # y方向滑窗，x=j

        # 将patch列表堆叠成张量，并统一成 [num_patches, 1, time, patch_size] 形状，方便后续DataLoader使用
        patches = torch.stack(patches).unsqueeze(1)         # [num_patches, 1, time, patch_size]
        zback_patches = torch.stack(zback_patches).unsqueeze(1)
        imp_patches = torch.stack(imp_patches).unsqueeze(1)
        shape3d = (n_time, n_x, n_y)
        return patches, zback_patches, imp_patches, indices, shape3d

    def process_test_data(self, axis=0,batch_size=500,patch_size=70,test_number=None):
        """
        返回测试patch loader、patch索引、shape3d、归一化参数，支持方向选择
        axis: 0(x方向滑窗/inline) 或 1(y方向滑窗/xline)
        """
        self.test_number=test_number
        impedance_model_full = self.load_impedance_data()
        tools.single_imshow(impedance_model_full[0])
        Z_back = self.generate_low_frequency_background(impedance_model_full)
        S_obs = self.load_seismic_data()
        logimpmax = impedance_model_full.max()
        logimpmin = impedance_model_full.min()
        S_obs_norm = 2 * (S_obs - S_obs.min()) / (S_obs.max() - S_obs.min()) - 1
        Z_back_norm = (Z_back - logimpmin) / (logimpmax - logimpmin)
        Z_full_norm = (impedance_model_full - logimpmin) / (logimpmax - logimpmin)
        # patch_size = self.config['PATCH_SIZE']
        self.test_axis=0
        patches, zback_patches, imp_patches, indices, shape3d = self.build_test_patches_regular(
            S_obs_norm, Z_back_norm, Z_full_norm, patch_size, patch_size-10, axis=self.test_axis
        )
        if test_number is None:
            test_number = len(patches)


        indices_tensor = torch.tensor(indices[:test_number], dtype=torch.int)
        test_loader = data.DataLoader(
            data.TensorDataset(patches[:test_number], imp_patches[:test_number], zback_patches[:test_number],indices_tensor[:test_number]),
            batch_size=batch_size, shuffle=False
        )
        normalization_params = {'logimpmax': logimpmax, 'logimpmin': logimpmin}

        return test_loader, indices, shape3d, normalization_params

    def reconstruct_3d_from_patches(self, pred_patches, indices):
        """
        pred_patches: patch列表
        indices: [(i, j)]
        """
        
        assert len(pred_patches) == len(indices), "pred_patches 和 indices 长度不匹配"
            
        # 获取patch尺寸
        n_time = pred_patches[0].shape[0]
        patch_size = pred_patches[0].shape[1]
        
        print(f"🔍 重建信息:")
        print(f"   - pred_patches 数量: {len(pred_patches)}")
        print(f"   - indices 数量: {len(indices)}")
        print(f"   - patch 形状: {pred_patches[0].shape}")
        print(f"   - test_axis: {self.test_axis}")
        
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
    processor = SeismicDataProcessor(cache_dir='cache')
    # train_loader, normalization_params, data_info = processor.process_train_data()
    test_loader, indices, shape3d, norm_params = processor.process_test_data()


    # 假设原始 3D 数据
    S_obs = np.random.rand(601, 1189, 251).astype(np.float32)
    Z_back = np.random.rand(601, 1189, 251).astype(np.float32)
    impedance_model_full = np.random.rand(601, 1189, 251).astype(np.float32)

    # 切分
    patches, zback_patches, imp_patches, indices, shape3d = processor.build_test_patches_regular(
        S_obs, Z_back, impedance_model_full, patch_size=500, oversize=70, axis=0
    )
    # pdb.set_trace()
    # 拼接
    reconstructed = processor.reconstruct_3d_from_patches(patches)

    # 验证
    assert reconstructed.shape == S_obs.shape, f"拼接后的形状 {reconstructed.shape} 与原始形状 {S_obs.shape} 不一致"
    print("切分和拼接逻辑匹配，数据一致！")

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