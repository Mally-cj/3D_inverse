import numpy as np
from scipy.io import savemat
import os

def npy_to_mat(npy_file_path, mat_file_path=None):
    """
    将npy文件转换为mat文件
    
    参数:
    npy_file_path: npy文件的路径
    mat_file_path: 输出的mat文件路径，默认为与npy文件同目录同名称的mat文件
    """
    # 如果未指定mat文件路径，则使用与npy文件相同的路径和名称
    if mat_file_path is None:
        mat_file_path = os.path.splitext(npy_file_path)[0] + '.mat'
    
    try:
        # 加载npy文件
        data = np.load(npy_file_path)
        
        # 保存为mat文件，变量名默认为'data'
        savemat(mat_file_path, {'data': data})
        
        print(f"成功转换: {npy_file_path} -> {mat_file_path}")
        return True
    except Exception as e:
        print(f"转换失败 {npy_file_path}: {str(e)}")
        return False

def batch_convert_npy_to_mat(folder_path):
    """
    批量转换文件夹中的所有npy文件为mat文件
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            npy_file_path = os.path.join(folder_path, filename)
            npy_to_mat(npy_file_path)

if __name__ == "__main__":
    # 示例使用
    
    # 单个文件转换
    npy_to_mat('/home/shendi_gjh_cj/codes/3D_project/logs/E11_5_5_20/prediction_impedance.npy', 'output.mat')
    
    # 批量转换文件夹中的所有npy文件
    # batch_convert_npy_to_mat('./npy_files')
    
    # 请取消上面的注释并替换为你的文件或文件夹路径
    print("请修改脚本中的文件路径后再运行")
    