import numpy as np
from icecream import ic
from data_tools import single_imshow
import os
import matplotlib.pyplot as plt
import torch

# 使用前面定义的 Bresenham 算法函数
def generate_line_points_integer(start, end):
    """
    使用 Bresenham 算法生成两点之间直线上的所有整数坐标点
    """
    x1, y1 = start
    x2, y2 = end
    
    if x1 == x2 and y1 == y2:
        return [start]
    
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    points.append((x, y))
    
    while x != x2 or y != y2:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        points.append((x, y))
    
    return points

def generate_polyline_points(points):
    """
    生成连接多个点的折线轨迹（整数坐标）
    
    参数:
    points -- 点列表，格式为 [(x1, y1), (x2, y2), ...]，所有坐标为整数
    
    返回:
    折线轨迹上所有整数坐标点的列表
    """
    if not points:
        return []
    
    # 存储所有轨迹点
    trajectory = []
    
    # 添加第一个点
    trajectory.append(points[0])
    
    # 依次连接所有点
    origin_pos=[]
    ic(len(points))
    for i in range(1, len(points)):
        start = points[i-1]
        end = points[i]
        # 使用 Bresenham 算法生成两点之间的整数坐标点
        segment_points = generate_line_points_integer(start, end)
        # 跳过第一个点（因为它已经是轨迹的最后一个点）

        origin_pos.append(i)
        trajectory.extend(segment_points[1:])
        origin_pos.extend([0]*(len(segment_points)-2))
        # origin_pos.append(len(trajectory)-1)

    origin_pos.append(len(points))
    return trajectory,origin_pos

def visual_trajectory(polyline_points,origin_points):
    print(f"轨迹为:{polyline_points}")
    # 可视化轨迹
    print("\n轨迹可视化:")
    max_x = max(p[0] for p in origin_points) + 1
    max_y = max(p[1] for p in origin_points) + 1
    
    # 创建网格
    grid = [['·' for _ in range(max_x)] for _ in range(max_y)]
    # 标记轨迹点
    for x, y in polyline_points:
        if 0 <= y < max_y and 0 <= x < max_x:
            grid[y][x] = 'o'
    # 标记原始点
    for i, (x, y) in enumerate(origin_points):
        if 0 <= y < max_y and 0 <= x < max_x:
            grid[y][x] = str(i)
    
    # # 打印网格 (y轴从上到下)
    # for row in reversed(grid):
    #     print(' '.join(row))

def add_well_to_3d(pred_3d,true_3d,well_positions,k=3):
    '''
    把井位置加入
    pred_3d:3D预测阻抗
    true_3d:3D真实阻抗
    well_positions:井位置
    k:周围k个点,为了让井位置加粗
    '''
    for x,y in well_positions:
        pred_3d[:,y,x]=true_3d[:,y,x]
        for a in range(-k,k):
            for b in range(-k,k):
                pred_3d[:,y+a,x+b]=true_3d[:,y,x]
    return pred_3d



# 示例用法
if __name__ == "__main__":
    # 定义点列表
    # points = [
    #     (0, 0),   # 点0
    #     (3, 2),   # 点1
    #     (5, 5),   # 点2
    #     (2, 7),   # 点3
    #     (0, 5)    # 点4
    # ]
    
    base_line = 450   # Line起始值
    base_cmp = 212    # CMP起始值
    # 原始井位坐标 (Line, CMP)
    pos = [  [532,1053], [561,842],[572,692],[594,295],  [603,1212], [591,996],
    
    [504,846], [499,597]]
    # 转换为相对坐标 (inline_idx, xline_idx)
    well_positions = [[line-base_line-1, cmp-base_cmp-1] for [line, cmp] in pos]
    # 生成折线轨迹
    polyline_points,origin_pos = generate_polyline_points(well_positions)
    
    ##可视化
    visual_trajectory(polyline_points,origin_points=pos)

    # folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11_0.25_0.25_1_0.1_4"
    folder="/home/shendi_gjh_cj/codes/3D_project/logs/E11_2_1_0.5_0_4"

    name=folder.split('/')[-1]
    pred_3d=np.load(os.path.join(folder,"prediction_impedance.npy"))  ##601*1189*251
    true_3d=np.load(os.path.join(folder,"true_impedance.npy"))
    seis_3d =np.load(os.path.join(folder,"seismic_record.npy"))

    fan_min= 7.9577527
    fan_max=9.507863
    seis2=true_3d.copy()

    true_3d=np.exp(true_3d*(fan_max-fan_min)+fan_min)
    pred_3d=np.exp(pred_3d*(fan_max-fan_min)+fan_min)

    x_coords = [p[0] for p in polyline_points]
    y_coords = [p[1] for p in polyline_points]

    img=add_well_to_3d(pred_3d,true_3d,well_positions,k=10)

    img = pred_3d[:, y_coords, x_coords]
    # ic(img.shape)

    single_imshow(img[250:500,:],title=f"{name}",vmin=5000,vmax=14000)



    # img2=add_well_to_3d(seis_3d,seis2,well_positions,k=10)

    img2 = seis_3d[:, y_coords, x_coords]
    # ic(img.shape)

    single_imshow(img2[250:500,:],title=f"{name}",cmap=plt.cm.seismic)