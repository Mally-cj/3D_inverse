import random
import math
import matplotlib.pyplot as plt
import numpy as np

# vector computation to calculate angle
def calculate_angle(p1, p2, p3):
    # 计算p1p2p3三个点之间的夹角（以p2为顶点）
    vector1 = (p1[0] - p2[0], p1[1] - p2[1])
    vector2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    angle = math.degrees(math.atan2(cross_product, dot_product)) #method?
    angle = (angle + 360) % 360  # 转换为0到360度的角度
    return angle


# random path passing the known well position
def generate_random_path(points, extension_length):

    # 确定起点延长的方向
    start_direction = (random.uniform(0, 300), random.uniform(0, 300))
    start_direction = (start_direction[0] / math.sqrt(start_direction[0] ** 2 + start_direction[1] ** 2),
                       start_direction[1] / math.sqrt(start_direction[0] ** 2 + start_direction[1] ** 2))


    for i in range(100000):
        path = []
        visited_wells = set()
        random.shuffle(points)
        # 确定起点延长的位置 vector computation
        start_point = ((points[0][0] + start_direction[0] * extension_length),
                       (points[0][1] + start_direction[1] * extension_length),
                       '')
        path.append(start_point)

        for next_point in points:

            # 检查路径是否经过相同的一口井
            if next_point[2] in visited_wells:
                continue

            # 计算相邻路径之间的夹角
            if len(path) >= 2:
                prev_point = path[-1]
                angle = calculate_angle(path[-2], prev_point, next_point)

                # 检查夹角是否满足要求
                if angle <= 80 or angle >= 280:
                    continue

            # 添加下一个点到路径中
            path.append(next_point)
            visited_wells.add(next_point[2])

            # 检查路径是否足够长
            if len(visited_wells) >= 3:
                break

        if len(visited_wells) >= 3:
            break

        print(i)

    # 添加延长部分
    if len(path) >= 2:
        direction = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        direction = (direction[0] / math.sqrt(direction[0] ** 2 + direction[1] ** 2),
                     direction[1] / math.sqrt(direction[0] ** 2 + direction[1] ** 2))

        extension_point = (
            path[-1][0] + direction[0] * extension_length,
            path[-1][1] + direction[1] * extension_length,
            ''
        )
        path.append(extension_point)

    return path


# interpolate the path to extract integer points
def remove_duplicate_points(points):
    seen = []  # 辅助列表用于记录已经出现过的点
    result = []  # 结果列表，用于存储去重后的点

    for point in points:
        if point not in seen:
            result.append(point)
            seen.append(point)

    return result


from scipy.interpolate import CubicSpline
def interpolate_path_points1(path):
    # 创建路径的坐标数组
    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])

    # 计算参数化的样条插值
    t = np.arange(len(path))
    cs = CubicSpline(t, np.column_stack((x, y)), bc_type='natural')

    # 对插值曲线进行采样
    t_new = np.linspace(0, len(path) - 1, 10 * (len(path) - 1))  # 增加采样点数量以获得更连续的整数点
    interpolated_points = [(int(cs_x), int(cs_y)) for cs_x, cs_y in zip(cs(t_new)[:, 0], cs(t_new)[:, 1])]

    #  # 去除重复的点
    interpolated_points = remove_duplicate_points(interpolated_points)

    return interpolated_points

def interpolate_path_points(path):
    interpolated_points = []

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        # 计算路径段的长度
        segment_length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        # 计算路径段的方向向量
        segment_direction = ((p2[0] - p1[0]) / segment_length, (p2[1] - p1[1]) / segment_length)

        # 计算路径段上的整数点
        num_points = int(segment_length) + 1
        for j in range(num_points):
            x = round(p1[0] + j * segment_direction[0])
            y = round(p1[1] + j * segment_direction[1])
            interpolated_points.append((x, y))

    # # 移除重复的点
    interpolated_points = remove_duplicate_points(interpolated_points)

    return interpolated_points


# generate random well-points
def generate_well_points(num_well, x_range, y_range):
    points = []
    for _ in range(num_well):
        x = random.randint(*x_range)
        y = random.randint(*y_range)
        points.append((x, y))
    return points

def add_labels(lst):
    labels = [chr(ord('A') + i) for i in range(len(lst))]  # 生成字母标签列表
    return [(item[0], item[1], label) for item, label in zip(lst, labels)]  # 将字母标签与列表项进行组合


def find_points_indices(points, target_points):
    indices = []

    for point in target_points:
        if point in points:
            index = points.index(point)
            indices.append(index)

    return indices