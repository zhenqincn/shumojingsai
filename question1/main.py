import random
import numpy as np
import math
from scipy.interpolate import spline
import matplotlib.pyplot as plt


def nearest(point, targets):
    distance_list = np.zeros(len(targets), dtype=np.float32)
    for i in range(len(targets)):
        distance_list[i] = math.sqrt(math.pow((point[0] - targets[i][0]), 2) + math.pow((point[1] - targets[i][1]), 2))
    return np.argmin(distance_list, axis=0)


def compute_wrong_bit_num(predicted_index_list, right_index, wrong_weight_list):
    """
    计算错误的bit个数
    :param predicted_index_list: 预测出的点所在的下标
    :param right_index: 正确的点所在的下标
    :param wrong_weight_list: 错误的点的权重列表
    :return:
    """
    wrong_bit_num = 0
    for index in predicted_index_list:
        if index != right_index:
            wrong_bit_num += wrong_weight_list[index]
    return wrong_bit_num


def get_std(snr, sk):
    ps = np.sum([math.pow(tmp, 2) for tmp in sk]) / len(sk)   # 信号平均功率
    std = ps / snr
    return std


def get_std_db(snr, sk):
    ps = np.sum([math.pow(tmp, 2) for tmp in sk]) / len(sk)  # 信号平均功率
    index = math.log10(ps) - snr / 10
    std = math.pow(10, index)
    return std


def get_random_points(base_point, std, num):
    """
    生成一系列随机点
    :param base_point: 基础点（随机点到基础点的距离符合正态分布）
    :param std: 正态分布的标准差
    :param num: 生成点的数目
    :return:
    """
    random_distance_list = np.random.normal(loc=0.0, scale=std, size=num)
    random_points_list = np.array([base_point for _ in range(num)])
    for i in range(num):
        theta = random.randint(0, 360)
        random_points_list[i][0] += random_distance_list[i] * math.sin(theta)
        random_points_list[i][1] += random_distance_list[i] * math.cos(theta)
    return random_points_list


def get_assigned_points_index_list(random_points_list, standard_points):
    """
    计算每个点被分配到的标准点的index
    :param random_points_list:
    :param standard_points:
    :return:
    """
    pred_list = []
    for point in random_points_list:
        pred_list.append(nearest(point, standard_points))
    return pred_list


def show_points(points_list):
    x = []
    y = []
    for point in points_list:
        x.append(point[0])
        y.append(point[1])
    plt.scatter(x, y)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.show()


def bit_err_ratio(snr, sk, num_bit_per_point, standard_points, wrong_weight):
    """

    :param snr: 信噪比
    :param sk:
    :param num_bit_per_point: 每一个点携带的信息的位数
    :param standard_points: 标准点列表
    :param wrong_weight: 每个点错误的权重
    :return:
    """
    std = get_std_db(snr=snr, sk=sk)  # 通过信噪比和sk来计算标准差
    num_points = 100000   # 生成随机点的个数
    random_points_list = get_random_points([1.0, 1.0], std, num_points)
    show_points(random_points_list)
    pred_list_index = get_assigned_points_index_list(random_points_list, standard_points)
    wrong_bit_num = compute_wrong_bit_num(pred_list_index, 0, wrong_weight)
    return wrong_bit_num / (num_bit_per_point * num_points)


if __name__ == '__main__':
    x = []
    y = []
    for i in range(2, 11):
        snr = math.log10(i)
        ber = bit_err_ratio(snr=snr,
                            sk=[1.0, 1.0, 1.0, 1.0],
                            num_bit_per_point=2,
                            standard_points=[[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
                            wrong_weight=[0, 1, 2, 1]
                            )
        print("信噪比:", snr, "误码率:", ber)
        x.append(snr)
        y.append(ber)
    # x = np.array(x)
    # y = np.array(y)
    # xnew = np.linspace(x.min(), x.max(), 1000)  # 300 represents number of points to make between T.min and T.max
    #
    # power_smooth = spline(x, y, xnew)
    #
    # plt.plot(xnew, power_smooth)
    # plt.show()
