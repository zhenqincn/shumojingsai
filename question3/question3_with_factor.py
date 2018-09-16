from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
import numpy as np
import math
from scipy.interpolate import spline
import matplotlib.pyplot as plt


def get_source_data(num_standard_points):
    if num_standard_points == 4:
        print("\n\n方框的个数为4")
        base_points = [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]
        base_points_info = [[0, 1], [1, 1], [1, 0], [0, 0]]
        sk2 = [2.0, 2.0, 2.0, 2.0]
    elif num_standard_points == 8:
        print("\n\n方框的个数为8")
        base_points = [
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            [0, (math.sqrt(3) + 1.0) / 2.0],
            [(math.sqrt(3) + 1.0) / 2.0, 0],
            [0, - (math.sqrt(3) + 1.0) / 2.0],
            [- (math.sqrt(3) + 1.0) / 2.0, 0],
        ]
        base_points_info = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0]
        ]
        tmp = math.pow(((math.sqrt(3) + 1) / 2), 2)
        sk2 = [0.5, 0.5, 0.5, 0.5, tmp, tmp, tmp, tmp]
    elif num_standard_points == 16:
        gen2 = math.sqrt(2)
        external_r = 4.15
        pi = math.pi
        base_points = [[gen2, gen2],
                       [gen2, -gen2],
                       [-gen2, -gen2],
                       [-gen2, gen2],
                       [external_r * math.cos(pi * 5 / 12), external_r * math.sin(pi * 5 / 12)],
                       [external_r * math.cos(pi / 4), external_r * math.sin(pi / 4)],
                       [external_r * math.cos(pi / 12), external_r * math.sin(pi / 12)],
                       [external_r * math.cos(-pi / 12), external_r * math.sin(-pi / 12)],
                       [external_r * math.cos(-pi / 4), external_r * math.sin(-pi / 4)],
                       [external_r * math.cos(-pi * 5 / 12), external_r * math.sin(-pi * 5 / 12)],
                       [external_r * math.cos(-pi * 7 / 12), external_r * math.sin(-pi * 7 / 12)],
                       [external_r * math.cos(-pi * 9 / 12), external_r * math.sin(-pi * 9 / 12)],
                       [external_r * math.cos(-pi * 11 / 12), external_r * math.sin(-pi * 11 / 12)],
                       [external_r * math.cos(pi * 11 / 12), external_r * math.sin(pi * 11 / 12)],
                       [external_r * math.cos(pi * 9 / 12), external_r * math.sin(pi * 9 / 12)],
                       [external_r * math.cos(pi * 7 / 12), external_r * math.sin(pi * 7 / 12)]
                       ]
        print(base_points)
        base_points_info = [[1, 1, 0, 0],
                            [1, 1, 0, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 0, 1],
                            [0, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 0],
                            [0, 0, 1, 0],
                            [1, 0, 1, 0]
                            ]
        sk2 = [1.0,
               1.0,
               1.0,
               1.0,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r,
               external_r * external_r]
    else:
        base_points = None
        base_points_info = None
        sk2 = None
    return base_points, base_points_info, sk2


def nearest(point, targets, factor):
    """
    计算距离最近的点
    :param point: (x, y)
    :param targets: (x, y)
    :return: 距离最近的点对应标准点的index
    """
    distance_list = np.zeros(len(targets), dtype=np.float32)
    for i in range(len(targets)):
        distance_list[i] = math.sqrt(math.pow((point[0] - targets[i][0]), 2) + math.pow((point[1] - targets[i][1]), 2))
        if 0 <= i < 4:
            distance_list[i] /= factor
    return np.argmin(distance_list, axis=0)


def compute_wrong_bit_num(origin_index_list, noised_index_list, standard_points_info_list):
    """
    计算错误的bit个数
    :param origin_index_list 没加噪声的点的index
    :param noised_index_list 加了噪声的点的index
    :param standard_points_info_list: 标准点的信息的列表
    :return:
    """
    wrong_bit_num = 0
    for i in range(len(origin_index_list)):
        for j in range(len(standard_points_info_list[0])):
            if standard_points_info_list[origin_index_list[i]][j] != standard_points_info_list[noised_index_list[i]][j]:
                wrong_bit_num += 1
    return wrong_bit_num


def get_std(snr, sk):
    ps = np.sum([math.pow(tmp, 2) for tmp in sk]) / len(sk)  # 信号平均功率
    std = ps / snr
    return std


def get_pn(snr, sk2, probablity):
    ps = np.sum(sk2 * np.array(probablity))
    pn = ps * math.pow(10, (- snr / 10))
    return pn


def get_noised_points(index_list, base_points, pn):
    """
    给一个随机的列表（base_point的index），生成加入噪声的点[(x, y)], 噪声向量的模服从正态分布
    :param index_list:
    :param base_points: 基础点的列表
    :param std:
    :return:
    """
    std = math.sqrt(pn)
    my_points = []
    for index in index_list:
        my_points.append(np.array(base_points[index]))
    random_distance_list = np.random.normal(loc=0.0, scale=std, size=len(my_points))
    for i in range(len(my_points)):
        theta = random.randint(0, 360)
        my_points[i][0] += random_distance_list[i] * math.sin(theta)
        my_points[i][1] += random_distance_list[i] * math.cos(theta)
    return my_points


def get_noised_points_x_y_independent(index_list, base_points, pn):
    """
    在x, y 分别为独立的高斯分布的情况下，获得被噪声污染后的点
    :param index_list:
    :param base_points:
    :param std:
    :return:
    """
    std = math.sqrt(pn / 2)
    my_points = []
    for index in index_list:
        my_points.append(np.array(base_points[index]))
    random_distance_list_x = np.random.normal(loc=0.0, scale=std, size=len(my_points))
    random_distance_list_y = np.random.normal(loc=0.0, scale=std, size=len(my_points))
    for i in range(len(my_points)):
        my_points[i][0] += random_distance_list_x[i]
        my_points[i][1] += random_distance_list_y[i]
    return my_points


def get_assigned_points_index_list(random_points_list, standard_points, factor):
    """
    计算每个点被分配到的标准点的index
    :param random_points_list:
    :param standard_points:
    :return:
    """
    pred_list = []
    for point in random_points_list:
        pred_list.append(nearest(point, standard_points, factor))
    return pred_list


def show_points(point_list, origin_index_list, noised_index_list, title=None, info=None):
    x = []
    y = []
    color = []
    for i in range(len(point_list)):
        x.append(point_list[i][0])
        y.append(point_list[i][1])
        if origin_index_list[i] != noised_index_list[i]:
            color.append('red')
        else:
            color.append('blue')
    plt.scatter(x, y, c=color, s=1.5)
    if max(origin_index_list) > 8:
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        if info is not None:
            plt.annotate(info, xy=(-4.5, -4.5))
    else:
        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        if info is not None:
            plt.annotate(info, xy=(-2.5, -2.5))
    if title is not None:
        plt.savefig(title + ".png", dpi=600)
    plt.clf()


def bit_err_ratio(snr, sk2, num_bit_per_point, standard_points, standard_points_info, show=False, factor=1):
    """
    :param snr: 信噪比
    :param sk2:
    :param num_bit_per_point: 每一个点携带的信息的位数
    :param standard_points: 标准点列表
    :param standard_points_info: 标准点携带的信息的列表
    :param show: 是否绘制散点图
    :return:
    """
    num_points = 16384 * 4 # 生成随机点的个数
    # 生成num个随机点（以index映射到standard points）
    probablity = [0.20267, 0.20267, 0.20267, 0.20267,
                  0.01577, 0.01577, 0.01577, 0.01577,
                  0.01577, 0.01577, 0.01577, 0.01577,
                  0.01577, 0.01577, 0.01577, 0.01577]
    pn = get_pn(snr=snr, sk2=sk2, probablity=probablity)  # 通过信噪比和sk来计算标准差
    seed = []
    for i in range(len(probablity)):
        for _ in range(int(probablity[i] * 10000)):
            seed.append(i)
    rand_list = [seed[index] for index in np.random.randint(low=0, high=len(seed), size=num_points)]
    noised_points = get_noised_points_x_y_independent(index_list=rand_list, base_points=standard_points, pn=pn)
    pred_list_index = get_assigned_points_index_list(noised_points, standard_points, factor)
    wrong_bit_num = compute_wrong_bit_num(origin_index_list=rand_list,
                                          noised_index_list=pred_list_index,
                                          standard_points_info_list=standard_points_info)
    ber = wrong_bit_num / (num_bit_per_point * num_points)
    if show:
        show_points(point_list=noised_points, origin_index_list=rand_list,
                    noised_index_list=pred_list_index, title=str(len(standard_points_info[0])),
                    info='SNR(dB): ' + str(snr))
    return ber


if __name__ == '__main__':
    x_list = []
    y_list = []
    num_box_list = [16, 16]
    snr_range = [[2, 18], [2, 18]]
    num_bit_list = [4, 4]
    plot_list = []
    factor_list = [1.1, 1]
    for i in range(len(num_box_list)):
        base_points, base_points_info, sk2 = get_source_data(num_box_list[i])
        print(sk2)
        x = []
        y = []
        for snr in range(snr_range[i][0], snr_range[i][1]):
            flag = False
            if snr == snr_range[i][1] - 1:
                flag = True
            ber = bit_err_ratio(snr=snr,
                                sk2=sk2,
                                num_bit_per_point=num_bit_list[i],
                                standard_points=base_points,
                                standard_points_info=base_points_info,
                                show=flag,
                                factor=factor_list[i]
                                )
            print("信噪比:", snr, "误码率:", ber)
            x.append(snr)
            y.append(ber)
        x_list.append(np.array(x))
        y_list.append(np.array(y))
        # xnew = np.linspace(x.min(), x.max(), 1)  # 300 represents number of points to make between T.min and T.max
        #
        # power_smooth = spline(x, y, xnew)
        #
        # plt.plot(xnew, power_smooth)
    for i in range(len(num_box_list)):
        plt.yscale('log')
        plot, = plt.plot(x_list[i], y_list[i])
        plot_list.append(plot)
        plt.plot(x_list[i], y_list[i], '*')
    for i in range(len(x_list)):
        print('\n' + str(num_box_list[i]) + '格:')
        for j in range(len(x_list[i])):
            print(x_list[i][j], y_list[i][j])
    plt.plot([2, 17], [0.02, 0.02], c='red')
    plt.annotate("BER threshold = 0.02", xy=(2, 0.015))
    plt.legend(plot_list, ['QPSK', '8QAM', '16QAM'], loc='lower left')
    plt.ylabel("Bit Error Rate")
    plt.xlabel("Signal-to-Noise Ratio(dB)")
    plt.savefig('line.png', dpi=600)

