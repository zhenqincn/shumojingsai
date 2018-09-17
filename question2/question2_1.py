import numpy as np
import math
from question2.my_util import read_population_economic_from_excel, read_distance_from_excel
from question2.road import Road


def compute_population(a, b):
    """
    给出光纤连接的两个区域的人口数，计算光纤的人口比重
    :param a:
    :param b:
    :return:
    """
    return math.sqrt(a * b)


def compute_value(road):
    """
    计算一条路的价值
    :param road:
    :return:
    """
    return road.weight * road.capacity * road.population


def get_capacity(length):
    if length > 3000:
        return 0
    elif 3000 >= length > 1200:
        return 8
    elif 1200 >= length > 600:
        return 16
    else:
        return 32


def get_max_value_road(road_list):
    value_list = np.array([compute_value(road) for road in road_list])
    return road_list[value_list.argmax(axis=0)]


def get_road_from_list(x, y, road_list):
    for one_road in road_list:
        if one_road.start == x and one_road.end == y:
            return one_road


if __name__ == '__main__':
    num_links = 33

    distance_matrix, city_list = read_distance_from_excel()
    # print(city_list)
    city_population_list, city_gdp_aver_list = read_population_economic_from_excel()
    adj_matrix = [[0 for _ in range(12)] for _ in range(12)]  # 存储整个图的邻接矩阵

    all_road_info_list = []             # 保存所有路的信息的列表
    cur_road_info_list = []             # 保存当前已有的路的信息的列表
    part_a = []                         # 已经联通的一部分
    part_b = [i for i in range(12)]     # 尚未联通的部分
    for i in range(12):
        for j in range(i, 12):
            if i == j:
                continue
            road = Road(start=i, end=j, weight=1, capacity=get_capacity(distance_matrix[i][j]),
                        population=compute_population(city_population_list[i], city_population_list[j]))
            all_road_info_list.append(road)
    # print("所有可能的路径:")
    # for item in sorted(all_road_info_list, key=lambda asd:compute_value(asd), reverse=True):
    #     print(city_list[item.start], city_list[item.end], compute_value(item))
    # print(len(all_road_info_list))
    for _ in range(num_links):
        if len(part_a) < 12:
            if len(part_a) == 0:
                max_value_road = get_max_value_road(all_road_info_list)
                # print(city_list[max_value_road.start], city_list[max_value_road.end], compute_value(max_value_road))
                part_a.append(max_value_road.start)
                part_b.remove(max_value_road.start)
                part_a.append(max_value_road.end)
                part_b.remove(max_value_road.end)
                cur_road_info_list.append(max_value_road)
                all_road_info_list.remove(max_value_road)
                # print(len(all_road_info_list))
            else:
                possible_road_x_y = []
                for item in all_road_info_list:
                    if (item.start in part_a and item.end in part_b) or (item.start in part_b and item.end in part_a):
                        possible_road_x_y.append(item)
                max_value_road = get_max_value_road(possible_road_x_y)
                # print(city_list[max_value_road.start], city_list[max_value_road.end], compute_value(max_value_road))
                # print('part_a', part_a)
                # print('part_b', part_b)
                # print(max_value_road.start, max_value_road.end)
                if max_value_road.start in part_a:
                    part_a.append(max_value_road.end)
                    part_b.remove(max_value_road.end)
                elif max_value_road.start in part_b:
                    part_a.append(max_value_road.start)

                    part_b.remove(max_value_road.start)
                cur_road_info_list.append(max_value_road)
                all_road_info_list.remove(max_value_road)
        else:
            max_value_road = get_max_value_road(all_road_info_list)
            cur_road_info_list.append(max_value_road)
            all_road_info_list.remove(max_value_road)
    print("架设的", len(cur_road_info_list), "条光纤如下所示：")
    for item in cur_road_info_list:
        print(city_list[item.start], city_list[item.end], compute_value(item))
    print(len(cur_road_info_list), "条光纤的总价值之和为", sum([compute_value(item) for item in cur_road_info_list]))
