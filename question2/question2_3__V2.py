import numpy as np
import math
from question2.my_util import read_population_economic_from_excel, read_distance_from_excel
from question2.road import Road

"""
问题2子问题2的第二种做法
"""


def compute_joint_gdp(a, b):
    return math.sqrt(a * b)


def compute_joint_population(a, b):
    """
    给出光纤连接的两个区域的人口数，两个区域的联合
    :param a:
    :param b:
    :return:
    """
    return math.sqrt(a * b)


# def compute_value(rd):
#     """
#     计算一条路的价值
#     :param rd: Road对象
#     :return:
#     """
#     return rd.weight * rd.capacity * rd.population


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
    value_list = np.array([a_road.value for a_road in road_list])
    return road_list[value_list.argmax(axis=0)]


def get_road_from_list(x, y, road_list):
    if x < y:
        a, b = x, y
    else:
        a, b = y, x
    for one_road in road_list:
        if one_road.start == a and one_road.end == b:
            return one_road
    return None


def convert_distance_to_level(length):
    if length > 3000:
        return 0
    elif 3000 >= length > 1200:
        return 1
    elif 1200 >= length > 600:
        return 2
    else:
        return 3


if __name__ == '__main__':
    num_links = 33
    triple_points_road_list = []  # 记录三个节点的路径所经过的每一个节点
    distance_matrix, city_list = read_distance_from_excel()
    print(city_list)
    city_population_list, city_gdp_aver_list = read_population_economic_from_excel()
    adj_matrix = [[0 for _ in range(12)] for _ in range(12)]  # 存储整个图的邻接矩阵
    all_road_info_list = []  # 保存所有路的信息的列表
    cur_road_info_list = []  # 保存当前已有的路的信息的列表
    part_a = []  # 已经联通的一部分
    part_b = [i for i in range(12)]  # 尚未联通的部分
    for i in range(12):
        for j in range(i, 12):
            if i == j:
                continue
            # 初始化所有可能的Road，权重设为1，表示不考虑经济状况
            weight_gdp = compute_joint_gdp(city_gdp_aver_list[i], city_gdp_aver_list[j])
            road = Road(start=i, end=j, weight=1,
                        capacity=get_capacity(distance_matrix[i][j]),
                        population=compute_joint_population(city_population_list[i],
                                                            city_population_list[j]),
                        value=weight_gdp * get_capacity(
                            distance_matrix[i][j]) * compute_joint_population(
                            city_population_list[i], city_population_list[j]))
            all_road_info_list.append(road)
    print("所有可能的路径:")
    for item in sorted(all_road_info_list, key=lambda asd: asd.value, reverse=True):
        print(city_list[item.start], city_list[item.end], item.value)
    # print(len(all_road_info_list))
    # for item in all_road_info_list:
    #     print(item)
    for _ in range(11):
        if len(part_a) == 0:
            max_value_road = get_max_value_road(all_road_info_list)
            # print(city_list[max_value_road.start], city_list[max_value_road.end], compute_value(max_value_road))
            part_a.append(max_value_road.start)
            part_b.remove(max_value_road.start)
            part_a.append(max_value_road.end)
            part_b.remove(max_value_road.end)
            adj_matrix[max_value_road.start][max_value_road.end] = 1
            adj_matrix[max_value_road.end][max_value_road.start] = 1
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
            adj_matrix[max_value_road.start][max_value_road.end] = 1
            adj_matrix[max_value_road.end][max_value_road.start] = 1
            cur_road_info_list.append(max_value_road)
            all_road_info_list.remove(max_value_road)
    print("为了满足连通图而架设的", len(cur_road_info_list), "条光纤如下所示：")
    for item in cur_road_info_list:
        print(city_list[item.start], city_list[item.end], item.value)
    total_value = sum([item.value for item in cur_road_info_list])
    print(len(cur_road_info_list), "条光纤的总价值之和为", total_value, '\n')

    # print("邻接矩阵为:")
    # print(np.array(adj_matrix))
    # for item in all_road_info_list:
    #     print(item)
    while len(cur_road_info_list) < num_links:
        print('当前连接个数为:', len(cur_road_info_list))
        # 获得当前价值最高的直接连接的一条路
        max_value_dir_road = get_max_value_road(all_road_info_list)
        # 计算当前价值最高的一条直接连接的路的价值
        max_dir_link_value = max_value_dir_road.value
        max_add_middle_value = max_dir_link_value  # 添加一个中间节点所能带来的最大价值增量
        road_added_for_middle_point = None
        shared_capacity_road = None
        cur_abc_for_middle_road = None
        for tmp_a in range(0, 12):
            for tmp_b in range(tmp_a + 1, 12):
                # 判断是否加入了一条中转节点的路
                if_add_one_link = False  # 如果一次循环中没有加入任何的中转节点连接，则加入一条直接连接的边
                # print('查找', city_list[tmp_a], city_list[tmp_b], '是否有可能通过中继节点连接')
                # 遍历12个城市，尝试查询中间节点
                for middle in range(12):
                    # 如果中间节点与这两个城市任意一个重复，换下一个中间节点
                    if middle == tmp_a or middle == tmp_b:
                        continue
                    # print('计算', city_list[middle], '能否作为中间节点')
                    # 中间节点和a没有直接连接，和b有直接连接
                    if adj_matrix[tmp_a][middle] != 1 and adj_matrix[tmp_b][middle] == 1:
                        # 如果ab的人口数乘积大于a,middle人口数乘积，才加中继节点
                        # 可以给ab分配的最大容量为a, middle 和 b, middle之间的最小容量
                        max_capacity = min(get_capacity(distance_matrix[tmp_a][middle]),
                                           get_capacity(distance_matrix[tmp_b][middle]))
                        value_added = max_capacity * compute_joint_gdp(city_gdp_aver_list[tmp_a], city_gdp_aver_list[tmp_b]) * \
                                      compute_joint_population(city_population_list[tmp_a], city_population_list[tmp_b]) - \
                                      max_capacity * compute_joint_gdp(city_gdp_aver_list[tmp_b], city_gdp_aver_list[middle]) * \
                                      compute_joint_population(city_population_list[tmp_b], city_population_list[middle])
                        # print('max_dir_link_value', max_dir_link_value)
                        # print('value_added', value_added)
                        if value_added > max_add_middle_value:
                            max_add_middle_value = value_added
                            # 将a和middle之间的通路设为当前可能添加的路径
                            road_added_for_middle_point = get_road_from_list(tmp_a, middle, all_road_info_list)
                            shared_capacity_road = get_road_from_list(middle, tmp_b, cur_road_info_list)
                            cur_abc_for_middle_road = (tmp_a, middle, tmp_b)
                    # 中间节点和a有直接连接，和b没有直接连接
                    if adj_matrix[tmp_a][middle] == 1 and adj_matrix[tmp_b][middle] != 1:
                        # 可以给ab分配的最大容量为a, middle 和 b, middle之间的最小容量
                        max_capacity = min(get_capacity(distance_matrix[tmp_a][middle]),
                                           get_capacity(distance_matrix[tmp_b][middle]))
                        value_added = max_capacity * compute_joint_gdp(city_gdp_aver_list[tmp_a], city_gdp_aver_list[tmp_b]) * \
                                      compute_joint_population(city_population_list[tmp_a], city_population_list[tmp_b]) - \
                                      max_capacity * compute_joint_gdp(city_gdp_aver_list[tmp_a], city_gdp_aver_list[middle]) * \
                                      compute_joint_population(city_population_list[tmp_a], city_population_list[middle])
                        # print('max_dir_link_value', max_dir_link_value)
                        # print('value_added', value_added)
                        if value_added > max_add_middle_value:
                            max_add_middle_value = value_added
                            # 将b和middle之间的通路设为当前可能添加的路径
                            road_added_for_middle_point = get_road_from_list(middle, tmp_b, all_road_info_list)
                            shared_capacity_road = get_road_from_list(tmp_a, middle, cur_road_info_list)
                            cur_abc_for_middle_road = (tmp_a, middle, tmp_b)
        if road_added_for_middle_point is None:
            # 如果没有添加任何的中继节点，选一条尚未添加的，价值最大的边作为直接连接加入网络
            total_value += max_dir_link_value
            print("无合适的中间节点，加入(", city_list[max_value_dir_road.start], '-',
                  city_list[max_value_dir_road.end], ')这条连接\n')
            # 在邻接矩阵中表明这两个点已经连接
            adj_matrix[max_value_dir_road.start][max_value_dir_road.end] = 1
            adj_matrix[max_value_dir_road.end][max_value_dir_road.start] = 1
            # 在当前连接的列表中加入这一条路
            cur_road_info_list.append(max_value_dir_road)
            # 在所有路的列表中删除这一条路
            all_road_info_list.remove(max_value_dir_road)
        else:
            total_value += max_add_middle_value
            print('直接连接的路径最大价值为', max_dir_link_value)
            print('通过中转点连接的路径最大增加价值为', max_add_middle_value)
            cur_road_info_list.append(road_added_for_middle_point)
            all_road_info_list.remove(road_added_for_middle_point)
            shared_capacity_road.value += max_add_middle_value
            print("加入",
                  city_list[road_added_for_middle_point.start],
                  '-',
                  city_list[road_added_for_middle_point.end], "构成具有中间节点的路径")
            print("这一条中间路径为:(",
                  city_list[cur_abc_for_middle_road[0]], '-',
                  city_list[cur_abc_for_middle_road[1]], '-',
                  city_list[cur_abc_for_middle_road[2]], '-',
                  ")\n")

    print('\n目前地图中一共有', len(cur_road_info_list), '条边\n')
    print('总价值为', total_value)
