import numpy as np
import math
from question2.my_util import read_population_economic_from_excel, read_distance_from_excel
from question2.road import Road


def compute_joint_population(a, b):
    """
    给出光纤连接的两个区域的人口数，两个区域的联合
    :param a:
    :param b:
    :return:
    """
    return math.sqrt(a * b)


def compute_value(rd):
    """
    计算一条路的价值
    :param rd: Road对象
    :return:
    """
    return rd.weight * rd.capacity * rd.population


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
    value_list = np.array([compute_value(a_road) for a_road in road_list])
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
    city_pair_population_tuple_list = []
    for i in range(len(city_population_list)):
        for j in range(i + 1, len(city_population_list)):
            city_pair_population_tuple_list.append((i, j, math.sqrt(city_population_list[i] * city_population_list[j])))
    city_pair_population_tuple_list = sorted(city_pair_population_tuple_list, key=lambda pair: pair[2], reverse=True)
    print(city_pair_population_tuple_list)
    adj_matrix = [[0 for _ in range(12)] for _ in range(12)]  # 存储整个图的邻接矩阵
    all_road_info_list = []  # 保存所有路的信息的列表
    cur_road_info_list = []  # 保存当前已有的路的信息的列表
    part_a = []  # 已经联通的一部分
    part_b = [i for i in range(12)]  # 尚未联通的部分
    for i in range(12):
        for j in range(i, 12):
            if i == j:
                continue
            road = Road(start=i, end=j, weight=1, capacity=get_capacity(distance_matrix[i][j]),
                        population=compute_joint_population(city_population_list[i], city_population_list[j]))
            all_road_info_list.append(road)
    # print("所有可能的路径:")
    # for item in sorted(all_road_info_list, key=lambda asd:compute_value(asd), reverse=True):
    #     print(city_list[item.start], city_list[item.end], compute_value(item))
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
        print(city_list[item.start], city_list[item.end], compute_value(item))
    total_value = sum([compute_value(item) for item in cur_road_info_list])
    print(len(cur_road_info_list), "条光纤的总价值之和为", total_value)

    # print("邻接矩阵为:")
    # print(np.array(adj_matrix))
    # for item in all_road_info_list:
    #     print(item)
    while len(cur_road_info_list) < num_links:
        # 获得通过直接连接得到的当前价值最高的一条路
        max_value_road = get_max_value_road(all_road_info_list)
        # 计算得到通过直接连接得到的当前价值最高的一条路的价值
        max_dir_link_value = compute_value(max_value_road)

        print('当前连接个数为:', len(cur_road_info_list))
        if_add_one_link = False  # 如果一次循环中没有加入任何的中转节点连接，则加入一条直接连接的边
        city_tuple_list = city_pair_population_tuple_list[0]
        # 获得当前城市对的两个城市编号，以及两个城市的联合人口权重
        tmp_a, tmp_b, tmp_population = city_tuple_list[0], city_tuple_list[1], city_tuple_list[2]
        # 如果当前城市对中两个城市已经直接连接，那么不需要再考虑中继节点,换下一组节点
        while adj_matrix[tmp_a][tmp_b] == 1:
            city_pair_population_tuple_list.remove(city_tuple_list)
            city_tuple_list = city_pair_population_tuple_list[0]
            # 获得当前城市对的两个城市编号，以及两个城市的联合人口权重
            tmp_a, tmp_b, tmp_population = city_tuple_list[0], city_tuple_list[1], city_tuple_list[2]
        print('查找', city_list[tmp_a], city_list[tmp_b], '是否有可能通过中继节点连接')
        # 将两个城市间的直线距离映射到0,1,2,3
        cur_distance_level = convert_distance_to_level(distance_matrix[tmp_a][tmp_b])
        # 遍历12个城市，尝试查询中间节点
        for middle in range(12):
            # 如果中间节点与这两个城市任意一个重复，换下一个中间节点
            if middle == tmp_a or middle == tmp_b:
                continue
            # 判断这两个点是否和所有的中转路径中的任意两个点重复
            if_triple_roads_occupy = False
            for triple_road in triple_points_road_list:
                # 如果这两个点和所有的中转路径中的任意两个点重复，不加这条路径
                if (tmp_a in triple_road and middle in triple_road) or \
                        (tmp_b in triple_road and middle in triple_road):
                    # 如果重复，放弃对当前城市对的遍历
                    if_triple_roads_occupy = True
                    break
            if if_triple_roads_occupy:
                break
            # 中间节点和a没有直接连接，和b有直接连接
            if adj_matrix[tmp_a][middle] != 1 and adj_matrix[tmp_b][middle] == 1:
                # a和middle、b和middle的直线距离级别必须都高于a和b之间的距离级别，才有加边的必要
                if convert_distance_to_level(distance_matrix[tmp_a][middle]) > \
                        cur_distance_level and convert_distance_to_level(distance_matrix[tmp_b][middle]) > \
                        cur_distance_level:
                    # 如果ab的人口数乘积大于a,middle人口数乘积，才加中继节点
                    if compute_joint_population(city_population_list[tmp_a],
                                                city_population_list[tmp_b]) > \
                            compute_joint_population(city_population_list[tmp_a], city_population_list[middle]):
                        ''''''
                        # 从所有的路的列表中获得a和middle之间的路的信息
                        road_a_middle = get_road_from_list(tmp_a, middle, all_road_info_list)
                        # 可以给ab分配的最大容量为a, middle 和 b, middle之间的最小容量
                        max_capacity = min(get_capacity(distance_matrix[tmp_a][middle]),
                                           get_capacity(distance_matrix[tmp_b][middle]))
                        value_added = max_capacity * 1 * math.sqrt(
                            city_population_list[tmp_a] * city_population_list[tmp_b]) - compute_value(road_a_middle)
                        ''''''
                        if value_added > max_dir_link_value:
                            # 当前路列表加入一条
                            total_value += value_added
                            cur_road_info_list.append(road_a_middle)
                            # 全部路列表中减去一条
                            all_road_info_list.remove(road_a_middle)
                            print("加入", city_list[road_a_middle.start], '-', city_list[road_a_middle.end], '这条连接(中继)\n')
                            adj_matrix[tmp_a][middle] = 1
                            adj_matrix[middle][tmp_a] = 1
                            # 加入了一条边，删除这条边两端的城市组成的城市对
                            city_pair_population_tuple_list.remove(city_tuple_list)
                            triple_points_road_list.append([tmp_a, middle, tmp_b])
                            # print('triple', triple_points_road_list)
                            # 是否添加了一条边的标志位置为True
                            if_add_one_link = True
                            # 停止for middle in range(12):
                            break
            # 中间节点和a有直接连接，和b没有直接连接
            if adj_matrix[tmp_a][middle] == 1 and adj_matrix[tmp_b][middle] != 1:
                # a和middle、b和middle的直线距离级别必须都高于a和b之间的距离级别，才有加边的必要
                if convert_distance_to_level(distance_matrix[tmp_a][middle]) > cur_distance_level and \
                        convert_distance_to_level(distance_matrix[tmp_b][middle]) > cur_distance_level:
                    if compute_joint_population(city_population_list[tmp_a],
                                                city_population_list[tmp_b]) > \
                            compute_joint_population(city_population_list[tmp_b], city_population_list[middle]):
                        # 从所有的路的节点中获取middle和b之间的路的权重
                        road_b_middle = get_road_from_list(tmp_b, middle, all_road_info_list)
                        # 可以给ab分配的最大容量为a, middle 和 b, middle之间的最小容量
                        max_capacity = min(get_capacity(distance_matrix[tmp_a][middle]),
                                           get_capacity(distance_matrix[tmp_b][middle]))
                        value_added = max_capacity * 1 * math.sqrt(
                            city_population_list[tmp_a] * city_population_list[tmp_b]) - compute_value(road_b_middle)
                        if value_added > max_dir_link_value:
                            total_value += value_added
                            # 当前路列表加入一条
                            cur_road_info_list.append(road_b_middle)
                            # 全部路的列表删除这一条
                            all_road_info_list.remove(road_b_middle)
                            print("加入", city_list[road_b_middle.start], '-', city_list[road_b_middle.end], '这条连接(中继)\n')
                            adj_matrix[tmp_b][middle] = 1
                            adj_matrix[middle][tmp_b] = 1
                            # 是否添加了一条边的标志位置为True
                            triple_points_road_list.append([tmp_a, middle, tmp_b])
                            # print('triple', triple_points_road_list)
                            # 加入了一条边，删除这条边两边的城市对
                            city_pair_population_tuple_list.remove(city_tuple_list)
                            if_add_one_link = True
                            # 停止for middle in range(12):
                            break
        if not if_add_one_link:
            total_value += max_dir_link_value
            print("加入", city_list[max_value_road.start], '-', city_list[max_value_road.end], '这条连接(直接连接)\n')
            # 在邻接矩阵中表明这两个点已经连接
            adj_matrix[max_value_road.start][max_value_road.end] = 1
            adj_matrix[max_value_road.end][max_value_road.start] = 1
            cur_road_info_list.append(max_value_road)
            all_road_info_list.remove(max_value_road)

    print('\n目前地图中一共有', len(cur_road_info_list), '条边\n')
    print('总价值为', total_value)
