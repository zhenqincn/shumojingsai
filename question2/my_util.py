
def read_distance_from_excel():
    """
    从excel中读取各个地区之间的直线距离信息
    :return:
    """
    import xlrd
    ExcelFile = xlrd.open_workbook('城市信息统计表.xlsx')
    sheet = ExcelFile.sheet_by_name('各城市之间距离')
    city_list = []
    distance_matrix = [[0.0 for _ in range(12)] for _ in range(12)]
    for i in range(1, 13):
        city_list.append(str(sheet.cell(i, 0).value))
    for i in range(0, 12):
        for j in range(0, 12):
            if i > j:
                distance_matrix[i][j] = float(sheet.cell(i + 1, j + 1).value)
            else:
                distance_matrix[i][j] = float(sheet.cell(j + 1, i + 1).value)
    return distance_matrix, city_list


def read_population_economic_from_excel():
    """
    从excel中读取各个地区的经济、人口状况
    :return:
    """
    import xlrd
    ExcelFile = xlrd.open_workbook('城市信息统计表.xlsx')
    sheet = ExcelFile.sheet_by_name('各城市人口经济状况')
    city_population_list = []
    city_gdp_aver_list = []
    for i in range(0, 12):
        city_population_list.append(float(sheet.cell(i + 1, 1).value) / 100)  # 以百万为单位
        city_gdp_aver_list.append(float(sheet.cell(i + 1, 4).value))
    return city_population_list, city_gdp_aver_list


def read_population_economic_from_excel_province():
    """
    从excel中读取各个地区的经济、人口状况，人口状况为省的人口状况
    :return:
    """
    import xlrd
    ExcelFile = xlrd.open_workbook('城市信息统计表_省.xlsx')
    sheet = ExcelFile.sheet_by_name('各城市人口经济状况')
    city_population_list = []
    city_gdp_aver_list = []
    for i in range(0, 12):
        city_population_list.append(float(sheet.cell(i + 15, 1).value) / 100)  # 以百万为单位
        city_gdp_aver_list.append(float(sheet.cell(i + 1, 4).value))
    return city_population_list, city_gdp_aver_list
