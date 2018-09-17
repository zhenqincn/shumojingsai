from question2.my_util import read_population_economic_from_excel
import math


city_population_list, _ = read_population_economic_from_excel()
double_city_population = []
quadra_city_population = []
for i in range(len(city_population_list)):
    for j in range(i + 1, len(city_population_list)):
        double_city_population.append(math.sqrt(city_population_list[i] * city_population_list[j]))

for i in range(len(city_population_list)):
    for j in range(i + 1, len(city_population_list)):
        for k in range(j + 1, len(city_population_list)):
            for n in range(k + 1, len(city_population_list)):
                tmp_city_list = [i, j, k, n]
                tmp_value_list = []
                for p in range(len(tmp_city_list)):
                    for q in range(p + 1, len(tmp_city_list)):
                        tmp_value_list.append(math.sqrt(city_population_list[tmp_city_list[p]] *
                                                                city_population_list[tmp_city_list[q]]))
                quadra_city_population.append(sum(tmp_value_list))

print(sorted(double_city_population, reverse=True))
print(sorted(quadra_city_population))
