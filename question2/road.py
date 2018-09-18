class Road(object):
    def __init__(self, start, end, weight, capacity, population, value):
        self.start = start
        self.end = end
        self.weight = weight
        self.capacity = capacity
        self.population = population
        self.value = value

    def __str__(self):
        return '(' + str(self.start) + ', ' + str(self.end) + ')'
