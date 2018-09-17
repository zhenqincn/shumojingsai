class Road(object):
    def __init__(self, start, end, weight, capacity, population):
        self.start = start
        self.end = end
        self.weight = weight
        self.capacity = capacity
        self.population = population

    def __str__(self):
        return '(' + str(self.start) + ', ' + str(self.end) + ')'