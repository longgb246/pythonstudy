# coding=utf-8

import numpy as np
from scipy.stats import rv_discrete


np.random.seed(1234)


class Test:

    def __init__(self):
        self.val = np.array([1, 2, 3, 4])
        self.prob = np.array([0.25, 0.25, 0.25, 0.25])
        self.dist = rv_discrete(values=(self.val, self.prob))

    def rand_number(self):
        return self.dist.rvs()

    def run_simulation(self):
        result = []
        index = 0
        while index < 10:
            result.append(self.rand_number())
        return result

if __name__ == '__main__':
    test = Test()
    print test.run_simulation()
