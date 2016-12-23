from scipy.stats import poisson
from scipy.stats import geom
import seaborn as sns
import numpy as np


class StutteringPoisson:

    def __init__(self, m, p):
        self.m = m
        self.p = p

    def rvs(self, size):
        pois = poisson.rvs(self.m, size=size)
        return [np.sum(geom.rvs(self.p, size=i)) for i in pois.tolist()]


if __name__ == '__main__':

    sp = StutteringPoisson(10, 0.2)
    rs = sp.rvs(100)
    print(np.mean(rs))
    print(np.var(rs))
