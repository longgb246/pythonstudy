# coding=utf-8
import os
from sys import path
pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
path.append(pth)
from scipy.stats import norm
from scipy.optimize import bisect
import numpy as np


class MixtureGaussian:

    def __init__(self, mu, sigma, weight):
        self.mu = mu
        self.sigma = sigma
        self.weight = weight
        self.step = max(self.sigma)+1

    def cdf(self, x):
        return sum([k * norm.cdf((x-i)/(j+1e-4)) for i, j, k in zip(self.mu, self.sigma, self.weight)])

    def inverse_cdf(self, alpha, xtol=1e-4):
        """给定0到1之间的概率值，返回对应的分位数"""

        if alpha == 1:
            return MixtureGaussian.inverse_cdf(self, 0.99, xtol)

        # 定义要求解的方程
        def f(x):
            return self.cdf(x) - alpha

        # 确定搜索区间[a, b]，保证[a, b]包含方程的根
        a = min(self.mu)-self.step
        b = max(self.mu)+self.step
        while f(a) > 0:
            a -= self.step
        while f(b) <= 0:
            b += self.step

        return bisect(f, a=a, b=b, xtol=xtol)


if __name__ == '__main__':

    mu_list = [4., 10., 13., 19.]
    sigma_list = [2.]
    weight_list = [1.]

    mg = MixtureGaussian(mu_list, sigma_list, weight_list)

    cr_list = [0.95, 0.99, 1]
    for p in cr_list:
        print('Quantile of ' + str(p) + ' is ' + str(mg.inverse_cdf(p)))
