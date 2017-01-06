from scipy.stats import poisson
from scipy.stats import geom
import seaborn as sns
import numpy as np

p = poisson.rvs(10, size=100000)
g = geom.rvs(0.5, size=1000)

sns.distplot(g)

sp = [np.sum(geom.rvs(0.2, size=i)) for i in p.tolist()]


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(sum(map(ord, "aesthetics")))


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data)