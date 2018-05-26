#-*- coding:utf-8 -*-
"""
傅里叶级数变换。
"""

# %matplotlib gtk
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-10, 10, 0.1)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, 2*np.sin(x))
ax.plot(x, 2*np.sin(x)-np.sin(2*x))
ax.plot(x, 2*np.sin(x)-np.sin(2*x)+2.0/3*np.sin(3*x))
ax.plot(x, 2*np.sin(x)-np.sin(2*x)+2.0/3*np.sin(3*x)-0.5*np.sin(5*x))
plt.show()



