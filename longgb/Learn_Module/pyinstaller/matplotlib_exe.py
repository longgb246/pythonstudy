#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def draw():
    X = np.linspace(0, 10.0, 100)
    Y = np.sin(X)
    plt.plot(X, Y, '-', color='g')
    plt.show()

if __name__ == '__main__':
    draw()
