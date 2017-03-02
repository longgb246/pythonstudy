#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def barh_demo():
    # fig, ax = plt.subplots()
    people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
    y_pos = np.arange(len(people))
    performance = 3 + 10 * np.random.rand(len(people))
    error = np.random.rand(len(people))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(y_pos, performance, xerr=error, align='center', color='green', ecolor='black')      # xerr表示上下误差
    ax.set_yticks(y_pos)                        # 设置y轴的刻度范围，原来是[-1,5]，从-1到5，设置后是从0-4， 与 ax.set_yticks([0, 4]) 等价
    ax.set_yticklabels(people)                  # 设置y轴的刻度标识
    ax.invert_yaxis()                           # 把y轴的方向倒转过来
    ax.set_xlabel('Performance')
    ax.set_title('How fast do you want to go today?')


if __name__ == '__main__':
    pass


