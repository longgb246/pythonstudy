#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


# 画横着的 bar 图, ax.barh
def barh_demo():
    '''
    画横着的bar图
    '''
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


# 画填充图, ax.fill
def fill_demo():
    x = np.linspace(0, 1, 500)
    y = np.sin(4 * np.pi * x) * np.exp(-5 * x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill(x, y)
    ax.grid(True)
    plt.show()
# 填充图，ax.fill_between
def fill_between_demo():
    x = np.arange(0.0, 2, 0.01)
    y1 = np.sin(2 * np.pi * x)
    y2 = 1.2 * np.sin(4 * np.pi * x)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)          # sharex 分享x轴坐标刻度，三个图都有x轴刻度，否则只有最下面的有
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # 分开添加不可以指定是否共享同一 x 坐标轴。以后使用上面的形式开启图片。
    # fig = plt.figure()
    # ax1 = fig.add_subplot(311)
    # ax1.set_xticklabels([''])
    # ax2 = fig.add_subplot(312)
    # ax2.set_xticklabels([''])
    # ax3 = fig.add_subplot(313)

    ax1.fill_between(x, 0, y1)
    ax1.set_ylabel('between y1 and 0')
    ax2.fill_between(x, y1, 1)
    ax2.set_ylabel('between y1 and 1')
    ax3.fill_between(x, y1, y2)
    ax3.set_ylabel('between y1 and y2')
    ax3.set_xlabel('x')

if __name__ == '__main__':
    pass


