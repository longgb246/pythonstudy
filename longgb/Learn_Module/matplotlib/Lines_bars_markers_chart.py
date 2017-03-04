#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


# 1、画横着的 bar 图, ax.barh
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


# 2、画填充图, ax.fill
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
    # -------------------------- 画两个函数之间的图 --------------------------
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
    # -------------------------- 画两个函数(区分函数大小)的图 --------------------------
    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)
    ax.plot(x, y1, x, y2, color='black')
    ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
    ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
    ax.set_title('fill between where')
    y2 = np.ma.masked_greater(y2, 1.0)
    ax1.plot(x, y1, x, y2, color='black')
    ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
    ax1.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
    ax1.set_title('Now regions with y2>1 are masked')
    # 不知道画什么图
    fig, ax = plt.subplots()
    y = np.sin(4 * np.pi * x)
    ax.plot(x, y, color='black')
    # use the data coordinates for the x-axis and the axes coordinates for the y-axis
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    theta = 0.9
    ax.axhline(theta, color='green', lw=2, alpha=0.5)
    ax.axhline(-theta, color='red', lw=2, alpha=0.5)
    ax.fill_between(x, 0, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(x, 0, 1, where=y < -theta, facecolor='red', alpha=0.5, transform=trans)
def fill_demo_features():
    x = np.linspace(0, 2*np.pi, 500)
    x_max = np.max(x) + (np.max(x) - np.min(x))*0.05
    x_min = np.min(x) - (np.max(x) - np.min(x))*0.05
    y1 = np.sin(x)
    y2 = np.sin(x*3)
    y_max = np.max([y1,y2]) + (np.max([y1,y2]) - np.min([y1,y2]))*0.05
    y_min = np.min([y1,y2]) - (np.max([y1,y2]) - np.min([y1,y2]))*0.05
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.fill(x, y1, 'b', alpha=0.3)
    ax.fill(x, y2, 'r', alpha=0.3)
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    plt.show()

def line_demo_dash_control():
    x = np.linspace(0, 10, 500)
    dashes = [10, 5, 100, 5]        # 10 points on, 5 off, 100 on, 5 off
    fig = plt.figure(figsize=(18,7))
    ax = fig.add_subplot(121)
    line1, = ax.plot(x, np.sin(x), '--', linewidth=2, label='Dashes set retroactively')     # 之所以用“，”，是因为想直接取返回的list中的第一个元素
    ax2 = fig.add_subplot(122)
    line2, = ax2.plot(x, np.sin(x), '--', linewidth=2, label='Dashes set retroactively')  # 之所以用“，”，是因为想直接取返回的list中的第一个元素
    line2.set_dashes(dashes)
    line2, = ax.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5], label='Dashes set proactively')


    ax.legend(loc='lower right')
    plt.show()



if __name__ == '__main__':
    pass


