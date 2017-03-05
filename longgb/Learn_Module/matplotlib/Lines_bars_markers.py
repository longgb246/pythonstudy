#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


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
# 简单的图
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


# 3、设置线(Line)、点(Marker)
def line_demo_dash_control():
    x = np.linspace(0, 10, 500)
    dashes = [10, 5, 100, 5]        # 10 points 画, 5 不画, 100 画, 5 不画，点数根据 x 的linspace的那个决定
    fig = plt.figure(figsize=(18,7))
    ax = fig.add_subplot(121)
    line1, = ax.plot(x, np.sin(x), '--', linewidth=2, label='Dashes set retroactively')     # 之所以用“，”，是因为想直接取返回的list中的第一个元素
    ax2 = fig.add_subplot(122)
    line2, = ax2.plot(x, np.sin(x), '--', linewidth=2, label='Dashes set retroactively')  # 之所以用“，”，是因为想直接取返回的list中的第一个元素
    line2.set_dashes(dashes)
    line2, = ax2.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5], label='Dashes set proactively')
    ax2.legend(loc='lower right')
    plt.show()
# 3.1、nice_repr 为有用函数，处理原unicode的编码变为str
# 3.2、ax.margins(0.2)    ax.set_axis_off()    ax.text(-0.1, y, nice_repr(linestyle), **text_style)
# text_style
def line_styles_reference():
    color = 'cornflowerblue'
    points = np.ones(5)  # Draw 5 points for each line
    text_style = dict(horizontalalignment='right', verticalalignment='center', fontsize=12, fontdict={'family': 'monospace'})
    # text_style 是一个 dict 的属性
    def format_axes(ax):
        ax.margins(0.2)                         # 设置自动边距
        ax.set_axis_off()                       # 把坐标轴取消
    def nice_repr(text):
        return repr(text).lstrip('u')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    linestyles = ['-', '--', '-.', ':']
    for y, linestyle in enumerate(linestyles):
        # y = 2
        # linestyle = linestyles[2]
        ax.text(-0.1, y, nice_repr(linestyle), **text_style)        # 这个地方使用属性的时候，要使用**在前面
        # ax.text(-0.1, y, nice_repr(linestyle), text_style)
        ax.plot(y * points, linestyle=linestyle, color=color, linewidth=3)
        format_axes(ax)
        ax.set_title('line styles')
    plt.show()
# 3.3、plt.tight_layout()   ax.annotate()   ax.set_xticks([])   ax.set_yticks()   ax.set_yticklabels()
def linestyles():
    from collections import OrderedDict             # 引入有序的字典，不改变字典的顺序
    # from matplotlib.transforms import blended_transform_factory
    linestyles = OrderedDict(
        [('solid', (0, ())),
         ('loosely dotted', (0, (1, 10))),
         ('dotted', (0, (1, 5))),
         ('densely dotted', (0, (1, 1))),
         ('loosely dashed', (0, (5, 10))),
         ('dashed', (0, (5, 5))),
         ('densely dashed', (0, (5, 1))),
         ('loosely dashdotted', (0, (3, 10, 1, 10))),
         ('dashdotted', (0, (3, 5, 1, 5))),
         ('densely dashdotted', (0, (3, 1, 1, 1))),
         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    # 开始画图
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    X, Y = np.linspace(0, 100, 10), np.zeros(10)
    for i, (name, linestyle) in enumerate(linestyles.items()):
        ax.plot(X, Y + i, linestyle=linestyle, linewidth=1.5, color='black')
    ax.set_ylim(-0.5, len(linestyles) - 0.5)
    # plt.yticks(np.arange(len(linestyles)), linestyles.keys())
    ax.set_yticks(np.arange(len(linestyles)))
    ax.set_yticklabels(linestyles.keys())
    # plt.xticks([])
    # ax.set_xticklabels([''])
    ax.set_xticks([])
    # reference_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (name, linestyle) in enumerate(linestyles.items()):
        ax.annotate(str(linestyle),
                    xy=(0.0, i),
                    # xycoords=reference_transform,
                    xytext=(-6, -14),        # 标注的位置，(x,y)，x为正表示向右，负表示向左；y为正表示向上，负表示向下
                    textcoords='offset points',     # 'offset pixels' 不知道是什么意思，但是这个属性必须有
                    color="blue",
                    fontsize=8,
                    ha="right",              # right：以右为基准向左边开始写；left：以左为基准向右边开始写；
                    family="monospace")
    plt.tight_layout()                  # 使得布局可以全部布满，并且都显示出来
    plt.show()
    pass
# text_style    marker_style    Line2D
def marker_fillstyle_reference():
    from matplotlib.lines import Line2D
    def format_axes(ax):
        ax.margins(0.2)
        ax.set_axis_off()
    def nice_repr(text):
        return repr(text).lstrip('u')
    points = np.ones(5)  # Draw 3 points for each line
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                      fontsize=12, fontdict={'family': 'monospace'})
    marker_style = dict(color='cornflowerblue', linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='gray')
    fig, ax = plt.subplots()
    # Plot all fill styles.
    for y, fill_style in enumerate(Line2D.fillStyles):
        ax.text(-0.5, y, nice_repr(fill_style), **text_style)
        ax.plot(y * points, fillstyle=fill_style, **marker_style)
        format_axes(ax)
        ax.set_title('fill style')
    plt.show()
    pass
# text_style    marker_style    Line2D
def marker_reference():
    from six import iteritems
    from matplotlib.lines import Line2D
    def format_axes(ax):
        ax.margins(0.2)
        ax.set_axis_off()
    def nice_repr(text):
        return repr(text).lstrip('u')
    def split_list(a_list):
        i_half = len(a_list) // 2
        return (a_list[:i_half], a_list[i_half:])
    points = np.ones(3)  # Draw 3 points for each line
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                      fontsize=12, fontdict={'family': 'monospace'})
    marker_style = dict(linestyle=':', color='cornflowerblue', markersize=10)
    fig, axes = plt.subplots(ncols=2)
    unfilled_markers = [m for m, func in iteritems(Line2D.markers)
                        if func != 'nothing' and m not in Line2D.filled_markers]
    unfilled_markers = sorted(unfilled_markers,
                              key=lambda x: (str(type(x)), str(x)))[::-1]
    for ax, markers in zip(axes, split_list(unfilled_markers)):
        for y, marker in enumerate(markers):
            ax.text(-0.5, y, nice_repr(marker), **text_style)
            ax.plot(y * points, marker=marker, **marker_style)
            format_axes(ax)
    fig.suptitle('un-filled markers', fontsize=14)
    fig, axes = plt.subplots(ncols=2)
    for ax, markers in zip(axes, split_list(Line2D.filled_markers)):
        for y, marker in enumerate(markers):
            ax.text(-0.5, y, nice_repr(marker), **text_style)
            ax.plot(y * points, marker=marker, **marker_style)
            format_axes(ax)
    fig.suptitle('filled markers', fontsize=14)
    plt.show()
    pass


# 4、散点图
# 4.1、散点图(指定点大小，把点边界取消)+legend透明的
# ax.scatter(edgecolors='none')   ax.legend(framealpha=0.7)
def scatter_with_legend():
    fig, ax = plt.subplots()
    for color in ['red', 'green', 'blue']:
        n = 750
        x, y = np.random.rand(2, n)
        scale = 200.0 * np.random.rand(n)
        ax.scatter(x, y, c=color, s=scale, label=color, alpha=0.3
                   ,edgecolors='none'   # 把边界取消
                   )
    ax.legend(framealpha=0.7)
    # ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    pass


