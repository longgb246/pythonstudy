#-*- coding:utf-8 -*-
# 绿色：#6AB27B
# 土色：#a27712
# 浅紫色：#8172B2
# 蓝色：#4C72B0
# 红色：#C44E52
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 添加鼠标的组件
def plot_cursor():
    from matplotlib.widgets import Cursor
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = 4*(np.random.rand(2, 100) - .5)
    ax.plot(x, y, 'o')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    cursor = Cursor(ax, useblit=True, color='#C44E52', linewidth=2, linestyle='--')


# 组件，非常有意思
def plot_slider():
    from matplotlib.widgets import Slider, Button, RadioButtons, Cursor
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    s = a0*np.sin(2*np.pi*f0*t)
    l, = plt.plot(t, s, lw=2, color='red')
    plt.axis([0, 1, -10, 10])
    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))                 # 重设y的值
        fig.canvas.draw_idle()                                  # 马上画出来
    def reset(event):
        sfreq.reset()                                           # 组件重置
        samp.reset()
    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    # ====================== 1、添加水平轴 ======================
    axfreq = plt.axes([0.25, 0.08, 0.65, 0.03])
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03])                  # 添加一个轴
    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)       # 给轴变成 Slider 滚动条, valinit 表示初始化值
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)          # 最小值0.1，最大值10.0
    sfreq.on_changed(update)
    samp.on_changed(update)
    # ====================== 2、添加重置 ========================
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    button.on_clicked(reset)
    # ====================== 3、添加颜色 ========================
    rax = plt.axes([0.025, 0.5, 0.15, 0.2])
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    radio.on_clicked(colorfunc)
    # ====================== 4、添加鼠标 ========================
    cursor = Cursor(ax, useblit=True, color='#C44E52', linewidth=2, linestyle='--')


# ======================== 自己 ========================
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
class AnchoredText(AnchoredOffsetbox):
    def __init__(self, s, loc, pad=0.4, borderpad=0.5, prop=None, frameon=True):
        self.txt = TextArea(s, minimumdescent=False)
        super(AnchoredText, self).__init__(loc, pad=pad, borderpad=borderpad, child=self.txt, prop=prop, frameon=frameon)
# 重构复写函数
# class FooChild(FooParent):
#     def bar(self, message):
#         super(FooChild, self).bar(message)


def script_hist():
    '''
    画出有选项的正态分布
    '''
    from matplotlib.widgets import Cursor, Slider, Button, RadioButtons
    import matplotlib.ticker as ticker
    import matplotlib.mlab as mlab
    plt.style.use('ggplot')
    # plt.style.use('seaborn-darkgrid')
    # plt.style.use('classic')
    def sc_update_plot(data_num, first=False):
        # plt.清图，  plt.cla() # 清坐标轴。 plt.close()  # 关窗口
        # fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.clear()
        plt.subplots_adjust(bottom=0.25)
        data = np.random.randn(data_num)
        n, bins, patches = ax.hist(data, bins=20, color='#4C72B0', normed=True)
        ymin, ymax = plt.ylim()
        ax.set_ylim(ymin - ymax * 0.05, ymax * 1.05)
        mu = np.mean(data)
        sigma = np.std(data)
        y = mlab.normpdf(bins, mu, sigma)
        ax.plot(bins, y, '--', color='r')
        ax.set_title('Number of Samples : {0:.0f}'.format(data_num))
        at = AnchoredText("$\mu = {0:.4f}$ \n $\sigma = {1:.4f}$".format(mu, sigma), loc=1, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        # ax2 = ax.twinx()
        # ax2.plot(bins, y, '--', color='r')
        # ymin, ymax = plt.ylim()
        # ax2.set_ylim(ymin - ymax * 0.05, ymax * 1.05)
        # ax2.yaxis.set_major_formatter(ticker.NullFormatter())             # 设置空坐标
        # ax2.grid(False)
    def sc_update(val):
        sval = np.floor(sfreq.val)
        sc_update_plot(sval)
    def sc_reset(event):
        sfreq.reset()
    def sc_add_update(event):
        sval = np.floor(sfreq.val)
        if (0 < sval + 1000) and (10000 >= sval + 1000):
            sval = sval + 1000
        elif (0 < sval + 10000) and (100000 >= sval + 10000):
            sval = sval + 10000
        elif (100000 <= sval + 100000) and (1000000 >= sval + 100000):
            sval = sval + 100000
        else:
            sval = 1000000
        sfreq.set_val(sval)
        sc_update_plot(sval)
    def sc_min_update(event):
        sval = np.floor(sfreq.val)
        if (0 < sval - 100000) and (1000000 >= sval - 100000):
            sval = sval - 100000
        elif (0 < sval - 10000) and (100000 >= sval - 10000):
            sval = sval - 10000
        elif (0 < sval - 1000) and (10000 >= sval - 1000):
            sval = sval - 1000
        else:
            sval = 50
        sfreq.set_val(sval)
        sc_update_plot(sval)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cursor = Cursor(ax, useblit=True, color='#C44E52', linewidth=1, linestyle='--')
    data_num = 100000
    sc_update_plot(data_num, first=True)
    axfreq = plt.axes([0.2, 0.1, 0.65, 0.05])
    sfreq = Slider(axfreq, 'Number_n', 5, 1000000, valinit=data_num, valfmt='%1.f')
    axreset = plt.axes([0.43, 0.02, 0.2, 0.05])
    button = Button(axreset, 'Reset', hovercolor='0.975')
    button.on_clicked(sc_reset)
    sfreq.on_changed(sc_update)
    axmin = plt.axes([0.2, 0.02, 0.05, 0.05])
    button_min = Button(axmin, '-', hovercolor='0.975')
    button_min.on_clicked(sc_min_update)
    axadd = plt.axes([0.8, 0.02, 0.05, 0.05])
    button_add = Button(axadd, '+', hovercolor='0.975')
    button_add.on_clicked(sc_add_update)
    plt.show()
    # sfreq.set_val(2000)                           # 设置值


if __name__ == '__main__':
    # plot_cursor()
    # plot_slider()
    script_hist()

