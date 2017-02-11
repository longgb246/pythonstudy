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


if __name__ == '__main__':
    plot_cursor()
    plot_slider()
