#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 1、箱型线  boxplot
def boxplot_color_demo():
    import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')
    np.random.seed(123)             # 设置种子
    all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = axes[0].boxplot(all_data,
                             vert=True,         # vertical box aligmnent
                             showmeans=True,    # 显示均值
                             # meanline=True,   # 均值使用线
                             patch_artist=True  # fill with color
                             )
    bplot2 = axes[1].boxplot(all_data,
                             notch=True,        # 有缺口形状
                             showmeans=True,
                             vert=True,         # vertical box aligmnent
                             patch_artist=True
                             )
    colors = ['pink', 'lightblue', 'lightgreen']
    # 添加 box 的颜色
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(all_data))], )
        ax.set_xlabel('xlabel')
        ax.set_ylabel('ylabel')
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=['x1', 'x2', 'x3'])
    axes[0].set_xticklabels(['x1', 'x2', 'x3'])
    axes[1].set_xticklabels(['x1', 'x2', 'x3'])
    plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))         # 设置左，上，右，下的位置比例
    plt.show()
    pass
#
def boxplot_demo():
    # fake data
    np.random.seed(937)
    data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
    labels = list('ABCD')
    fs = 10  # fontsize
    # demonstrate how to toggle the display of different elements:
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
    axes[0, 0].boxplot(data, labels=labels)
    axes[0, 0].set_title('Default', fontsize=fs)
    axes[0, 1].boxplot(data, labels=labels, showmeans=True)
    axes[0, 1].set_title('showmeans=True', fontsize=fs)
    axes[0, 2].boxplot(data, labels=labels, showmeans=True, meanline=True)
    axes[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)
    axes[1, 0].boxplot(data, labels=labels, showbox=False, showcaps=False)
    tufte_title = 'Tufte Style \n(showbox=False,\nshowcaps=False)'
    axes[1, 0].set_title(tufte_title, fontsize=fs)
    axes[1, 1].boxplot(data, labels=labels, notch=True, bootstrap=10000)
    axes[1, 1].set_title('notch=True,\nbootstrap=10000', fontsize=fs)
    axes[1, 2].boxplot(data, labels=labels, showfliers=False)
    axes[1, 2].set_title('showfliers=False', fontsize=fs)
    for ax in axes.flatten():
        ax.set_yscale('log')
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=0.4)
    plt.show()
    # demonstrate how to customize the display different elements:
    boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
    flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                      linestyle='none')
    medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
    axes[0, 0].boxplot(data, boxprops=boxprops)
    axes[0, 0].set_title('Custom boxprops', fontsize=fs)
    axes[0, 1].boxplot(data, flierprops=flierprops, medianprops=medianprops)
    axes[0, 1].set_title('Custom medianprops\nand flierprops', fontsize=fs)
    axes[0, 2].boxplot(data, whis='range')
    axes[0, 2].set_title('whis="range"', fontsize=fs)
    axes[1, 0].boxplot(data, meanprops=meanpointprops, meanline=False,
                       showmeans=True)
    axes[1, 0].set_title('Custom mean\nas point', fontsize=fs)
    axes[1, 1].boxplot(data, meanprops=meanlineprops, meanline=True,
                       showmeans=True)
    axes[1, 1].set_title('Custom mean\nas line', fontsize=fs)
    axes[1, 2].boxplot(data, whis=[15, 85])
    axes[1, 2].set_title('whis=[15, 85]\n#percentiles', fontsize=fs)
    for ax in axes.flatten():
        ax.set_yscale('log')
        ax.set_yticklabels([])
    fig.suptitle("I never said they'd be pretty")
    fig.subplots_adjust(hspace=0.4)
    plt.show()
    pass
# 画累计分布函数
def histogram_demo_cumulative():
    from matplotlib import mlab
    np.random.seed(0)
    mu = 200
    sigma = 25
    n_bins = 50
    x = np.random.normal(mu, sigma, size=100)
    fig, ax = plt.subplots(figsize=(8, 4))
    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, normed=1, histtype='step',
                               cumulative=True, label='Empirical')          # 累积分布函数
    # Add a line showing the expected distribution.
    y = mlab.normpdf(bins, mu, sigma).cumsum()                  # 展现期望分布
    y /= y[-1]
    ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
    # Overlay a reversed cumulative histogram.
    ax.hist(x, bins=bins, normed=1, histtype='step', cumulative=-1,
            label='Reversed emp.')
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Annual rainfall (mm)')
    ax.set_ylabel('Likelihood of occurrence')
    plt.show()
    pass

# test 画局部图   dpi
# p1.text(tx, ty, label_f0, fontsize=15, verticalalignment="top", horizontalalignment="left")
def test():
    # import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')
    from matplotlib.patches import ConnectionPatch
    def f1(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)
    def f11(t):
        return np.exp(-t) * np.cos(2 * np.pi * t + 0.2)
    def f111(t):
        return np.exp(-t + 0.2) * np.cos(2 * np.pi * t)
    t = np.arange(0.0, 5.0, 0.02)
    label_f0 = r"$f(t)=e^{-t+\alpha} \cos (2 \pi t+\beta)$"
    label_f1 = r"$\alpha=0,\beta=0$"
    label_f11 = r"$\alpha=0,\beta=0.2$"
    label_f111 = r"$\alpha=0.2,\beta=0$"
    tx = 0.5                # 添加 text 文本
    ty = 0.9
    fig = plt.figure(figsize=(16, 8), dpi=98)             # 加个dpi调整。
    # ax1 = fig.add_subplot(121)
    ax1 = fig.add_subplot(121,aspect=5/2.5)
    # ax2 = fig.add_subplot(122)
    ax2 = fig.add_subplot(122,aspect=0.5/0.05)
    ax1.plot(t, f1(t), "g", label=label_f1, linewidth=2)
    ax1.plot(t, f11(t), "r-.", label=label_f11, linewidth=2)
    ax1.plot(t, f111(t), "b:", label=label_f111, linewidth=2)
    ax2.plot(t, f1(t), "g", label=label_f1, linewidth=2)
    ax2.plot(t, f11(t), "r-.", label=label_f11, linewidth=2)
    ax2.plot(t, f111(t), "b:", label=label_f111, linewidth=2)
    ax1.axis([0.0, 5.01, -1.0, 1.5])
    ax1.set_ylabel("v", fontsize=14)
    ax1.set_xlabel("t", fontsize=14)
    # p1.set_title("A simple example",fontsize=18)
    ax1.grid(True)
    ax1.legend()
    ax1.text(tx, ty, label_f0, fontsize=15, verticalalignment="top", horizontalalignment="left")

    ax2.axis([4, 4.5, -0.02, 0.03])          # 设置不同的范围
    ax2.set_ylabel("v", fontsize=14)
    ax2.set_xlabel("t", fontsize=14)
    ax2.grid(True)
    ax2.legend()
    # plot the box
    tx0 = 4
    tx1 = 4.5
    ty0 = -0.1
    ty1 = 0.1
    sx = [tx0, tx1, tx1, tx0, tx0]          # 画方框
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax1.plot(sx, sy, "purple")
    # plot patch lines
    xy = (4.45, 0.09)
    xy2 = (4.02, 0.026)
    # 重点：连接线
    con = ConnectionPatch(xyA=xy2, xyB=xy,
                          coordsA="data", coordsB="data",           # 这个参数必须要！
                          axesA=ax2, axesB=ax1)
    ax2.add_artist(con)                      # 在p2上添加
    xy = (4.45, -0.09)
    xy2 = (4.02, -0.018)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1)
    ax2.add_artist(con)
    plt.show()
    pass


if __name__ == '__main__':
    test()
    pass
