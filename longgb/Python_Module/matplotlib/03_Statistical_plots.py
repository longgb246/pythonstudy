#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 1、箱型线  boxplot
def boxplot_color_demo():
    import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')
    np.random.seed(123)             # 设置种子
    all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    # boxplot的属性
    boxprops = dict(linewidth=2, facecolor='#4C72B0', alpha=0.3)
    whiskerprops = dict(linewidth=2, linestyle='--', color='#797979')
    flierprops = dict(linewidth=2, marker='o', markerfacecolor='none', markersize=6, linestyle='none')
    medianprops = dict(linestyle='-', linewidth=2.5, color='#FFA455')
    # meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='#C44E52')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='r', alpha=0.6)
    capprops = dict(linestyle='-', linewidth=2.5, color='r', alpha=0.6)
    # boxplot的添加属性
    bplot1 = axes[0].boxplot(all_data,
                             vert=True,                     # vertical box aligmnent
                             showmeans=True,                # 显示均值
                             meanline=True,                 # 均值使用线
                             patch_artist=True,             # fill with color
                             # showcaps=False,              # 边界横线
                             # showbox=False,               # 盒子展示
                             # showfliers=False,            # 异常值
                             boxprops=boxprops,             # 盒子属性
                             whiskerprops=whiskerprops,     # 虚线条属性
                             capprops=capprops,             # 边界横线
                             flierprops=flierprops,         # 异常值
                             medianprops=medianprops,       # 中位数  #FFA455   #797979    #3E3E3E
                             meanprops=meanlineprops        # 异常值
                             )
    bplot2 = axes[1].boxplot(all_data,
                             notch=True,                    # 有缺口形状
                             showmeans=True,
                             patch_artist=True,
                             meanline=True,                 # 均值使用线
                             vert=True,                     # vertical box aligmnent
                             boxprops=boxprops,             # 盒子属性
                             whiskerprops=whiskerprops,     # 虚线条属性
                             capprops=capprops,             # 边界横线
                             flierprops=flierprops,         # 异常值
                             medianprops=medianprops,       # 中位数  #FFA455   #797979    #3E3E3E
                             meanprops=meanlineprops        # 异常值
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
    # plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))], xticklabels=['x1', 'x2', 'x3'])
    axes[0].set_xticklabels(['x1', 'x2', 'x3'])
    axes[1].set_xticklabels(['x1', 'x2', 'x3'])
    plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))         # 设置左，上，右，下的位置比例
    plt.show()
    pass
# box的demo
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


# 2、小提琴图
def customized_violin_demo():
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5         # np.clip(a, 1, 8) 限制 a 的范围到1~8，小于1的为1，大于8的为8
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')
    np.random.seed(123)
    data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
    ax1.set_title('Default violin plot')
    ax1.set_ylabel('Observed values')
    ax1.violinplot(data, showmeans=True, showmedians=True,
                   showextrema=True
                   )
    ax2.set_title('Customized violin plot')
    parts = ax2.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    # set style for the axes
    labels = ['A', 'B', 'C', 'D']
    for ax in [ax1, ax2]:
        set_axis_style(ax, labels)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()
    pass


# 3、误差图
def errorbar_demo():
    # 只能做出以下效果
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=0.2, yerr=0.4)       # 画出 (x +- xerr),(y +- yerr)
    plt.show()
    pass
# error 图
def errorbar_demo_features():
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    error = 0.1 + 0.2 * x
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.errorbar(x, y, yerr=error, fmt='-o')        # error 要和 x、y 的数据一样，如果只是一列，则是对称的
    ax0.set_title('variable, symmetric error')
    lower_error = 0.4 * error
    upper_error = error
    asymmetric_error = [lower_error, upper_error]
    ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')      # [lower_error, upper_error] 上下线的list，则数据有上下限不一样
    ax1.set_title('variable, asymmetric error')
    ax1.set_yscale('log')
    plt.show()
    pass
# 四面八方 error 图
def errorbar_limits():
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    y = np.exp(-x)
    xerr = 0.1
    yerr = 0.2
    lolims = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    uplims = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    ls = 'dotted'
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)
    ax.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims, linestyle=ls)
    ax.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims, linestyle=ls)
    ax.errorbar(x, y + 1.5, xerr=xerr, yerr=yerr, lolims=lolims, uplims=uplims,
                marker='o', markersize=8, linestyle=ls)
    xerr = 0.2
    yerr = np.zeros(x.shape) + 0.2
    yerr[[3, 6]] = 0.3
    # mock up some limits by modifying previous data
    xlolims = lolims
    xuplims = uplims
    lolims = np.zeros(x.shape)
    uplims = np.zeros(x.shape)
    lolims[[6]] = True  # only limited at this index
    uplims[[3]] = True  # only limited at this index
    # do the plotting
    ax.errorbar(x, y + 2.1, xerr=xerr, yerr=yerr,
                xlolims=xlolims, xuplims=xuplims,
                uplims=uplims, lolims=lolims,
                marker='o', markersize=8,
                linestyle='none')
    # tidy up the figure
    ax.set_xlim((0, 5.5))
    ax.set_title('Errorbar upper and lower limits')
    plt.show()
    pass


# 4、histogram 图
def histogram_demo_features():
    import matplotlib.mlab as mlab
    np.random.seed(0)
    # example data
    mu = 100  # mean of distribution
    sigma = 15  # standard deviation of distribution
    x = mu + sigma * np.random.randn(437)
    num_bins = 50
    fig, ax = plt.subplots()
    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, '--')
    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    pass
# 画的是没有边界的 histogram 图，不等宽的 histogram 图
def histogram_demo_histtypes():
    np.random.seed(0)
    mu = 200
    sigma = 25
    x = np.random.normal(mu, sigma, size=100)
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
    ax0.hist(x, 20, normed=1, histtype='stepfilled', facecolor='g', alpha=0.75)
    ax0.set_title('stepfilled')
    # Create a histogram by providing the bin edges (unequally spaced).
    bins = [100, 150, 180, 195, 205, 220, 250, 300]
    ax1.hist(x, bins, normed=1, histtype='bar', rwidth=0.8)
    ax1.set_title('unequal bins')
    fig.tight_layout()
    plt.show()
    pass



# test 画局部图   dpi
# ax.text(tx, ty, label_f0, fontsize=15, verticalalignment="top", horizontalalignment="left")
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
