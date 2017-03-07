#-*- coding:utf-8 -*-

# 1、画分布图
# 1.1 需要引入的包
import numpy as np
import matplotlib.pyplot as plt
# 1.2 主函数
def plotHistPer(plot_data, binsn=[], xlabeln='x', ylabeln='y', titlen='', save_path='', cum_True=True, size=(12,8), is_int=True, is_save=False, is_drop_zero=False):
    '''
    画hist的百分比图，指定bins
    :param data: pd.DataFrame 单列数据
    :param binsn: numeric 指定的bins
    :param xlabeln: unicode x轴名称
    :param ylabeln: unicode y轴名称
    :param titlen: unicode 图的标题
    :param save_path: string 文件路径
    :param cum_True: boolean 是否添加累计概率线
    :param size: tuple 画图的size大小
    :param is_int: boolean 是否把标签变成整数
    :return: None 仅用于作图
    '''
    # plot_data=z_value_frame.z_value; binsn=[-np.inf, 0, 2, 4, 6, 8, 10, 12, 14, np.inf]
    # xlabeln = u'z值'; ylabeln = u'频数'; titlen = u"Z值分布图"; size=(12,8); intshu=True
    plt.style.use('seaborn-darkgrid')
    if binsn == []:
        ret = plt.hist(plot_data, label='Z', color='#0070C0', histtype='bar', rwidth=0.6)
    else:
        ret = plt.hist(plot_data, bins=binsn, label='Z', color='#0070C0',histtype='bar', rwidth=0.6)
    plt.close()
    counts, bins, patches = ret[0], ret[1], ret[2]
    if is_int:
        bins = map(lambda x: int(x) if (x != -np.inf) and (x != np.inf) and (x != np.nan) else x,bins)
    bins_name = ["["+str(bins[i])+","+str(bins[i+1])+")" for i in range(len(bins)-1)]
    if is_drop_zero:
        tmp_counts = []
        tmp_bins_name = []
        for i, each in enumerate(counts):
            if each != 0:
                tmp_counts.append(counts[i])
                tmp_bins_name.append(bins_name[i])
        counts = tmp_counts
        bins_name = tmp_bins_name
    ind = np.arange(len(counts))
    fig1, ax1 = plt.subplots(figsize=size)
    ax1.set_xlabel(xlabeln)
    ax1.set_ylabel(ylabeln)
    width = 0.5
    width2 = 0
    ax1.bar(ind + width2, counts, width, color="#0070C0", tick_label=bins_name, align='center', alpha=0.6)
    counts_per = counts/np.sum(counts)
    counts_per_cum = np.cumsum(counts_per)
    i = 0
    ymin, ymax = plt.ylim()
    ax1.set_ylim(ymin - ymax * 0.05, ymax * 1.05)
    # ax1.set_xlim(-1, len(bins_name)+1)
    for x, y in zip(ind, counts):
        ax1.text(x + width2, y + 0.05, '{0:.2f}%'.format(counts_per[i]*100), ha='center', va='bottom')
        i += 1
    plt.title(titlen)
    if cum_True:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative probability distribution')
        ax2.plot(ind + width2, counts_per_cum, '--', color="red")
        ax2.yaxis.grid(False)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(-0.5, len(bins_name) - 0.5)
    plt.show()
    if is_save:
        plt.savefig(save_path)
# 1.3 demo运行函数
def plotHistPerDemo():
    plot_data = np.random.randint(0,100,1000)
    plothistper(plot_data)
    plothistper(plot_data, is_drop_zero=True)
    pass


# 2、画放大图
# 2.1 需要引入的包
from matplotlib.patches import ConnectionPatch
# 参考代码
# test 画局部图   dpi
# ax.text(tx, ty, label_f0, fontsize=15, verticalalignment="top", horizontalalignment="left")
def plotEnlarge(data_x, data_y, scale=[], label=[], colors=[], linestyle=[], xlabel='X', ylabel='Y', title=''):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(16, 8), dpi=98)  # 加个dpi调整。
    ax1 = fig.add_subplot(121, aspect=5 / 2.5)
    ax2 = fig.add_subplot(122, aspect=0.5 / 0.05)
    if colors == []:
        colors = ['#6AB27B', '#C44E52', '#4C72B0', '#FFA455']
    if linestyle == []:
        linestyle = ['-', '--', '-.', ':']
    for i, x in enumerate(data_x):
        label_tmp = 'line_{0}'.format(i + 1) if label == [] else label[i]
        ax1.plot(x, data_y[i], color=colors[i], linestyle=linestyle[i], label=label_tmp, linewidth=2)
        ax2.plot(x, data_y[i], color=colors[i], linestyle=linestyle[i], label=label_tmp, linewidth=2)
    # ax1.axis([0.0, 5.01, -1.0, 1.5])
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_title(title,fontsize=18)
    ax1.grid(True)
    ax1.legend()
    # ax1.text(tx, ty, label_f0, fontsize=15, verticalalignment="top", horizontalalignment="left")
    if scale == []:
        tx0 = 4
        tx1 = 4.5
        ty0 = -0.1
        ty1 = 0.1
    else:
        tx0 = scale[0]
        tx1 = scale[1]
        ty0 = scale[2]
        ty1 = scale[3]
    ax2.axis([tx0, tx1, ty0, ty1])          # 设置不同的范围
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.grid(True)
    ax2.legend()
    sx = [tx0, tx1, tx1, tx0, tx0]          # 画方框
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax1.plot(sx, sy, "purple")
    # plot patch lines
    a = 0.05
    xy = (tx1 - a, ty1 - a)
    xy2 = (tx0 + a, ty0 + a)
    # 重点：连接线
    con = ConnectionPatch(xyA=xy2, xyB=xy,
                          coordsA="data", coordsB="data",           # 这个参数必须要！
                          axesA=ax2, axesB=ax1)
    ax2.add_artist(con)                      # 在p2上添加
    xy = (tx1 - a, ty1 - a)
    xy2 = (tx0 + a, ty0 + a)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1)
    ax2.add_artist(con)
    plt.show()
    pass


def plotEnlargeDemo():
    def f1(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)
    def f11(t):
        return np.exp(-t) * np.cos(2 * np.pi * t + 0.2)
    def f111(t):
        return np.exp(-t + 0.2) * np.cos(2 * np.pi * t)
    data_x = np.arange(0.0, 5.0, 0.02)
    data_y = f1(data_x)
    label_f0 = r"$f(t)=e^{-t+\alpha} \cos (2 \pi t+\beta)$"
    label_f1 = r"$\alpha=0,\beta=0$"
    label_f11 = r"$\alpha=0,\beta=0.2$"
    label_f111 = r"$\alpha=0.2,\beta=0$"
    tx = 0.5  # 添加 text 文本
    ty = 0.9



if __name__ == '__main__':
    plotHistPerDemo()
    pass

