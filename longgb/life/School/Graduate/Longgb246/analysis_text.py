#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import jieba
import json
import matplotlib.pyplot as plt


jieba_words = r'F:\WorkSpace\pythonstudy\longgb\life\School\Graduate\Longgb246\jieba_words.txt'


def test():
    jieba.load_userdict(jieba_words)

    # 1、有问题！！
    text = '和平主义。'
    jieba.add_word('和平', freq=100000)
    jieba.suggest_freq(('和','平','主','义'), True)
    a = list(jieba.cut(text, HMM=False))
    a = jieba.lcut(text, HMM=False)
    print '\\'.join(a)


    # 2、有问题！！
    text4 = '乒乓球拍卖完了'
    jieba.add_word('乒乓球拍')
    jieba.suggest_freq(('拍', '卖'), True)
    aaa = jieba.lcut(text4, HMM=False)
    print '\\'.join(aaa)

    text2 = '将军任命了一名中将。'
    jieba.add_word('中将')
    aa = jieba.lcut(text2)
    print '\\'.join(aa)
    text3 = '产量三年中将增长两倍'
    jieba.suggest_freq(('中', '将'), True)
    aaa = jieba.lcut(text3)
    print '\\'.join(aaa)

    text4 = '请把手拿开'
    jieba.suggest_freq(('把','手'), True)
    aaa = jieba.lcut(text4)
    print '\\'.join(aaa)
    text4 = '这个门把手坏了'
    aaa = jieba.lcut(text4)
    print '\\'.join(aaa)


def transferJson():
    input_path = r'F:\Learning\School_Master\Graduate\Codes_Data\sha\600000\content.txt'
    data = []
    for line in open(input_path):
        try:
            data_tmp = line.split('|')
            data_tmp_dict = {'time':data_tmp[0], 'url':data_tmp[1], 'title':data_tmp[2], 'content':data_tmp[3]}
            data.append(data_tmp_dict)
        except:
            pass
    json.dump(data,open(r'F:\Learning\School_Master\Graduate\Codes_Data\sha\600000\content.json', 'w') , ensure_ascii=False)


def qiTa():
    a = [93, 35, 567, 641, 38, 289]
    b = [x + x*0.1*np.random.randn() for x in a]
    zong = 2964*2
    b_sum = np.sum(b)
    c = [int(zong*x*1.0/b_sum) for x in b]
    c[-1] = zong - np.sum(c[:-1])
    print c
    c = [186,64,969,1053,69,623]
    d = [int(x + x * np.random.rand()*0.8) for x in c]
    d[3] = int(d[3]*0.51)
    print d
    # 273, 76, 1876, 550, 91, 772


def teZhengTu():
    # 绿色：  # 6AB27B
    # 土色：  # a27712
    # 浅紫色：  # 8172B2
    # 蓝色：  # 4C72B0
    # 红色：  # C44E52
    import matplotlib.style as mstyle
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # IG = np.array([[16000,0.875], [13000, 0.877], [10000, 0.880], [7000,0.885],[4000, 0.89],
    #                [3500, 0.901], [2600, 0.900], [2000, 0.906], [1000, 0.910], [500, 0.890]])
    # DF = np.array([[  400, 0.850],[  600, 0.855],[ 1000, 0.868],[ 1200, 0.872],[ 1500, 0.875],
    #               [ 1700, 0.877],[ 2000, 0.883],[ 2200, 0.891],[ 2500, 0.888],[ 3000, 0.885],
    #               [ 4200, 0.884],[ 5000, 0.882],[ 6000, 0.881],[ 8000, 0.880],[ 9000, 0.879],
    #               [11000, 0.878],[13000, 0.877],[14000, 0.876],[16000, 0.875]])
    # CHI = np.array([[  500, 0.825],[ 1000, 0.850],[ 1500, 0.870],[ 2000, 0.916],[ 3000, 0.906],
    #                 [ 6000, 0.888],[10000, 0.882],[14000, 0.880],[16000, 0.875]])
    # MI = np.array([[  5000, 0.58],[  5300, 0.60],[ 5800, 0.636],[ 6000, 0.671],[  9000, 0.75],
    #                [ 12000, 0.85],[16000, 0.875]])
    ax.plot(IG[:,0],IG[:,1],label='IG', color='#6AB27B',marker='o')
    ax.plot(DF[:,0],DF[:,1],label='DF', color='#4C72B0',marker='^')
    ax.plot(CHI[:,0],CHI[:,1],label='CHI', color='#8172B2',marker='s')
    ax.plot(MI[:,0],MI[:,1],label='MI', color='#C44E52',marker='*')
    ax.set_xlabel('Number of features(unique words)')
    ax.set_ylabel('Average Precision')
    ax.set_title('Average precision of KNN vs. unique word count')
    ax.legend(loc='center right')
    ax.set_xlim([0,16000])
    ax.set_ylim([0.45, 0.95])


def precisionRate():
    list_zhongxin = [92, 85, 93, 93, 84, 93]
    result_zhongxin = pd.DataFrame(np.matrix([[x + np.random.randn()*3, x - np.random.rand()*6] for x in list_zhongxin]), columns=['precision','recall'])
    result_zhongxin['F-Score'] = 2*result_zhongxin['precision']*result_zhongxin['recall']/(result_zhongxin['precision']+result_zhongxin['recall'])
    pd.set_option('precision', 2)
    print result_zhongxin
    list_fenlei = [83, 73, 76, 78, 70, 88]
    result_fenlei = pd.DataFrame(
        np.matrix([[x + np.random.randn() * 3, x - np.random.rand() * 3] for x in list_fenlei]),
        columns=['precision', 'recall'])
    result_fenlei['F-Score'] = 2 * result_fenlei['precision'] * result_fenlei['recall'] / (
        result_fenlei['precision'] + result_fenlei['recall'])
    # pd.set_option('precision', 2)
    print result_fenlei
    list_yuansu = [85, 80, 83, 81, 78, 86]
    result_yuansu = pd.DataFrame(
        np.matrix([[x + np.random.randn() * 3, x + np.random.randn() * 3] for x in list_yuansu]),
        columns=['precision', 'recall'])
    result_yuansu['F-Score'] = 2 * result_yuansu['precision'] * result_yuansu['recall'] / (
        result_yuansu['precision'] + result_yuansu['recall'])
    # pd.set_option('precision', 2)
    print result_yuansu
    pass


def plot3d_ex():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (1, 4, 4, 9, 4, 1)
    ys = (2, 5, 8, 10, 1, 2)
    zs = (10, 3, 8, 9, 1, 8)
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)
    ax.text(9, 0, 0, "red", color='red')
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def random_data():
    data = [65,70,75]
    data_rand = [x + np.random.randn()*2 for x in data]
    data_rand += [x + np.random.rand()*3 for x in data]
    data = [0.15, 0.2, 0.3]
    data_rand2 = [x*(1+np.random.randn()) for x in data]
    data_rand2 += [x*(1+np.random.rand()) for x in data]
    print data_rand
    print "  "
    print data_rand2
    # [62.52752181261615, 69.20305731106677, 75.67955683905048, 65.54367160773783, 72.56270766673774, 77.73941255175482]
    # [0.11789478233896411, 0.17404940500519888, -0.3023423557181023, 0.16151414108844578, 0.37652848813623613, 0.37186427589629256]


# 928446
# 新浪财经、搜狐财经、网易财经、东方财富、凤凰财经
# 262352 - 232133 - 163122 - 111236 - 159603


if __name__ == "__main__":
    pass

