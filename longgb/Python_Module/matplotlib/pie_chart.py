#-*- coding:utf-8 -*-
# 绿色：#6AB27B
# 土色：#a27712
# 浅紫色：#8172B2
# 蓝色：#4C72B0
# 红色：#C44E52


def plot_mix_method():
    '''
    这种方法是混合方法，使用外部的SVG。
    '''
    import matplotlib
    matplotlib.use("Svg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Shadow
    fig1 = plt.figure(1, figsize=(6, 6))
    ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
    fracs = [15, 30, 45, 10]            # 百分比
    explode = (0, 0.05, 0, 0)
    pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%')
    for w in pies[0]:
        # set the id with the label.
        w.set_gid(w.get_label())
        # we don't want to draw the edge of the pie
        w.set_ec("none")
    for w in pies[0]:
        # create shadow patch
        s = Shadow(w, -0.01, -0.01)
        s.set_gid(w.get_gid() + "_shadow")
        s.set_zorder(w.get_zorder() - 0.1)
        ax.add_patch(s)
    # save
    from io import BytesIO
    f = BytesIO()
    plt.savefig(f, format="svg")
    import xml.etree.cElementTree as ET
    # filter definition for shadow using a gaussian blur
    # and lightening effect.
    # The lightening filter is copied from http://www.w3.org/TR/SVG/filters.html
    # I tested it with Inkscape and Firefox3. "Gaussian blur" is supported
    # in both, but the lightening effect only in the Inkscape. Also note
    # that, Inkscape's exporting also may not support it.
    filter_def = """
      <defs  xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'>
        <filter id='dropshadow' height='1.2' width='1.2'>
          <feGaussianBlur result='blur' stdDeviation='2'/>
        </filter>

        <filter id='MyFilter' filterUnits='objectBoundingBox' x='0' y='0' width='1' height='1'>
          <feGaussianBlur in='SourceAlpha' stdDeviation='4%' result='blur'/>
          <feOffset in='blur' dx='4%' dy='4%' result='offsetBlur'/>
          <feSpecularLighting in='blur' surfaceScale='5' specularConstant='.75'
               specularExponent='20' lighting-color='#bbbbbb' result='specOut'>
            <fePointLight x='-5000%' y='-10000%' z='20000%'/>
          </feSpecularLighting>
          <feComposite in='specOut' in2='SourceAlpha' operator='in' result='specOut'/>
          <feComposite in='SourceGraphic' in2='specOut' operator='arithmetic'
        k1='0' k2='1' k3='1' k4='0'/>
        </filter>
      </defs>
    """
    tree, xmlid = ET.XMLID(f.getvalue())
    # insert the filter definition in the svg dom tree.
    tree.insert(0, ET.XML(filter_def))
    for i, pie_name in enumerate(labels):
        pie = xmlid[pie_name]
        pie.set("filter", 'url(#MyFilter)')
        shadow = xmlid[pie_name + "_shadow"]
        shadow.set("filter", 'url(#dropshadow)')
    fn = "svg_filter_pie.svg"
    print("Saving '%s'" % fn)
    ET.ElementTree(tree).write(fn)


def plot_axpie():
    '''
    使用ax.pie画图
    '''
    import matplotlib.pyplot as plt
    from pylab import mpl
    # import Image
    mpl.rcParams['font.sans-serif'] = ['SimHei']          # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False              # 解决保存图像是负号'-'显示为方块的问题
    # import seaborn as sns                                   # seaborn不起作用。
    labels = [u'清华大学', u'上海财经大学', u'武汉大学', u'中央财经大学']
    sizes = [15, 30, 45, 10]
    explode = (0, 0, 0, 0.1)                                #  向外突出的距离
    colors = ['#C44E52', '#4C72B0', '#8172B2', '#6AB27B']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
    # startangle：第一个（15）开始的角度，shadow：阴影，autopct：自动百分比显示
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.savefig(r'C:\Users\longgb246\Desktop\test.eps')
    # fig1.savefig(r'C:\Users\longgb246\Desktop\test.jpeg')
    # fig1.savefig(r'F:\test.png')
    # Image.open(r'F:\test.png').save(r'F:\test.jpg','JPEG')


if __name__ == '__main__':
    # plot_mix_method()
    plot_axpie()
    # import numpy as np
    # N = 20
    # np.linspace(0.0, 2 * np.pi, N, endpoint=True)
