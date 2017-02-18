#-*- coding:utf-8 -*-
# 绿色：#6AB27B
# 土色：#a27712
# 浅紫色：#8172B2
# 蓝色：#4C72B0
# 红色：#C44E52
from mpl_toolkits.mplot3d import Axes3D             # 3d 库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import warnings
warnings.filterwarnings('ignore')


def plot_2dcollections3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')           # 设置 3d 的画质, 产生一个3d的坐标
    # =========================== 1、画出 z 轴 sin 图 ===========================
    x = np.linspace(0, 1, 100)
    y = np.sin(x * 2 * np.pi) / 2 + 0.5
    ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')      # zs=0 是 z 的坐标，zdir='z' 表示在 z 轴
    # =========================== 2、画出 y 轴 散点图 ===========================
    colors = ['r', 'g', 'b', 'k']
    x = np.random.sample(20*len(colors))    # sample 取出均匀分布[0,1]之间的样本
    y = np.random.sample(20*len(colors))
    c_list = []
    for c in colors:
        c_list.append([c]*20)
    ax.scatter(x, y, zs=0, zdir='y', c=c_list ,label='points in (x, z)')
    # =========================== 3、调整图片 ====================================
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-35)
    plt.show()


def plot_bars3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)             # [0, 1] 均匀分布
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-35)         # elev是高度角度，azim是平面角度


def plot_contour3d():
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    # 样式1
    cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
    # 样式2
    cset = ax.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)


def plot_contour3d2():
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)                   # 画表面
    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)         # 画侧面？奇怪
    cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)


def plot_contourf3d():
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    # =============================== 样式1 ================================
    cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
    # =============================== 样式2 ================================
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)


def plot_custom_shaded_3d_surface():
    # 还没有搞清楚这个代码的意思。
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    filename = cbook.get_sample_data('jacksboro_fault_dem.npz', asfileobj=False)
    with np.load(filename) as dem:
        z = dem['elevation']
        nrows, ncols = z.shape
        x = np.linspace(dem['xmin'], dem['xmax'], ncols)
        y = np.linspace(dem['ymin'], dem['ymax'], nrows)
        x, y = np.meshgrid(x, y)
    region = np.s_[5:50, 5:50]
    x, y, z = x[region], y[region], z[region]
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)


def plot_hist3d():
    # 效果很不好，不算是想要的
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.random.rand(2, 100)*4                 # 2*100 的随机数
    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0,4], [0,4]])           # 生成2*2的平面的坐标的高低
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')                        # 竖着拉伸成一个数组
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)                      # 生成全部是 0 的np数组，规格和xpos一样
    dx = 0.5 * np.ones_like(zpos)                   # 生成全部是 1 的np数组，规格和zpos一样
    dy = dx.copy()
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#6AB27B', zsort='average')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_lines3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()          # 默认：mpl.rcParams['figure.figsize']=[8.0, 6.0]
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='parametric curve', color='#4C72B0')
    ax.legend()


def plot_lorenz_attractor():
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot
    dt = 0.01
    stepCnt = 10000
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))
    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, lw=0.5, color='#4C72B0')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")


def plot_mixed_subplots():
    from mpl_toolkits.mplot3d import Axes3D
    def f(t):
        s1 = np.cos(2*np.pi*t)
        e1 = np.exp(-t)
        return np.multiply(s1, e1)
    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)
    t3 = np.arange(0, 2, 0.01)
    fig = plt.figure(figsize=plt.figaspect(2.0))
    fig.suptitle('A tale of 2 subplots')
    ax = fig.add_subplot(2, 1, 1)
    l = ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)
    ax.set_ylabel('Damped oscillation')
    ax.set_title('Damped oscillation')
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    surf = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, color='#4C72B0')
    ax2.set_zlim3d(-1, 1)
    ax2.set_title('A picture')


def plot_quiver3d():
    # 看不清楚
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.2), np.arange(-0.8, 1, 0.8))
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z))
    ax.quiver(x, y, z, u, v, w, length=0.1)


# 旋转动图
def plot_rotate_axes3d():
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure(figsize=(8,18))
    ax = fig.add_subplot(211, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.1)
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)                # 轮廓图
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_surface(X, Y, Z, rstride=5, cstride=5, linewidth=0, color='#4C72B0')
    for angle in range(0, 360):
        ax.view_init(30, angle)
        ax2.view_init(30, angle)
        plt.draw()
        plt.pause(0.05)


def plot_scatter3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    def randrange(n, vmin, vmax):
        return (vmax - vmin) * np.random.rand(n) + vmin
    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    for angle in range(360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.001)


def plot_subplot3d():
    from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
    from matplotlib import cm
    fig = plt.figure(figsize=(15,7))
    fig.suptitle("Two Pictures")
    ax = fig.add_subplot(121, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.set_zlim(-1.01, 1.01)
    ax.set_title("First..")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y, Z = get_test_data(0.05)
    ax2.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    ax2.set_title("Second..")
    for angle in range(100):
        ax.view_init(30, angle)
        ax2.view_init(30, angle)
        plt.draw()
        plt.pause(0.001)


# 调整显示格式
def plot_surface3d():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.colorbar(surf, shrink=0.5, aspect=10)


def plot_ball():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='#4C72B0', alpha=0.7)


# 特殊颜色
def plot_surface3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = np.linspace(0, 1.25, 50)
    p = np.linspace(0, 2 * np.pi, 50)
    R, P = np.meshgrid(r, p)
    Z = ((R**2 - 1)**2)
    X, Y = R * np.cos(P), R * np.sin(P)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r, rstride=1, cstride=1, linewidth=0)
    ax.set_zlim(0, 1)
    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')


# 添加文字
def plot_text3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))         # (1, 1, 0)表示方位
    xs = (1, 4, 4, 9, 4, 1)
    ys = (2, 5, 8, 10, 1, 2)
    zs = (10, 3, 8, 9, 1, 8)
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)
        ax.plot([x], [y], [z], color='#4C72B0', marker='o')
    ax.text(9, 0, 0, "red", color='red')
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


# np 独特用法 np.repeat(angles[..., np.newaxis], n_radii, axis=1) 拆成竖着的  .flatten()反向操作
def plot_trisurf():
    from mpl_toolkits.mplot3d import Axes3D
    n_radii = 8
    n_angles = 36
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)           # 重复
    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x * y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, color='#4C72B0')


def plot_trisurf2():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.tri as mtri
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # ============
    # First plot
    # ============
    u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
    v = np.linspace(-0.5, 0.5, endpoint=True, num=10)
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()
    x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
    y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
    z = 0.5 * v * np.sin(u / 2.0)
    tri = mtri.Triangulation(u, v)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral, linewidth=0)
    ax.set_zlim(-1, 1)
    # ============
    # Second plot
    # ============
    n_angles = 36
    n_radii = 8
    min_radius = 0.25
    radii = np.linspace(min_radius, 0.95, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    z = (np.cos(radii) * np.cos(angles * 3.0)).flatten()
    triang = mtri.Triangulation(x, y)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid ** 2 + ymid ** 2 < min_radius ** 2, 1, 0)
    triang.set_mask(mask)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_trisurf(triang, z, cmap=plt.cm.CMRmap, linewidth=0)


# 海浪动画
def plot_wire3d_animation():
    from mpl_toolkits.mplot3d import axes3d
    import time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = np.linspace(-1, 1, 50)
    ys = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(xs, ys)
    ax.set_zlim(-1, 1)
    wframe = None
    tstart = time.time()
    def generate(X, Y, phi):
        R = 1 - np.sqrt(X**2 + Y**2)
        return np.cos(2 * np.pi * X + phi) * R
    for phi in np.linspace(0, 180. / np.pi, 100):
        if wframe:
            ax.collections.remove(wframe)
        Z = generate(X, Y, phi)
        wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='#4C72B0')
        plt.pause(0.05)
    print('Average FPS: %f' % (100 / (time.time() - tstart)))


def plot_wire3d_zero_stride():
    from mpl_toolkits.mplot3d import axes3d
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})
    X, Y, Z = axes3d.get_test_data(0.05)
    ax1.plot_wireframe(X, Y, Z, rstride=10, cstride=0)
    ax1.set_title("Column (x) stride set to 0")
    ax2.plot_wireframe(X, Y, Z, rstride=0, cstride=10)
    ax2.set_title("Row (y) stride set to 0")


# ================================= 自己函数 =================================
def self_bars3d_demo(n):
    data_demo = []
    for x in xrange(n):
        ys = np.random.rand(10) * 10  # [0, 1] 均匀分布
        data_demo.append(ys)
    data_demo = pd.DataFrame(data_demo).T
    data_demo.columns = [str(1998 + x) for x in range(n)]
    self_bars3d(data_demo)


# 画多年的 bar 图
def self_bars3d(data):
    '''
    data: 为数据框，传入数据仅仅包含以年份，如'1998'等包含的列
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data_len = len(data.columns)
    c_color = ['r', 'g', 'b', 'y', 'c'] * 10
    c_color = c_color[:data_len]
    i = 0
    for c, z in zip(c_color, data.columns):
        i += 1
        xs = np.arange(len(data[z]))
        ys = data[z]
        cs = [c] * len(xs)
        ax.bar(xs, ys, zs=i, zdir='y', color=cs, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel(u'Y（年份）')
    ax.set_ylim3d(1, data_len)
    if data_len == 1:
        data_columns = [""] * 2 + list(data.columns) + [""] * 2
    elif data_len == 2:
        print list(data.columns)
        print list(data.columns)[0]
        data_columns = [list(data.columns)[0]] + [""] * 4 + [list(data.columns)[1]]
    elif data_len <= 5:
        data_columns = []
        for x in data.columns:
            data_columns.append(x)
            if x != data.columns[-1]:
                data_columns.append('')
    else:
        data_columns = list(data.columns)
    ax.set_yticklabels(data_columns)
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-35)  # elev是高度角度，azim是平面角度


if __name__ == '__main__':
    # plot_2dcollections3d()
    # plot_bars3d()
    # plot_contour3d()
    # plot_contour3d2()
    # plot_contourf3d()
    # for i in xrange(10):
    #     self_bars3d_demo(i)
    # plot_lorenz_attractor()
    plot_subplot3d()
    pass
