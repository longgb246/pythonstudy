#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 1、画多边形，有点难。形状。
# np.mgrid  reshape(2, -1)  plt.subplots_adjust  plt.axis('equal')  plt.axis('off')
def artist_reference():
    import matplotlib.path as mpath
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    def label(xy, text):
        y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
        plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)     # ha = horizontalalignment
    fig, ax = plt.subplots()
    grid = np.mgrid[0.2:0.8:0.3, 0.2:0.8:0.3].reshape(2, -1).T
    patches = []
    # add a circle
    circle = mpatches.Circle(grid[0], 0.1, ec="none")
    patches.append(circle)
    label(grid[0], "Circle")
    # add a rectangle
    rect = mpatches.Rectangle(grid[1] - [0.025, 0.05], 0.05, 0.1, ec="none")
    patches.append(rect)
    label(grid[1], "Rectangle")
    # add a wedge
    wedge = mpatches.Wedge(grid[2], 0.1, 30, 270, ec="none")
    patches.append(wedge)
    label(grid[2], "Wedge")
    # add a Polygon
    polygon = mpatches.RegularPolygon(grid[3], 5, 0.1)
    patches.append(polygon)
    label(grid[3], "Polygon")
    # add an ellipse
    ellipse = mpatches.Ellipse(grid[4], 0.2, 0.1)
    patches.append(ellipse)
    label(grid[4], "Ellipse")
    # add an arrow
    arrow = mpatches.Arrow(grid[5, 0] - 0.05, grid[5, 1] - 0.05, 0.1, 0.1, width=0.1)
    patches.append(arrow)
    label(grid[5], "Arrow")
    # add a path patch
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, [0.018, -0.11]),
        (Path.CURVE4, [-0.031, -0.051]),
        (Path.CURVE4, [-0.115, 0.073]),
        (Path.CURVE4, [-0.03, 0.073]),
        (Path.LINETO, [-0.011, 0.039]),
        (Path.CURVE4, [0.043, 0.121]),
        (Path.CURVE4, [0.075, -0.005]),
        (Path.CURVE4, [0.035, -0.027]),
        (Path.CLOSEPOLY, [0.018, -0.11])
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts + grid[6], codes)
    patch = mpatches.PathPatch(path)
    patches.append(patch)
    label(grid[6], "PathPatch")
    # add a fancy box
    fancybox = mpatches.FancyBboxPatch(
        grid[7] - [0.025, 0.05], 0.05, 0.1,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02))
    patches.append(fancybox)
    label(grid[7], "FancyBboxPatch")
    # add a line
    x, y = np.array([[-0.06, 0.0, 0.1], [0.05, -0.05, 0.05]])
    line = mlines.Line2D(x + grid[8, 0], y + grid[8, 1], lw=5., alpha=0.3)
    label(grid[8], "Line2D")
    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    ax.add_line(line)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    pass
# 2、这样有点难
def path_patch_demo():
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots()
    Path = mpath.Path
    path_data = [                           # 1、设置点
        (Path.MOVETO, (1.58, -2.57)),       # 移动至(1.58, -2.57)
        (Path.CURVE4, (0.35, -1.1)),        # 应该是样条曲线，以下为3点连续
        (Path.CURVE4, (-1.75, 2.0)),
        (Path.CURVE4, (0.375, 2.0)),
        (Path.LINETO, (0.85, 1.15)),        # 直线
        (Path.CURVE4, (2.2, 3.2)),
        (Path.CURVE4, (3, 0.05)),
        (Path.CURVE4, (2.0, -0.5)),
        (Path.CLOSEPOLY, (1.58, -2.57)),    # 封闭？
    ]
    codes, verts = zip(*path_data)          # 拆分形式
    path = mpath.Path(verts, codes)                                 # 2.1、形成路径
    patch = mpatches.PathPatch(path, facecolor='r', alpha=0.5)      # 2.2、形成路径
    ax.add_patch(patch)                     # 3、添加补丁，画图
    x, y = zip(*path.vertices)              # 取出原始点
    line, = ax.plot(x, y, 'go-')            # 画原始点路径
    ax.grid()
    ax.axis('equal')
    plt.show()
    pass
# 3、散点图
def scatter_demo():
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    area = np.pi * (15 * np.random.rand(N)) ** 2            # 0 to 15 point radii
    # area = (15 * np.random.rand(N)) ** 2            # 0 to 15 point radii
    scat = ax.scatter(x, y, s=area
                , c=colors
                , alpha=0.5
                )
    plt.show()
    pass


if __name__ == '__main__':
    pass
