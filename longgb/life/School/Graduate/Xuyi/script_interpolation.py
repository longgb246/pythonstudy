# -*- coding:utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# Note that the output interpolated coords will be the same dtype as your input
# data. If we have an array of ints, and we want floating point precision in
# the output interpolated points, we need to cast the array as floats
data = np.arange(40).reshape((8,5)).astype(np.float)
# I'm writing these as row, column pairs for clarity...
coords = np.array([[1.2, 3.5], [6.7, 2.5], [7.9, 3.5], [3.5, 3.5]])
# However, map_coordinates expects the transpose of this
coords = coords.T
# The "mode" kwarg here just controls how the boundaries are treated
# mode='nearest' is _not_ nearest neighbor interpolation, it just uses the
# value of the nearest cell if the point lies outside the grid. The default is
# to treat the values outside the grid as zero, which can cause some edge
# effects if you're interpolating points near the edge
# The "order" kwarg controls the order of the splines used. The default is
# cubic splines, order=3
zi = ndimage.map_coordinates(data, coords, order=3, mode='nearest')
row, column = coords
nrows, ncols = data.shape
im = plt.imshow(data, interpolation='nearest', extent=[0, ncols, nrows, 0])
plt.colorbar(im)
plt.scatter(column, row, c=zi, vmin=data.min(), vmax=data.max())
for r, c, z in zip(row, column, zi):
    plt.annotate('%0.3f' % z, (c,r), xytext=(-10,10), textcoords='offset points', arrowprops=dict(arrowstyle='->'), ha='right')
plt.show()

