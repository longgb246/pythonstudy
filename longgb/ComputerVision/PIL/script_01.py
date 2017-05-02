#-*- coding:utf-8 -*-
import os
from PIL import Image



path = r'D:\Life\life-photo\于函总'
path_cn = unicode(path , "utf8")


this_pic = path_cn + os.sep + os.listdir(path_cn)[0]

im = Image.open(this_pic)
print im.format, im.size, im.mode

im.show()
im.rotate(45).show()
im_resize = im.resize()
im_resize.save(path_cn + os.sep + '')

