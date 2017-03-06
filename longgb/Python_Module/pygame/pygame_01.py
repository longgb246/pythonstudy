# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys


def project_01_2pictures():
    '''
    这个就是出现一个图片的窗口，改变鼠标的形状
    '''
    # 设置文件
    background_image_filename = r'F:\forread\python\pygame\01.png'
    mouse_image_filename = r'F:\forread\python\pygame\02.png'
    # 初始化pygame，为使用硬件做准备
    pygame.init()
    # 创建一个窗口
    # [ 解析 ] 返回一个Surface对象，
    # 参数的第一个为分辨率，第二个是一个标志位，具体意思见下表，如果不用什么特性，就指定0；第三个为色深。
    screen = pygame.display.set_mode((640, 480), 0, 32)
    # 设置窗口的标题
    pygame.display.set_caption("Hello world!")
    background = pygame.image.load(background_image_filename).convert()
    mouse_cursor = pygame.image.load(mouse_image_filename).convert_alpha()
    # 游戏主循环
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                # 接收到退出事件后退出程序
                sys.exit()
        # 将背景图画上去
        screen.blit(background, (0, 0))
        # 获得鼠标位置
        x, y = pygame.mouse.get_pos()
        # 计算光标的左上角位置
        x -= mouse_cursor.get_width() / 2
        y -= mouse_cursor.get_height() / 2
        # 把光标画上去
        screen.blit(mouse_cursor, (x, y))
        # 刷新一下画面
        pygame.display.update()


def project_02_test():
    pygame.init()
    screen_size = (640, 480)
    screen = pygame.display.set_mode(screen_size, 0, 32)
    font = pygame.font.SysFont("arial", 16)
    font_height = font.get_linesize()
    event_text = []
    while True:
        event = pygame.event.wait()
        event_text.append(str(event))
        event_text = event_text[-screen_size[1]/font_height:]
        if event.type == QUIT:
            sys.exit()
        screen.fill((0, 255, 0))
        y = screen_size[1] - font_height
        for text in reversed(event_text):
            screen.blit(font.render(text, True, (0, 0, 0)), (0, y))
            y -= font_height
        pygame.display.update()


if __name__ == '__main__':
    # project_01 : 就是出现一个图片的窗口，改变鼠标的形状
    # project_01_2pictures()
    project_02_test()


import os
import pandas as pd
import numpy as np
import time


def printruntime(t1, name):
    '''
    性能测试，运行时间
    '''
    d = time.time() - t1
    min_d = np.floor(d / 60)
    sec_d = d % 60
    hor_d = np.floor(min_d / 60)
    if hor_d >0:
        print 'Run Time ({3}) is : {2} hours {0} min {1:.4f} s'.format(min_d, sec_d, hor_d, name)
    else:
        print 'Run Time ({2}) is : {0} min {1:.4f} s'.format(min_d, sec_d, name)


path = r'F:\forread'
read_path = path + os.sep + 'allocation_sim_actual_select_sale_inv_final.csv'
data = pd.read_csv(read_path)
data.columns
t1 = time.time()
d = data.groupby('parent_sale_ord_id')
result = []
for parent_sale_ord_id, group in d:
    tmp_group = group.copy()
    tmp_group_non = tmp_group[tmp_group['sale_qtty_y'].notnull()]
    n_sum = len(tmp_group_non)
    n_actual = len(tmp_group['sku'].drop_duplicates())
    n_sim = len(tmp_group_non['sku'].drop_duplicates())
    if n_sum >= 1:
        tmp_group['sale_counts'] = n_sum
        tmp_group['sku_counts_actual'] = n_actual
        tmp_group['sku_counts_sim'] = n_sim
        result.append(tmp_group)
result_pd = pd.concat(result)
printruntime(t1, 'count')

result_pd.to_csv(path + os.sep + 'allocation_sale_inv_qtty_not_all_null2.csv', index=False)



