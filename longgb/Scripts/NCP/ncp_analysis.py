# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/02/09
"""
Usage Of 'ncp_analysis.py' :
"""

# 腾讯疫情实时追踪 : https://news.qq.com/zt2020/page/feiyan.htm?from=timeline&isappinstalled=0


import datetime
import json
import requests

import numpy as np
import pandas as pd

from pyecharts.globals import ChartType
from pyecharts.charts import Geo
from pyecharts import options as opts


def catch_data():
    """使用腾讯疫情实时追踪，抓取新冠病毒每日数据。

    Returns:
        dataframe, data: 疫情的详细数据, 抓取的data数据
    """
    url = 'https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5'
    data = json.loads(requests.get(url=url).json()['data'])
    distb = data['areaTree'][0]
    # growth = data['chinaDayList']
    provinces = distb['children']

    def get_detail_info(x):
        return [
            x['name'],
            x['total']['confirm'],  # 确诊人数
            x['total']['suspect'],  # 疑似人数
            x['total']['dead'],  # 死亡人数
            x['total']['heal'],  # 治愈人数
            x['today']['confirm'],  # 今日新增
            x['today']['suspect'],
            x['today']['dead'],
            x['today']['heal'],
        ]

    all_info = []
    for province_info in provinces:
        province = province_info['name']
        this_info = []
        this_info.append(
            get_detail_info(province_info) + [province, 'province']
        )
        for city_info in province_info['children']:
            this_info.append(get_detail_info(city_info) + [province, 'city'])
        all_info.extend(this_info)

    columns = [
        'name', 'confirm_all', 'suspect_all', 'dead_all', 'heal_all',
        'confirm_today', 'suspect_today', 'dead_today', 'heal_today',
        'province', 'type'
    ]
    all_df = pd.DataFrame(all_info, columns=columns)

    return all_df, data


def save_daily_data(all_df):
    """保存每日的数据

    Args:
        all_df (csv): 保存csv文件
    """
    all_df.to_csv(
        'E:/WorkSpace/NCP/data/area_daily_{}'.format(
            datetime.date.today().strftime('%Y%m%d')),
        index=False
    )


def plot_geo_heatmap(all_df):
    """画热力图

    Args:
        all_df (dataframe): 每日数据图
    """
    def valid_area(x, v):
        if x.get_coordinate(v) is not None:
            return 'Y'
        else:
            return 'N'

    c = Geo(
        init_opts=opts.InitOpts(width='800px', height='600px')
    )

    all_df['valid'] = all_df['name'].apply(lambda x: valid_area(c, x))
    all_df['plot_v'] = np.log2(all_df['confirm_all'] / 50 + 1) * 10

    c.add_schema(maptype="china")
    c.add(
        "ratio",
        all_df.loc[all_df['valid'] == 'Y', ['name', 'plot_v']].values,
        type_=ChartType.HEATMAP,
    )

    c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    c.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(),
        legend_opts=opts.LegendOpts(is_show=False),
        title_opts=opts.TitleOpts(
            pos_left='center',
            title='2020 Novel coronavirus pneumonia (NCP) ',
            title_link='https://news.qq.com/zt2020/page/feiyan.htm?from=timeline&isappinstalled=0'
        )
    )

    # map_p = Map()
    # map_p.set_global_opts(title_opts=opts.TitleOpts(title="实时疫情图"), visualmap_opts=opts.VisualMapOpts(max_=100))
    # map_p.add("确诊", data, maptype="china")
    Geo.render(c, path='E:/HEATMAP.html')


def load_latest_data():
    data_path = 'f:/Codes/DXY-2019-nCoV-Data/csv/DXYArea.csv'
    data_df = pd.read_csv(data_path)
    data_df['updateTime2'] = pd.to_datetime(data_df['updateTime'])
    data_df.groupby(['provinceName', 'cityName'])
    #
    pass


def main():
    all_df, _ = catch_data()
    save_daily_data(all_df)
    plot_geo_heatmap(all_df)


if __name__ == "__main__":
    main()
