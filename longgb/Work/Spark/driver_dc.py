#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'penghongxiao'

import os
import sys
import datetime as dt
import traceback

from pyspark import SparkContext, SparkConf

dc_store={'3':'上海','4':'成都','5':'武汉','6':'北京','7':'南京','8':'济南','9':'沈阳','10':'广州','316':'西安','322':'福州',
'545':'杭州','601':'天津','603':'深圳','605':'重庆','606':'苏州','607':'宁波','608':'郑州','609':'厦门','610':'青岛','614':'石家庄',
'615':'太原','616':'南宁','617':'哈尔滨','618':'大连','619':'长沙','628':'兰州','630':'乌鲁木齐','631':'合肥','632':'宿迁','633':'昆明',
'634':'贵阳','635':'南昌','636':'长春','644':'佛山','649':'襄阳','652':'金华','653':'汕头','654':'海口','655':'南充','658':'银川',
'678':'驻马店','679':'绵阳','680':'自贡','681':'潍坊','682':'固安','683':'唐山','684':'南通','696':'呼和浩特','701':'南海','711':'温州',
'731':'东莞','733':'浦东','734':'柳州','762':'济宁','767':'洛阳','768':'衡阳'};

today=dt.date.today().strftime('%Y-%m-%d')
yesterday=(dt.date.today()-dt.timedelta(days=1)).strftime('%Y-%m-%d')

def list2sum(array):
	result_array=[]
	for i in array:
		result_array.append(float(i))		
	return sum(result_array)

def sales_ratio(line):
	brand_code=line[0][0]
	brand_name=line[0][1]
	item_third_cate_cd=line[0][2]
	item_third_cate_name=line[0][3]
	dcid=line[1][0][0]
	dc_sales=line[1][0][1]
	brand_dsales=line[1][1]
	dc_name = None
	if dcid in dc_store.keys():
		dc_name = dc_store.get(dcid).decode('utf-8');
		sales_per=float(dc_sales)/brand_dsales
		result='%s\t%s\t%s\t%s\t%s\t%s\t%f' % (item_third_cate_cd,item_third_cate_name,brand_code,brand_name,dcid,dc_name,sales_per)
		return result

def table_formal(line):
	brand_code=line[2]
	brand_name=line[3]
	item_third_cate_cd=line[0]
	item_third_cate_name=line[1]
	dcid=line[4]
	dc_name=line[5]
	sales_per=round(float(line[6]),6)
	result='%s\t%s\t%s\t%s\t%s\t%s\t%f' % (item_third_cate_cd,item_third_cate_name,brand_code,brand_name,dcid,dc_name,sales_per)
	return result
		

def func(array):
	tmp_sum = []
	for e in array[:-1]:
		tmp_sum.append(round(float(e[-1]),6))
	result=round(sum(tmp_sum),6)
	return result		

	
def func1(line):
	tmp_list=[]
	cate3=line[0][0]
	cate3_name=line[0][1]
	brand=line[0][2]
	brand_name=line[0][3]
	rev_dc=line[1][0][-1][0]
	rev_dc_name=line[1][0][-1][1]
	rev=round(1-line[1][1],6)
	for e in line[1][0][:-1]:
		tmp_list.append([cate3,cate3_name,brand,brand_name,e[0],e[1],e[2]])
	tmp_list.append([cate3,cate3_name,brand,brand_name,rev_dc,rev_dc_name,rev])
	return tmp_list		


def main():
	conf = SparkConf().setAppName("spark_ipc_sfs_app_standard_dc_salesper")
	sc = SparkContext(conf=conf)

	
	input_url1 = "/user/cmo_ipc/app.db/app_pf_standard_forecast_result/plan=bulky_dc/steplength=1/dt=2017-02-01"
	input_url2 = "hdfs://ns1/user/dd_edw/gdm.db/gdm_m03_self_item_sku_da/dt=2017-01-31"
	output_url = "/user/cmo_ipc/forecast/result/rdc/ratio/tmp_dc_ratio_phx0201"
	dc_forecast = sc.textFile(input_url1).map(lambda line:line.split('\t')).map(lambda line:(line[1].split('#')[0],line[1].split('#')[1],line[6]))	
	
	brand_file1=sc.textFile(input_url2).map(lambda line:line.split('\t'))
	brand_file2=brand_file1.filter(lambda line:len(line)>=70)
	brand_file=brand_file2.map(lambda line:(line[0],line[6],line[9],line[20],line[21]))
	
	########合并###############
	brand_file_kv = brand_file.map(lambda line:(line[0],line[1:]))
	dc_forecast_kv = dc_forecast.map(lambda line:(line[0],line[0:]))
	brand_dc=dc_forecast_kv.join(brand_file_kv)
	brand_dc_kv=brand_dc.map(lambda line:(line[1][1],line[1][0]))
	
	##########groupby分组############
	fsales28=brand_dc_kv.map(lambda line:(line[0],line[1][0],line[1][1],list2sum(line[1][2].split(','))))
	brand_sales=fsales28.map(lambda line:(line[0],line[3]))
	dc_sales=fsales28.map(lambda line:((line[0],line[2]),line[3]))
	
	brand_sales1=brand_sales.reduceByKey(lambda m,n: m + n)
	dc_sales1=dc_sales.reduceByKey(lambda m,n: m + n)
	
	dc_sales1_kv=dc_sales1.map(lambda line:(line[0][0],(line[0][1],line[1])))
	brand_dc=dc_sales1_kv.join(brand_sales1)
	brand_dc1=brand_dc.filter(lambda line:((line[1][1]!=0) and (line[1][0][0] in dc_store.keys())))
			
	os.system("hadoop fs -rm -r " + output_url)
	brand_dc2=brand_dc1.map(lambda line: sales_ratio(line))
	
	brand_dc3=brand_dc2.map(lambda x: ((x.split('\t')[0],x.split('\t')[1],x.split('\t')[2],x.split('\t')[3]),(x.split('\t')[4],x.split('\t')[5],x.split('\t')[6])))
	brand_dc4=brand_dc3.groupByKey()
	brand_dc5=brand_dc4.map(lambda x : (x[0], list(x[1])))
	brand_dc6=brand_dc5.map(lambda (k, v) : (k, [v,func(v)]))
	
	brand_dc7=brand_dc6.map(lambda line:func1(line))
	brand_dc8=brand_dc7.flatMap(lambda line:line[0:])
	brand_dc9=brand_dc8.map(lambda line:table_formal(line))
	
	brand_dc9.saveAsTextFile(output_url)
	sc.stop()


if __name__ == "__main__":
	main()
	r = os.system("hadoop fs -test -e /user/cmo_ipc/forecast/result/rdc/ratio/tmp_dc_ratio_phx0901/_SUCCESS")
	if r != 0:
		sc.stop()
		raise Exception("1")
	else:
		print ("执行spark获取大件卫星仓预测数据销量占比成功！！")



