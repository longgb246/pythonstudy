#-*- coding:utf-8 -*-
import os
from string import Template
import time

# sh allocation_sku_data.sh  2016-07-01 2016-11-01 630/628/658 316

def printruntime(t1):
    d = time.time() - t1
    min_d = int(d / 60)
    sec_d = d % 60
    print 'Run Time is : {0} min {1:.4f} s'.format(min_d, sec_d)

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))

hive_skuinv = '''set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.max.dynamic.partitions=20000;
set hive.exec.max.dynamic.partitions.pernode=20000;
CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data_fdcinv${test}
	(
		sku_id	 		string,
		open_po_fdc  	int,
		inv_fdc			int)
		PARTITIONED by (date_s  string,dc_id int);
	insert OVERWRITE table dev.dev_allocation_sku_data_fdcinv${test}   partition(date_s,dc_id)
	SELECT
		sku_id,
		sum(in_transit_qtty) AS open_po_fdc,
		sum(stock_qtty) AS inv_fdc, --库存数量
		dt  as date_s,
		delv_center_num
	FROM
		gdm.gdm_m08_item_stock_day_sum	     	-- 【主表】商品库存日汇总
	WHERE
		dt>='${start_date}' AND
		dt<='${end_date}' AND
		delv_center_num='${dc_id}'
	group by
		delv_center_num,
		dt,
		sku_id
'''
hive_skuinv = Template(hive_skuinv)


# 按照 2 个分区，时间太长了
hive_skuinv_2 = '''set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.max.dynamic.partitions=20000;
set hive.exec.max.dynamic.partitions.pernode=20000;
CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data_fdcinv${test}
	(
		sku_id	 		string,
		open_po_fdc  	int,
		inv_fdc			int,
		dc_id int)
		PARTITIONED by (date_s  string);
	insert OVERWRITE table dev.dev_allocation_sku_data_fdcinv${test}   partition(date_s)
	SELECT
		sku_id,
		sum(in_transit_qtty) AS open_po_fdc,
		sum(stock_qtty) AS inv_fdc, --库存数量
		delv_center_num,
		dt  as date_s
	FROM
		gdm.gdm_m08_item_stock_day_sum	     	-- 【主表】商品库存日汇总
	WHERE
		dt>='${start_date}' AND
		dt<='${end_date}' AND
		delv_center_num='${dc_id}'
	group by
		delv_center_num,
		dt,
		sku_id
'''
hive_skuinv_2 = Template(hive_skuinv_2)


hive_drop = '''DROP TABLE IF EXISTS dev.dev_allocation_sku_data${test};
'''
hive_drop = Template(hive_drop)

# 2016-07-01 2016-11-01 630 316
start_date = '2016-07-01'
end_date = '2016-11-01'
dc_id_list = ['630','628','658']
# dc_id_list = ['630','658']
drop_table = 0      # 为 0 的时候，表示不删除表， 为 1 ，删除表
# istest = '_test'    # 为 ' ' 的时候，表示正式插入表，有任何字符表示创建 test 的表
istest = ' '    # 为 ' ' 的时候，表示正式插入表，有任何字符表示创建 test 的表
org_dc_id = '316'


if __name__ == '__main__':
    if drop_table == 1:
        print "Drop Table ...."
        os.system('hive -e "{0}";'.format(hive_skuinv.substitute(test=istest)))
    for each in dc_id_list:
        print "{0} ...".format(each)
        t1 = time.time()
        pyhive(hive_skuinv.substitute(start_date=start_date, end_date=end_date, dc_id=each, test=istest), 'sku_fdcinv.log')
        printruntime(t1)


