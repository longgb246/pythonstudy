#-*- coding:utf-8 -*-
import os
from string import Template
import time

# sh allocation_sku_data.sh  2016-07-01 2016-11-01 630/628/658 316

def printruntime(t1):
    d = time.time() - t1
    min_d = int(d / 60)
    sec_d = d % 60
    print 'Run Time is : {0}min {1:.4f}s'.format(min_d, sec_d)

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*50, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))


hive_order = '''set hive.exec.dynamic.partition=true;
	      set hive.exec.dynamic.partition.mode=nonstrict;
        CREATE TABLE IF NOT EXISTS dev.dev_allocation_order_data${test}
		(
			arrive_time 	string,	-- 到达时间
			item_sku_id 	string,	-- skuid
			arrive_quantity	string	-- 实际到达量
		)
		PARTITIONED by (rdc_id string);
	INSERT overwrite table dev.dev_allocation_order_data${test}  partition(rdc_id)
		SELECT
			to_date(t2.complete_dt) as arrive_time,
			t2.sku_id as item_sku_id,
			sum(t2.actual_pur_qtty) as arrive_quantity,
			t2.int_org_num as rdc_id
	    FROM
	        gdm.gdm_m04_pur_det_basic_sum t2
	    WHERE
	        t2.dt = sysdate(-1) AND
	        t2.valid_flag = 1 AND  						-- 有效标志
	        t2.cgdetail_yn = 1 AND 						-- 采购明细单有效标志
	        to_date(t2.create_tm) BETWEEN '${start_date}' AND '${end_date}' AND
	        t2.int_org_num = '${dc_id}'
	    group by
		    t2.int_org_num,
		    to_date(t2.complete_dt),
		    t2.sku_id;
'''
hive_order = Template(hive_order)

hive_select = '''select * from dev.dev_allocation_order_data${test} limit 10;
'''
hive_select = Template(hive_select)

hive_drop = '''DROP TABLE IF EXISTS dev.dev_allocation_order_data${test};
'''
hive_drop = Template(hive_drop)


start_date = '2016-07-01'
end_date = '2016-11-01'
drop_table = 0      # 为 0 的时候，表示不删除表， 为 1 ，删除表
istest = '_test'    # 为 ' ' 的时候，表示正式插入表，有任何字符表示创建 test 的表
org_dc_id = '316'


if __name__ == '__main__':
    if drop_table == 1:
        os.system('hive -e "{0}";'.format(hive_drop.substitute(test=istest)))
    pyhive(hive_order.substitute(start_date=start_date,end_date=end_date,dc_id=org_dc_id,test=istest), 'order_data.log')
    pyhive(hive_select.substitute(test=istest),'order_data_select.log')


