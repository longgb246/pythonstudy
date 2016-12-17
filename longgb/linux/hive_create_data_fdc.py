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


hive_fdc = '''DROP TABLE IF EXISTS dev.dev_allocation_fdc_data${test};
CREATE TABLE dev.dev_allocation_fdc_data${test}
   	as
	Select
		org_from,
		org_to,
		actiontime_max,
		alt_max,
		count(distinct id)  as alt_cnt
	from
		(	-- 每个出库订单、每个rdc->fdc的收货完成时间、分布、数量
		SELECT
			a.id,
			a.org_from,
			a.org_to,
			max(to_date(c.actiontime)) as actiontime_max,
			max(datediff(c.actiontime,a.create_date)) as alt_max,
			avg(datediff(c.actiontime,a.create_date)) as alt_avg
		FROM
			(	-- （1）选取出库信息表
			SELECT
				*
			FROM
				fdm.fdm_newdeploy_chuku_chain
			WHERE
				start_date<='${end_date}' AND
				end_date>='${end_date}'  AND
				create_date>='${yn_date}' AND
				yn=1 AND -- 是否删除
				export_state IN (41,42) AND
				export_type IN (2,4,7,8) -- 出库类型 2:非图书单品单件，4:非图书非单品单件，6:采购内配，8：FDC补货退回
			) a
		LEFT JOIN
			(	-- （2）出库与box关联表
			SELECT
				chuku_id,
				box_id
			FROM
				fdm.fdm_newdeploy_send_relation_chain
			WHERE
				start_date<='${end_date}'  AND
				end_date>='${end_date}'  AND
				create_date>='${yn_date}'
			) b
		ON
			a.id=b.chuku_id
		LEFT JOIN
			(	-- （3）box表-入库表
			SELECT
				id,
				actiontime
			FROM
				fdm.fdm_newdeploy_box_chain
			WHERE
				start_date<='${end_date}'  AND
				end_date>='${end_date}'  AND
				create_date>='${yn_date}'
			) c
		ON
			b.box_id=c.id
		group by
			a.id,
			a.org_from,
			a.org_to
		) d
	group by
		org_from,
		org_to,
		actiontime_max,
  		alt_max;
'''
hive_fdc = Template(hive_fdc)

hive_select = '''select * from dev.dev_allocation_fdc_data${test} limit 10;
'''
hive_select = Template(hive_select)

hive_drop = '''DROP TABLE IF EXISTS dev.dev_allocation_fdc_data${test};
'''
hive_drop = Template(hive_drop)

start_date = '2016-07-01'
end_date = '2016-11-01'
yn_date = '2016-11-01'      # 【数据未知】
drop_table = 0      # 为 0 的时候，表示不删除表， 为 1 ，删除表
istest = '_test'    # 为 ' ' 的时候，表示正式插入表，有任何字符表示创建 test 的表


if __name__ == '__main__':
    if drop_table == 1:
        os.system('hive -e "{0}";'.format(hive_drop.substitute(test=istest)))
    pyhive(hive_fdc.substitute(start_date=start_date,end_date=end_date,yn_date=yn_date,test=istest), 'fdc_data.log')
    pyhive(hive_select.substitute(test=istest),'fdc_data_select.log')



