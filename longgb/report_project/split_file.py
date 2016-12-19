#-*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np

# 拆分文件
split = 3
path_a = r''
file = pd.read_table(path_a + os.sep + '000000_0', sep='\001', header=None)
len_n = len(file.index)
split_line = map(int,np.linspace(0, len_n, split + 1))[:-1] + [len_n - 1]
for i in range(split):
    file_tmp = file.iloc[range(split_line[i], split_line[i+1]),:]
    file_tmp.to_csv(path_a + os.sep + '000000_0_{0}.csv'.format(i+1), index=False, header=None)




#-*- coding:utf-8 -*-
import os
import time

hive_sql = '''	SELECT
		a.sku_id,
		a.inv,
      c.modify_time,
		c.create_time,
		case when c.modify_time is not null and c.modify_time<='2016-10-11' then 1
	     	when c.modify_time is not null and a.dt>c.modify_time then 1
			else 0 end as white_flag,
		case when c.create_time is not null and a.dt>c.create_time then 1 else 0 end as white_flag_02,
		a.dt as date_s
	FROM
		( 	-- （1）选 每天、每个rdc、每个sku的在途量和、库存和
		SELECT
			delv_center_num,
			dt,
			sku_id,
			sum(stock_qtty) AS inv --库存数量
		FROM
			gdm.gdm_m08_item_stock_day_sum	     	-- 【主表】商品库存日汇总
		WHERE
			dt>='2016-07-01' AND
			dt<='2016-11-01' AND
			delv_center_num='316'
		group by
			delv_center_num,
			dt,
			sku_id
		) a
	LEFT JOIN
		(	-- （3）选 每个fdc、每个sku 的修改时间
		SELECT
			modify_time,
			create_time,
			wid, 		-- sku_id 的信息
			fdcid
		FROM
			fdm.fdm_fdc_whitelist_chain 			-- 白名单表
		WHERE
			start_date<='2016-11-01' AND 	-- 开始日期在 11-01 之前
			end_date>='2016-11-01' AND 		-- 结束日期在 11-01 之后
			yn= 1 AND
			fdcid='630'
		) c
    ON
		a.sku_id=c.wid
    ORDER BY
      date_s
'''

os.system('hive -e "{0}"  >  result.out'.format(hive_sql))


