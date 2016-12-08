#!/bin/bash  
# Author : longguangbin&zhangjianshen  
# Date   : 2016-12-05 
# allocation datasets 
# @ $1 start_date
# @ $2 end_date
# @ $3 dc_id
start_date=$1
end_date=$2
dc_id=$3



hive -e"set hive.exec.dynamic.partition=true;
	      set hive.exec.dynamic.partition.mode=nonstrict;
        CREATE TABLE IF NOT EXISTS dev.dev_allocation_order_data
		(
			arrive_time 	string,	-- 到达时间
			item_sku_id 	string,	-- skuid
			arrive_quantity	string	-- 实际到达量
		)
		PARTITIONED by (rdc_id string);
		
	INSERT overwrite table dev.dev_allocation_order_data partition(rdc_id)
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
	        to_date(t2.create_tm) BETWEEN ${start_date} AND ${end_date} AND
	        t2.int_org_num = ${dc_id}
	    group by 
		    t2.int_org_num,
		    to_date(t2.complete_dt),
		    t2.sku_id;"