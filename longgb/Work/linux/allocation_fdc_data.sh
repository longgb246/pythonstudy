#!/bin/bash  
# Author : longguangbin&zhangjianshen  
# Date   : 2016-12-05 
# allocation datasets 
# @ $1 start_date
# @ $2 end_date
# @ $3 yn_date
start_date=$1
end_date=$2
yn_date=$3


hive -e"
DROP TABLE IF EXISTS dev.dev_allocation_fdc_data;
CREATE TABLE dev.dev_allocation_fdc_data
   	as
	Select
		-- 【1】每个rdc->fdc的收货完成时间、分布、数量
		org_from,
		org_to,
		actiontime_max, 
		alt_max,
		count(distinct id)  as alt_cnt
	from	
		(	-- 【2】每个出库订单、每个rdc->fdc的收货完成时间、分布、数量
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
				yn=1 AND 						-- 是否删除 1:正常
				export_state IN (41,42) AND 	-- 状态  	41:配货完成， 42:缺货配货完成
				export_type IN (2,4,7,8) 		-- 出库类型 2:非图书单品单件，4:非图书非单品单件，6:采购内配，8：FDC补货退回
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
				start_date<='${end_date}' AND 
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
  		alt_max;"