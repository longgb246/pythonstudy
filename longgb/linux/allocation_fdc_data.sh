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
	(
		date_s	 	string,
		org_from 		int,
		org_to     int,
		alt 		array<int>, 	-- 调拨时长分布
		alt_prob 	array<double>  -- 调拨时长概率
	);
INSERT INTO dev.dev_allocation_fdc_data
	select
	  ${end_date},
		org_from,
		org_to,
		collect_all(alt_max), --若不支持数字，先将其转换为string
		collect_all(alt_prob)      
	from
		(select
			org_from,
			org_to,
			alt_max,
			alt_cnt/sum(alt_cnt)over(partition by org_from,org_to) as alt_prob
		from
			(Select
				org_from,
				org_to,
				alt_max,
				count(distinct id)  as alt_cnt
			from	
				(SELECT
					a.id,
					a.org_from,
					a.org_to,
					max(datediff(c.actiontime,a.create_date)) as alt_max,
					avg(datediff(c.actiontime,a.create_date)) as alt_avg		
				FROM
					(SELECT 
						* 
					FROM  
						fdm.fdm_newdeploy_chuku_chain 
					WHERE 
						start_date<=${end_date} AND 
						end_date>=${end_date}  AND 
						create_date>=${yn_date} AND 
						yn=1 AND -- 是否删除
						export_state IN (41,42) AND
						export_type IN (2,4,7,8) -- 出库类型 2:非图书单品单件，4:非图书非单品单件，6:采购内配，8：FDC补货退回
					) a
				LEFT JOIN
					(SELECT 
						chuku_id,					
						box_id						
					FROM 
						fdm.fdm_newdeploy_send_relation_chain
					WHERE 
						start_date<=${end_date}  AND 
						end_date>=${end_date}  AND 
						create_date>=${yn_date}
					) b
				ON 
					a.id=b.chuku_id
				LEFT JOIN
					(SELECT 
						id,							
						actiontime					
					FROM 
						fdm.fdm_newdeploy_box_chain
					WHERE 
						start_date<=${end_date}  AND 
						end_date>=${end_date}  AND 
						create_date>=${yn_date}
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
	      		alt_max
      		) e
  		) f
	group by 
		org_from,
		org_to;"