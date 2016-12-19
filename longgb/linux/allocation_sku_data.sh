#!/bin/bash  
# Author : longguangbin&zhangjianshen  
# Date   : 2016-12-05 
# allocation datasets 
# @ $1 start_date 2016-10-01
# @ $2 end_date 2016-10-01
# @ $3 dc_id
# @ $4 org_dc_id
start_date=$1
end_date=$2
dc_id=$3
org_dc_id=$4

#create table only one time
	hive -e"set hive.exec.dynamic.partition=true;
	set hive.exec.dynamic.partition.mode=nonstrict;
	CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data
	( 	
		-- 【1】每天、每个fdc、每个sku的 预测数据、白名单等
		sku_id	 string,
		forecast_begin_date	 string,
		forecast_days	int,
		forecast_daily_override_sales	array<double>,
		forecast_weekly_override_sales	array<double>,
		forecast_weekly_std	array<double>,
		forecast_daily_std array<double>,
		variance string,
		ofdsales string,
		inv	double,
		arrive_quantity	int,
		open_po	 int,
		white_flag	 int,
		white_flag_01  int)
		PARTITIONED by (date_s  string,dc_id int);
		
	insert OVERWRITE table dev.dev_allocation_sku_data partition(date_s,dc_id)
	SELECT
		a.sku_id,
		b.forecast_begin_date,
		b.forecast_days,
		b.forecast_daily_override_sales,
		b.forecast_weekly_override_sales,
		b.forecast_weekly_std,
		b.forecast_daily_std,
		d.variance,
		d.ofdsales,
		a.inv, 
		0 as arrive_quantity, --这个会在程序中更新
		a.open_po,
		case when c.modify_time is not null and c.modify_time<='2016-10-11' then 1 
	     	when c.modify_time is not null and a.dt>c.modify_time then 1
			else 0 end as white_flag,
		case when c.create_time is not null and a.dt>c.create_time then 1 else 0 end as white_flag_01,
		a.dt as date_s,
		b.dc_id  		  	
	FROM
		( 	-- （1）选 每天、每个rdc、每个sku的在途量和、库存和
		SELECT 
			delv_center_num,
			dt,
			sku_id,
			sum(in_transit_qtty) AS open_po,
			sum(stock_qtty) AS inv --库存数量	
		FROM 
			gdm.gdm_m08_item_stock_day_sum	     	-- 【主表】商品库存日汇总
		WHERE 
			dt>='${start_date}' AND 
			dt<='${end_date}' AND
			delv_center_num='${org_dc_id}'
		group by 
			delv_center_num,		
			dt,
			sku_id
		) a
	LEFT JOIN
		(	-- （2）选 每天、每个fdc、每个sku的预测数据
		SELECT
			dt,
			dc_id,
			sku_id,
			forecast_begin_date,
			forecast_days,
			forecast_daily_override_sales ,
			forecast_weekly_override_sales,
			forecast_weekly_std,
			forecast_daily_std--新增每日
		FROM
		    app.app_pf_forecast_result_fdc_di		-- 预测信息
		WHERE
		    dt >= '${start_date}' AND 
		    dt <='${end_date}'  And
		    dc_id='${dc_id}'
	    ) b
	ON
		a.sku_id=b.sku_id AND
		a.dt=b.dt
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
			start_date<='${end_date}' AND 	-- 开始日期在 11-01 之前
			end_date>='${end_date}' AND 		-- 结束日期在 11-01 之后
			yn= 1 AND
			fdcid='${dc_id}'
		) c
	ON
		a.sku_id=c.wid
	LEFT JOIN
		(	-- （4）选 每天、每个rdc、每个sku的标准差、预测销量
		select
		  	dt,  
			wid as sku_id,
			dcid,
			variance,
			ofdsales
		from 
			app.app_sfs_rdc_forecast_result 
		WHERE
		    dt >= '${start_date}' AND 
		    dt <='${end_date}'  And
		    dcid='${org_dc_id}'
	    ) d
    on 		
	    a.sku_id=d.sku_id AND
	  	a.dt=d.dt;"
		