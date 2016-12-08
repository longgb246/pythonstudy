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


# run the programe :
# sh allocation_sku_data.sh  2016-07-01 2016-11-01 630 316


#create table only one time
# hive -e"set hive.exec.dynamic.partition=true;
# set hive.exec.dynamic.partition.mode=nonstrict;
# CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data
# ( sku_id	 string,
# 	forecast_begin_date	 string,
# 	forecast_days	int,
# 	forecast_daily_override_sales	array<double>,
# 	forecast_weekly_override_sales	array<double>,
# 	forecast_weekly_std	array<double>,
# 	forecast_daily_std array<double>,
# 	variance string,
# 	ofdsales string,
# 	inv	double,
# 	arrive_quantity	int,
# 	open_po	 int,
# 	white_flag	 int)
# 	PARTITIONED by (date_s  string,dc_id int);
	
# 	insert OVERWRITE table dev.dev_allocation_sku_data partition(date_s,dc_id)
# 	SELECT
# 	a.sku_id,
# 	b.forecast_begin_date,
# 	b.forecast_days,
# 	b.forecast_daily_override_sales,
# 	b.forecast_weekly_override_sales,
# 	b.forecast_weekly_std,
# 	b.forecast_daily_std,
# 	d.variance,
# 	d.ofdsales,
# 	a.inv, 
# 	0 as arrive_quantity, --这个会在程序中更新
# 	a.open_po,
# 	case when c.modify_time<a.dt then 1 else 0 end as white_flag,
# 	a.dt as date_s,
# 	b.dc_id  		  	
# FROM
# 	(SELECT 
# 		delv_center_num,
# 		dt,
# 		sku_id,
# 		sum(in_transit_qtty) AS open_po,
# 		sum(stock_qtty) AS inv --库存数量	
# 	FROM 
# 		gdm.gdm_m08_item_stock_day_sum
# 	WHERE 
# 		dt>=${start_date} AND 
# 		dt<=${end_date} AND
# 		delv_center_num=${dc_id}
# 	group by 
# 		delv_center_num,		
# 		dt,
# 		sku_id
# 	) a
# LEFT JOIN
# 	(SELECT
# 		dt,
# 		dc_id,
# 		sku_id,
# 		forecast_begin_date,
# 		forecast_days,
# 		forecast_daily_override_sales ,
# 		forecast_weekly_override_sales,
# 		forecast_weekly_std,
# 		forecast_daily_std--新增每日
# 	FROM
# 	    app.app_pf_forecast_result_fdc_di
# 	WHERE
# 	    dt >= ${start_date} AND 
# 	    dt <=${end_date}  And
# 	    dc_id=${dc_id}
#     ) b
# ON
# 	a.sku_id=b.sku_id AND
# 	a.dt=b.dt
# LEFT JOIN
# 	(SELECT 
# 		modify_time,
# 		wid, 					
# 		fdcid
# 	FROM 
# 		fdm.fdm_fdc_whitelist_chain 
# 	WHERE  
# 		start_date<=${end_date} AND 
# 		end_date>=${end_date} AND
# 		yn= 1 AND
# 		fdcid=${dc_id}
# 	) c
# ON
# 	a.sku_id=c.wid
# LEFT JOIN
# 	(select
# 	  dt,  
# 		wid as sku_id,
# 		dcid,
# 		variance,
# 		ofdsales
# 		from app.app_sfs_rdc_forecast_result 
# 	WHERE
# 	    dt >= ${start_date} AND 
# 	    dt <=${end_date}  And
# 	    dcid=${org_dc_id}
#     ) d
#     on 		
#     a.sku_id=d.sku_id AND
# 	  a.dt=d.dt;"
		




# 创建临时表
	hive -e "set hive.exec.dynamic.partition=true;
	set hive.exec.dynamic.partition.mode=nonstrict;
	DROP TABLE IF EXISTS dev.dev_allocation_sku_data;
	CREATE TABLE IF Not EXISTS dev.dev_allocation_sku_data
	( sku_id	 string,
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
		white_flag	 int)
		PARTITIONED by (date_s  string,dc_id int);
		
	-- 01表
	DROP TABLE IF EXISTS dev.tmp_allocation_order_pre_mid02_01;
	CREATE TABLE dev.tmp_allocation_order_pre_mid02_01
    AS
	SELECT 
			delv_center_num,
			dt,
			sku_id,
			sum(in_transit_qtty) AS open_po,
			sum(stock_qtty) AS inv --库存数量	
		FROM 
			gdm.gdm_m08_item_stock_day_sum
		WHERE 
			dt>=${start_date} AND 
			dt<=${end_date} AND
			delv_center_num=${dc_id}
		group by 
			delv_center_num,		
			dt,
			sku_id;

	-- 02表
	DROP TABLE IF EXISTS dev.tmp_app_pf_forecast_result_fdc_di_01;
	CREATE TABLE dev.tmp_app_pf_forecast_result_fdc_di_01
    AS
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
		    app.app_pf_forecast_result_fdc_di
		WHERE
		    dt >= ${start_date} AND 
		    dt <=${end_date}  And
		    dc_id=${dc_id};

	-- 03表
	DROP TABLE IF EXISTS dev.tmp_fdm_fdc_whitelist_chain_01;
	CREATE TABLE dev.tmp_fdm_fdc_whitelist_chain_01
    AS
	SELECT 
			modify_time,
			wid, 					
			fdcid
		FROM 
			fdm.fdm_fdc_whitelist_chain 
		WHERE  
			start_date<=${end_date} AND 
			end_date>=${end_date} AND
			yn= 1 AND
			fdcid=${dc_id};

	-- 04表
	DROP TABLE IF EXISTS dev.tmp_app_sfs_rdc_forecast_result_01;
	CREATE TABLE dev.tmp_app_sfs_rdc_forecast_result_01
    AS
	SELECT
		  dt,  
			wid as sku_id,
			dcid,
			variance,
			ofdsales
			from app.app_sfs_rdc_forecast_result 
		WHERE
		    dt >= ${start_date} AND 
		    dt <=${end_date}  And
		    dcid=${org_dc_id};


	INSERT OVERWRITE table dev.dev_allocation_sku_data partition(date_s,dc_id)
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
		case when c.modify_time<a.dt then 1 else 0 end as white_flag,
		a.dt as date_s,
		b.dc_id  		  	
	FROM
		dev.tmp_allocation_order_pre_mid02_01 a
	LEFT JOIN
		dev.tmp_app_pf_forecast_result_fdc_di_01 b
	ON
		a.sku_id=b.sku_id AND
		a.dt=b.dt
	LEFT JOIN
		dev.tmp_fdm_fdc_whitelist_chain_01 c
	ON
		a.sku_id=c.wid
	LEFT JOIN
		dev.tmp_app_sfs_rdc_forecast_result_01 d
    ON 		
	    a.sku_id=d.sku_id AND
		a.dt=d.dt;"