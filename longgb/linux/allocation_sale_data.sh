#!/bin/bash  
# Author : longguangbin&zhangjianshen  
# Date   : 2016-12-05 
# allocation datasets 
# @ $1 start_date
# @ $2 end_date
# @ $3 dc_id
# @ $4 org_id
start_date=$1
end_date=$2
dc_id=$3
org_id=$4

echo 'create mid table dev.tmp_allocation_order_data_mid01'
# 取最全的，与下面的mid02、mid04逻辑相同，仅改变 wh_cate_desc 仓的覆盖范围。
hive -e"DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid01;
	create table dev.tmp_allocation_order_data_mid01
	as
 	select distinct
        f.org_dc_id,
        f.dc_id,
        f.city_id,
        b.dt,
		b.sale_ord_det_id,	
		b.sale_ord_id,		
		b.parent_sale_ord_id,
		b.item_sku_id,
		b.sale_qtty,
		b.sale_ord_tm,
		t4.item_third_cate_cd,
        t4.item_second_cate_cd,
        t4.shelves_dt,
        t4.shelves_tm		
    from
    	dev.tmp_allocation_order_pre_mid01 b
    join
    	dev.tmp_allocation_order_pre_mid02 t4
    on
        b.item_sku_id = t4.item_sku_id
    join
        (
        -- 3、仓分类，取最全的
        select
            *
        from
            dim.dim_store       
        where
            wh_cate_desc not in('大家电', '图书', 'EPT仓'			-- 库房分类名称
            , '生鲜仓', '平台仓', '闪购仓', '测试仓', '保税仓',
            '协同仓')
        ) e
    on
        b.delv_center_num = e.delv_center_num
        and b.store_id    = e.store_id								-- 库房编号
	      join (select * from dev.tmp_allocation_order_pre_mid03 where org_dc_id='${org_id}' and dc_id='${dc_id}') f
    on
        b.rev_addr_city_id = f.dim_city_id;"


echo 'create mid table dev.tmp_allocation_order_data_mid02'
# 创建仅仅包含RDC，订单转移，与上面的 mid01 逻辑一样
hive -e"DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid02;
create table dev.tmp_allocation_order_data_mid02
	as
  	select distinct
		f.org_dc_id,
		f.dc_id,
		f.city_id,
        b.dt,
		b.sale_ord_id,
		b.item_sku_id,
		b.sale_qtty,
		b.sale_ord_tm,
		t4.item_third_cate_cd,
        t4.item_second_cate_cd,
        t4.shelves_dt,
        t4.shelves_tm	
	from
       dev.tmp_allocation_order_pre_mid01 b
    join
        dev.tmp_allocation_order_pre_mid02 t4
    on
        b.item_sku_id = t4.item_sku_id
    join
        (
        -- 这个地方与 mid01 的区别
        select
            *
        from
            dim.dim_store
        where
            wh_cate_desc not in('大家电', '图书', 'EPT仓'
            , '生鲜仓', '平台仓', '闪购仓', '测试仓', '保税仓',
            '协同仓','FDC仓')
        ) e
    on
        b.delv_center_num = e.delv_center_num
        and b.store_id    = e.store_id
   join (select * from dev.tmp_allocation_order_pre_mid03 where org_dc_id='${org_id}' and dc_id='${dc_id}') f 
	on
        b.rev_addr_city_id = f.dim_city_id;"
						         
								         
								         
echo 'create mid table dev.tmp_allocation_order_data_mid03'
# 创建内配驱动
	hive -e"DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid03;
	create table dev.tmp_allocation_order_data_mid03
   	as
	select 
		b.* 
	from 
		(select 
			distinct orderid 
		from 
			fdm.fdm_newdeploy_chuku_chain c, 
			fdm.fdm_newdeploy_order_relation_chain o                    	
		where 
			c.id=o.chuku_id                             	
			and c.export_type in(2,4,7,8)  
			and c.create_by = '订单worker' 
			and c.org_from = '${org_id}'
			and c.org_to = '${dc_id}' 
			and c.yn in (1, 3, 5)  
			and c.create_date >='${start_date}'
			and c.create_date <='${end_date}'
		)	a
	join 
		dev.tmp_allocation_order_data_mid01 b
	on 
		a.orderid=b.sale_ord_id;"
		    
		    
		    
echo 'create mid table dev.tmp_allocation_order_data_mid04'
# 创建仅仅包含FDC，FDC本地发货 与上面的 mid01 逻辑一样
hive -e"DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid04;
	create table dev.tmp_allocation_order_data_mid04
		as
							select distinct
								f.org_dc_id,
								f.dc_id,
								f.city_id,
                b.dt,
								b.sale_ord_id,
								b.item_sku_id,
								b.sale_qtty,
								b.sale_ord_tm,
								t4.item_third_cate_cd,
                t4.item_second_cate_cd,
                w.white_flag	
        from
            (select 
            	* 
        	from 
        		dev.dev_allocation_sku_data
			where 
				dc_id = '${dc_id}' 
				and date_s >= '${start_date}'
			) w
        join
      dev.tmp_allocation_order_pre_mid01 b
        on  
        	w.sku_id = b.item_sku_id 
        	and w.date_s=b.dt
        join
         dev.tmp_allocation_order_pre_mid02 t4
        on
            b.item_sku_id = t4.item_sku_id
        join
            (
            -- 这个地方与 mid01 的区别
            select
                *
            from
                dim.dim_store
            where
                wh_cate_desc='FDC仓'
            ) e
        on
            b.delv_center_num = e.delv_center_num
            and b.store_id    = e.store_id
	   join (select * from dev.tmp_allocation_order_pre_mid03 where org_dc_id='${org_id}' and dc_id='${dc_id}') f 
		on
	        b.rev_addr_city_id = f.dim_city_id;"

echo 'gene the result table'
hive -e"CREATE TABLE IF NOT EXISTS dev.dev_allocation_sale_data
	(	org_dc_id         string,
		sale_ord_det_id string,	
		sale_ord_id string,		
		parent_sale_ord_id string,
		item_sku_id				string,		-- skuid
		sale_qtty				  int,		-- 销售数量
		sale_ord_tm				string,		-- 销售订单订购时间
		sale_ord_type 			string,		-- 订单配送类型
		sale_ord_white_flag		string, 		-- 是否包括白名单
		item_third_cate_cd  string,   --sku所属三级分类
		item_second_cate_cd  string,	  --sku所属二级分类
		shelves_dt  string, --上架日期
    shelves_tm   string --上架时间
	)
	PARTITIONED by (date_s string,dc_id int);
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.max.dynamic.partitions=2000;
set hive.exec.max.dynamic.partitions.pernode=2000;
insert overwrite table dev.dev_allocation_sale_data partition(date_s,dc_id)
	select
	 	a.org_dc_id,
		a.sale_ord_det_id,	
    a.sale_ord_id,		
    a.parent_sale_ord_id,
    a.item_sku_id,
		a.sale_qtty,
		a.sale_ord_tm	,
	  	case when b.sale_ord_id is not null then 'rdc'
	  when c.sale_ord_id is not null then 'fdc_rdc'
	  when d.sale_ord_id is not null then 'fdc'
	  	else 'other' end,
	  	d.white_flag,
	    a.item_third_cate_cd,
      a.item_second_cate_cd,
      a.shelves_dt,
      a.shelves_tm,	
	  	a.dt as date_s,
	  	a.dc_id
	from 
		dev.tmp_allocation_order_data_mid01 a
	left join 
		dev.tmp_allocation_order_data_mid02 b
	on 
		a.sale_ord_id=b.sale_ord_id
	left join 
		dev.tmp_allocation_order_data_mid03 c
	on 
		a.sale_ord_id=c.sale_ord_id
	left join 
		dev.tmp_allocation_order_data_mid04 d
	on 
		a.sale_ord_id=d.sale_ord_id
	left join 
		dev.dev_allocation_sku_data e
	on 
		a.item_sku_id=e.sku_id and a.sale_ord_dt=e.date_s;"
