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

echo 'create mid table dev.allocation_sale_pre_once talbes'
hive -e"DROP TABLE IF EXISTS dev.tmp_allocation_order_pre_mid01;
    create table dev.tmp_allocation_order_pre_mid01
    as
        -- 【1】选每天、每个sku、每个sale单 的 销售量
        select
            dt,
			sale_ord_id,
			item_sku_id,
			sale_qtty,               -- 销售量
			sale_ord_tm,             -- 销售时间
			delv_center_num,         -- 配送中心 - rdc
			store_id,                -- 库房编号
			rev_addr_city_id         -- 收货地址城市编号
        from
            gdm.gdm_m04_ord_det_sum
        where
            dt                     >= '${start_date}'
            and sale_ord_dt        >= '${start_date}'
            and sale_ord_dt        <= '${end_date}'
            and sale_ord_type_cd    = '0'                       --只取一般订单
            and split_status_cd    in('2', '3')                 -- 拆分状态
            and sale_ord_valid_flag = 1;                        --有效
    DROP TABLE IF EXISTS dev.tmp_allocation_order_pre_mid02;

    create table dev.tmp_allocation_order_pre_mid02
    as
        -- 【2】选sku、二级、三级品类
        SELECT
            item_sku_id,
            item_third_cate_cd,
            item_second_cate_cd
        FROM
            gdm.gdm_m03_item_sku_da
        WHERE
            dt= sysdate( - 1)
            and item_third_cate_cd NOT            IN
            ('1009', '874', '1137', '10011',
            '10970', '2643', '1446', '10010',
            '1076', '6980', '12258', '1195',
            '10969', '982', '12360', '11192',
            '9402', '9404', '12428', '7071', '7069'
            , '379', '12772', '705', '970', '12429'
            , '706', '709', '704', '707', '7370',
            '1117', '12216')
            and data_type = '1';                                --京东自营 1自营普通商品不包含图书
    DROP TABLE IF EXISTS dev.tmp_allocation_order_pre_mid03;

    create table dev.tmp_allocation_order_pre_mid03
    as
        --【3】选 rdc、fdc、城市、配送中心
        select distinct 
            c.org_dc_id,
            c.dc_id,
            b.city_id, 
            b.dim_city_id,
            a.delv_center_num			
		from
			(
            -- （3.1）配送中心
			select 
				* 
			from 
				dim.dim_base_delivery_area_supply_deduplication) a 
		JOIN
			-- （3.2）区县 -> 城市
			dim.dim_county B 
		on 
			a.county_id=b.county_id									-- 区县编号
		JOIN 
			(
			-- （3.3）RDC -> FDC
			select 
				dc_id,org_dc_id 
			from 
				dim.dim_dc_info 
			WHERE  dc_type in (0,1)
			) C
		on 
			a.delv_center_num=c.dc_id;
"