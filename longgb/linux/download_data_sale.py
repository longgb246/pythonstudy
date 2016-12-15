#-*- coding:utf-8 -*-
import os
from string import Template
import time

# sh allocation_sku_data.sh  2016-07-01 2016-11-01 630/628/658 316

def pyhive(com_str, log_str):
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*30, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format(' '*15 + log_str, log_str))
    os.system('echo "{0}" >> {1} 2>&1;'.format('*'*30, log_str))
    os.system('hive -e "{0}" >> {1} 2>&1;'.format(com_str, log_str))
    os.system('echo "" >> {0} 2>&1;'.format(log_str))

hive_01 = '''DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid01;
create table dev.tmp_allocation_order_data_mid01
as
     select distinct
            f.org_dc_id,
            f.dc_id,
            f.city_id,
            b.sale_ord_det_id,
            b.sale_ord_id,
            b.parent_sale_ord_id,
            b.item_sku_id,
            b.sale_qtty,
            b.sale_ord_tm,
            t4.item_third_cate_cd,
            t4.item_second_cate_cd,
            t4.shelves_dt,
            t4.shelves_tm,
            b.sale_ord_dt
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
            b.rev_addr_city_id = f.dim_city_id;
'''
hive_01 = Template(hive_01)


hive_02 = '''DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid02;
create table dev.tmp_allocation_order_data_mid02
    as
          select distinct
            f.org_dc_id,
            f.dc_id,
            f.city_id,
            b.sale_ord_dt,
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
        b.rev_addr_city_id = f.dim_city_id;
'''
hive_02 = Template(hive_02)


hive_03 = '''DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid03;
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
    a.orderid=b.sale_ord_id;
'''
hive_03 = Template(hive_03)


hive_04 = '''DROP TABLE IF EXISTS dev.tmp_allocation_order_data_mid04;
create table dev.tmp_allocation_order_data_mid04
    as
        select distinct
            f.org_dc_id,
            f.dc_id,
            f.city_id,
            b.sale_ord_dt,
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
	        b.rev_addr_city_id = f.dim_city_id;
'''
hive_04 = Template(hive_04)

hive_05 = '''CREATE TABLE IF NOT EXISTS dev.dev_allocation_sale_data
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
	  	e.white_flag,
	    a.item_third_cate_cd,
      a.item_second_cate_cd,
      a.shelves_dt,
      a.shelves_tm,
	  	a.sale_ord_dt as date_s,
	  	a.dc_id
	from
		dev.tmp_allocation_order_data_mid01 a
	left join
		dev.tmp_allocation_order_data_mid02 b
	on
		a.sale_ord_id=b.sale_ord_id
    and a.item_sku_id=b.item_sku_id
	left join
		dev.tmp_allocation_order_data_mid03 c
	on
		a.sale_ord_id=c.sale_ord_id
    and a.item_sku_id=c.item_sku_id
	left join
		dev.tmp_allocation_order_data_mid04 d
	on
		a.sale_ord_id=d.sale_ord_id
        and a.item_sku_id=d.item_sku_id
	left join
		dev.dev_allocation_sku_data e
	on
		a.item_sku_id=e.sku_id and a.sale_ord_dt=e.date_s
        and a.dc_id=e.dc_id;
'''

start_date = '2016-07-01'
end_date = '2016-11-01'
org_id = '316'
dc_id_list = ['630','628','658']
for each in dc_id_list:
    print "{0} ...".format(each)
    t1 = time.time()
    pyhive(hive_01.substitute(org_id=org_id, dc_id=each), 'sku_data_{0}_new.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)
    t1 = time.time()
    pyhive(hive_02.substitute(org_id=org_id, dc_id=each), 'sku_data_{0}_new.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)
    t1 = time.time()
    pyhive(hive_03.substitute(org_id=org_id, dc_id=each, start_date=start_date, end_date=end_date), 'sku_data_{0}_new.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)
    t1 = time.time()
    pyhive(hive_04.substitute(org_id=org_id, dc_id=each, start_date=start_date), 'sku_data_{0}_new.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)
    t1 = time.time()
    pyhive(hive_05, 'sku_data_{0}_new.log'.format(each))
    print 'run time is {0}s'.format(time.time() - t1)


