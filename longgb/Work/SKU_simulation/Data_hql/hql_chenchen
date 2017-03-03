

-- 查询【调拨量】的codes，原版
drop table if exists dev.dev_inv_opt_fdc_sku_daily_summary_tmp1_$dt_s;
create table dev.dev_inv_opt_fdc_sku_daily_summary_tmp1_$dt_s stored as orc as
select
    '$dt' as dt,
    case when A.fdcid is not null then A.fdcid else B.fdc_id end as fdc_id,
    case when A.wid is not null then A.wid else B.sku_id end as sku_id,
    case when A.wid is not null then 1 else 0 end as is_whitelist,
    B.rdc_id,
    B.plan_num_auto,
    B.delivered_num_auto,
    B.plan_num_manual,
    B.delivered_num_manual,
    B.plan_num_order,
    B.delivered_num_order
from
    (
    select
        t2.wid,
        t2.fdcid
    from
        dim.dim_dc_info t1
    join
        fdm.fdm_fdc_whitelist_chain t2
    on
        t1.dc_id = t2.fdcid
    where
        t2.start_date <= '$dt' and
        t2.end_date > '$dt' and
        to_date(t2.create_time) <= '$dt' and
        to_date(t2.modify_time) <= '$dt' and
        t2.yn = 1 and
        t1.dc_type = 1) A
full outer join
    (
    select
        to_date(ck.create_date) as dt,
        co.art_no as sku_id,
        ck.org_to as fdc_id,
        ck.org_from as rdc_id,
        sum(case when ck.export_type = 7 and ck.create_by = 'fdc' then plan_num else 0 end) as plan_num_auto,                   -- 计划调拨量
        sum(case when ck.export_type = 7 and ck.create_by = 'fdc' then delivered_num else 0 end) as delivered_num_auto,         -- 实际调拨量
        sum(case when ck.export_type in (5, 7) and ck.create_by not in ('fdc', 'system', '订单worker') then plan_num else 0 end) as plan_num_manual,
        sum(case when ck.export_type in (5, 7) and ck.create_by not in ('fdc', 'system', '订单worker') then delivered_num else 0 end) as delivered_num_manual,
        sum(case when ck.export_type in (1, 2, 3, 4, 7) and ck.create_by = '订单worker' then plan_num else 0 end) as plan_num_order,
        sum(case when ck.export_type in (1, 2, 3, 4, 7) and ck.create_by = '订单worker' then delivered_num else 0 end) as delivered_num_order
    from
        (select * from dim.dim_dc_info where dc_type = 1) di                    -- 配送中心所属关系， 取 dc_type = 1， 1-FDC。
    join
        fdm.fdm_newdeploy_chuku_chain ck                                        -- 内配计划出库表（内配单）
    on
        di.dc_id = ck.org_to
    join
        fdm.fdm_newdeploy_chuorders_chain co                                    -- 未知表，看看。
    on
        ck.id = co.chuku_id
    where
        co.dp = 'ACTIVE' and
        ck.dp = 'ACTIVE' and
        ck.yn in (1, 3, 5) and                                                  -- 1---正常，3---删除处理中， 5---删除失败
        ck.org_from in (3, 4, 5, 6, 9, 10, 316, 682) and                        -- 配出机构
        to_date(ck.create_date) = '$dt'
    group by
        to_date(ck.create_date),
        co.art_no,
        ck.org_to,
        ck.org_from) B
on
    A.wid = B.sku_id and
    A.fdcid = B.fdc_id;





-- 第二个SQL
select
    a.dt,
    a.fdcid,
    a.wid,
    b.forecast_daily_override_sales,
    c.total_sales,
    d.stock_qtty
from
    (
        select
            dt,
            wid,
            fdcid
        from
            dev.dev_inv_opt_fdc_sku_daily_summary_mid01
        where
            dt >= ${start_date} AND
            dt <=${end_date}  And
            dc_id=${dc_id}
    ) a
    left join
    (
        SELECT
            dt,
            dc_id,
            sku_id,
            forecast_begin_date,
            forecast_days,
            forecast_daily_override_sales  ---7天预测
        FROM
            app.app_pf_forecast_result_fdc_di  ---预测数据
        WHERE
            dt >= ${start_date} AND
            dt <=${end_date}  And
            dc_id=${dc_id}
     ) b
    on a.dt=b.dt
        and a.fdcid=b.dc_id
        and a.wid=b.sku_id
    left join
    (select
        dt,
        sku_id,
        dc_id,
        order_date,
        total_sales
    from
        app.app_sfs_sales_dc  ----FDC实际销量表
    where
        dt >= ${start_date} AND
        dt <=${end_date}  And
        dc_id=${dc_id}
    )  c
    on a.dt=c.dt
        and a.fdcid=c.dc_id
        and a.wid=c.sku_id
    left join
    (select
        dt,
        delv_center_num,
        sku_id,
        sum(stock_qtty+in_transit_qtty-sale_reserve_qtty)  as stock_qtty
    from
        gdm.gdm_m08_item_stock_day_sum   ---库存数据
    where
        dt between '$start_date' and '$end_date'
        and delv_center_num=${dc_id}
    group by
        dt,
        delv_center_num,
        sku_id) d
    on a.dt=d.dt
        and a.fdcid=d.delv_center_num
        and a.wid=d.sku_id;

