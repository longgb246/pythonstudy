-- [ 有用的表 ]：

-- 1、dev.dev_tmp_lgb_combine_allocation_Forecast_price
-- 由下面的 dev.dev_tmp_lgb_allocation_qttys、 dev.dev_tmp_lgb_allocation_qttys 加工而得
-- [ 字段 ]：
-- rdc_id  string,                      rdc 为 4
-- fdc_id  string,                      fdc 为 605
-- sku_id  string,
-- is_whitelist    int,                 是否白名单
-- plan_allocation_qtty    double,      计划调拨量
-- actual_allocation_qtty  double,      实际调拨量
-- forecast_daily_override_sales   array<double>,       销量预测均值array
-- forecast_sales_mean     double,                      销量预测均值，为 forecast_daily_override_sales 的值
-- sale        double,                  当日销量
-- sale_all    double,                  2016-12-20 到 2016-12-31 日的平均销量，以防 12.12 影响
-- stock_qtty  double,                  当日库存
-- date_s  string,                      日期，仅有 2016-12-20 数据
-- price   double,                      进货价

-- 2、dev.dev_tmp_lgb_allocation_qttys      # 按照 date_s 分区
-- fdc_id	 string,
-- sku_id	 string,
-- is_whitelist	int,
-- rdc_id   string,
-- plan_allocation_qtty    double,
-- actual_allocation_qtty  double,
-- date_s  string                           仅仅取了 2016-12-20 数据

-- 3、dev.dev_tmp_lgb_saleForecast
-- a.dt,                                    是 date_s， 有从 2016-12-01 到 2016-12-31 的数据
-- a.fdcid,
-- a.wid,                                   是 sku_id
-- b.forecast_daily_override_sales,
-- c.total_sales,                           当日销量
-- d.stock_qtty                             当日库存


-- ============================================================================
-- =                1、建表：dev.dev_tmp_lgb_allocation_qttys_1220             =
-- ============================================================================
-- 创建的是固定的。下面创建一个分区的
drop table if exists dev.dev_tmp_lgb_allocation_qttys_1220 ;
create table dev.dev_tmp_lgb_allocation_qttys_1220 as
select
    case when A.fdcid is not null then A.fdcid else B.fdc_id end as fdc_id,
    case when A.wid is not null then A.wid else B.sku_id end as sku_id,
    case when A.wid is not null then 1 else 0 end as is_whitelist,
    B.rdc_id,
    B.plan_num_auto as plan_allocation_qtty,
    B.delivered_num_auto as actual_allocation_qtty,
    '2016-12-20' as date_s
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
        t2.start_date <= '2016-12-20' and
        t2.end_date > '2016-12-20' and
        to_date(t2.create_time) <= '2016-12-20' and
        to_date(t2.modify_time) <= '2016-12-20' and
        t2.yn = 1 and
        t1.dc_type = 1) A
full outer join
    (
    select
        to_date(ck.create_date) as dt,
        co.art_no as sku_id,
        ck.org_to as fdc_id,
        ck.org_from as rdc_id,
        sum(case when ck.export_type = 7 and ck.create_by = "fdc" then plan_num else 0 end) as plan_num_auto,                   -- 计划调拨量
        sum(case when ck.export_type = 7 and ck.create_by = "fdc" then delivered_num else 0 end) as delivered_num_auto          -- 实际调拨量
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
        co.dp = "ACTIVE" and
        ck.dp = "ACTIVE" and
        ck.yn in (1, 3, 5) and                                                  -- 1---正常，3---删除处理中， 5---删除失败
        -- ck.org_from in (3, 4, 5, 6, 9, 10, 316, 682) and                     -- 配出机构
        ck.org_from = 4 and                                                     -- 只取出 RDC 为 4 的
        ck.org_to = 605 and                                                     -- 只取出 FDC 为 605 的
        to_date(ck.create_date) = '2016-12-20'
    group by
        to_date(ck.create_date),
        co.art_no,
        ck.org_to,
        ck.org_from) B
on
    A.wid = B.sku_id and
    A.fdcid = B.fdc_id;

-- 创建一个分区的插入。
drop table if exists dev.dev_tmp_lgb_allocation_qttys ;

set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
CREATE TABLE IF Not EXISTS dev.dev_tmp_lgb_allocation_qttys
(
    fdc_id	 string,
    sku_id	 string,
    is_whitelist	int,
    rdc_id   string,
    plan_allocation_qtty    double,
    actual_allocation_qtty  double)
    PARTITIONED by (date_s  string);
insert OVERWRITE table dev.dev_tmp_lgb_allocation_qttys  partition(date_s)
select
    case when A.fdcid is not null then A.fdcid else B.fdc_id end as fdc_id,
    case when A.wid is not null then A.wid else B.sku_id end as sku_id,
    case when A.wid is not null then 1 else 0 end as is_whitelist,
    B.rdc_id,
    B.plan_num_auto as plan_allocation_qtty,
    B.delivered_num_auto as actual_allocation_qtty,
    '2016-12-20' as date_s
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
        t2.start_date <= '2016-12-20' and
        t2.end_date > '2016-12-20' and
        to_date(t2.create_time) <= '2016-12-20' and
        to_date(t2.modify_time) <= '2016-12-20' and
        t2.yn = 1 and
        t1.dc_type = 1) A
full outer join
    (
    select
        to_date(ck.create_date) as dt,
        co.art_no as sku_id,
        ck.org_to as fdc_id,
        ck.org_from as rdc_id,
        sum(case when ck.export_type = 7 and ck.create_by = "fdc" then plan_num else 0 end) as plan_num_auto,                   -- 计划调拨量
        sum(case when ck.export_type = 7 and ck.create_by = "fdc" then delivered_num else 0 end) as delivered_num_auto          -- 实际调拨量
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
        co.dp = "ACTIVE" and
        ck.dp = "ACTIVE" and
        ck.yn in (1, 3, 5) and                                                  -- 1---正常，3---删除处理中， 5---删除失败
        -- ck.org_from in (3, 4, 5, 6, 9, 10, 316, 682) and                     -- 配出机构
        ck.org_from = 4 and                                                     -- 只取出 RDC 为 4 的
        ck.org_to = 605 and                                                     -- 只取出 FDC 为 605 的
        to_date(ck.create_date) = '2016-12-20'
    group by
        to_date(ck.create_date),
        co.art_no,
        ck.org_to,
        ck.org_from) B
on
    A.wid = B.sku_id and
    A.fdcid = B.fdc_id;


-- ============================================================================
-- =                2、建表：dev.dev_tmp_lgb_saleForecast                      =
-- ============================================================================
drop table if exists dev.dev_tmp_lgb_saleForecast;
create table dev.dev_tmp_lgb_saleForecast as
select
    a.dt,
    a.fdcid,
    a.wid,          -- sku_id
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
            dt >= '2016-12-01' AND
            dt <= '2016-12-31'  And
            fdcid = "605"
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
            dt >= '2016-12-01' AND
            dt <='2016-12-31'  And
            dc_id="605"
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
        dt >= '2016-12-01' AND
        dt <='2016-12-31'  And
        dc_id="605"
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
        dt between '2016-12-01' and '2016-12-31'
        and delv_center_num="605"
    group by
        dt,
        delv_center_num,
        sku_id) d
    on a.dt=d.dt
        and a.fdcid=d.delv_center_num
        and a.wid=d.sku_id;



-- ============================================================================
-- =                              3、统计数据                                  =
-- ============================================================================
-- 1、参与调拨的数量是多少
select
    count(distinct sku_id),
    fdc_id,
    rdc_id
from
    dev.dev_tmp_lgb_allocation_qttys
WHERE
	fdc_id = 605    AND
	rdc_id = 4      and
	date_s = '2016-12-20'
group by
    fdc_id,
    rdc_id


-- 2、合并表
drop table if exists dev.dev_tmp_lgb_combine_allocation_Forecast_1220;
create table dev.dev_tmp_lgb_combine_allocation_Forecast_1220 as
select
    a.rdc_id,
    a.fdc_id,
    a.sku_id,
    a.is_whitelist,
    a.plan_allocation_qtty,
    a.actual_allocation_qtty,
    d.forecast_daily_override_sales,
    d.forecast_sales_mean,
    d.sale,
    d.sale_all,
    d.stock_qtty,
    a.date_s
from
    (
        select
            a.rdc_id,
            a.fdc_id,
            a.sku_id,
            a.is_whitelist,
            a.plan_allocation_qtty,
            a.actual_allocation_qtty,
            a.date_s
        from
            dev.dev_tmp_lgb_allocation_qttys_1220 a
        where
            a.fdc_id = 605  AND
            a.rdc_id = 4    AND
            a.date_s = '2016-12-20'
    ) a
left join
    (
        select
            b.sku_id,
            b.forecast_daily_override_sales,
            b.forecast_sales_mean,
            b.sale,
            b.sale_all,
            c.stock_qtty
        from
            (
                select
                    m.sku_id,
                    m.forecast_daily_override_sales,
                    m.forecast_sales_mean,
                    m.sale,
                    n.sale_all
                from
                    (
                        -- 取20号的数据
                        select
                            wid as sku_id,
                            forecast_daily_override_sales,
                            (forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])/7  as forecast_sales_mean,
                            total_sales  as sale
                        from
                            dev.dev_tmp_lgb_saleForecast
                        where
                            fdcid = 605 and
                            dt = '2016-12-20'
                    )   m
                left join
                    (
                        -- 取平均的销量
                        select
                            wid as sku_id,
                            avg(total_sales)   as sale_all
                        from
                            dev.dev_tmp_lgb_saleForecast
                        where
                            fdcid = 605     and
                            dt >= '2016-12-20'
                        group by
                            wid
                    )   n
                on
                    m.sku_id = n.sku_id
            )  b
        left join
            (
                -- 取inv的前一天数据
                select
                    wid as sku_id,
                    stock_qtty
                from
                    dev.dev_tmp_lgb_saleForecast
                where
                    fdcid = 605 and
                    dt = '2016-12-19'
            )  c
        on
            b.sku_id = c.sku_id
    ) d
on
    a.sku_id = d.sku_id


-- 合并分区表
drop table if exists dev.dev_tmp_lgb_combine_allocation_Forecast;

set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
CREATE TABLE IF Not EXISTS dev.dev_tmp_lgb_combine_allocation_Forecast
(
    rdc_id  string,
    fdc_id  string,
    sku_id  string,
    is_whitelist    int,
    plan_allocation_qtty    double,
    actual_allocation_qtty  double,
    forecast_daily_override_sales   array<double>,
    forecast_sales_mean     double,
    sale        double,
    sale_all    double,
    stock_qtty  double)
    PARTITIONED by (date_s  string);
insert OVERWRITE table dev.dev_tmp_lgb_combine_allocation_Forecast  partition(date_s)
select
    a.rdc_id,
    a.fdc_id,
    a.sku_id,
    a.is_whitelist,
    a.plan_allocation_qtty,
    a.actual_allocation_qtty,
    d.forecast_daily_override_sales,
    d.forecast_sales_mean,
    d.sale,
    d.sale_all,
    d.stock_qtty,
    a.date_s
from
    (
        select
            a.rdc_id,
            a.fdc_id,
            a.sku_id,
            a.is_whitelist,
            a.plan_allocation_qtty,
            a.actual_allocation_qtty,
            a.date_s
        from
            dev.dev_tmp_lgb_allocation_qttys a
        where
            a.fdc_id = 605  AND
            a.rdc_id = 4    AND
            a.date_s = '2016-12-20'
    ) a
left join
    (
        select
            b.sku_id,
            b.forecast_daily_override_sales,
            b.forecast_sales_mean,
            b.sale,
            b.sale_all,
            c.stock_qtty
        from
            (
                select
                    m.sku_id,
                    m.forecast_daily_override_sales,
                    m.forecast_sales_mean,
                    m.sale,
                    n.sale_all
                from
                    (
                        -- 取20号的数据
                        select
                            wid as sku_id,
                            forecast_daily_override_sales,
                            (forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])/7  as forecast_sales_mean,
                            total_sales  as sale
                        from
                            dev.dev_tmp_lgb_saleForecast
                        where
                            fdcid = 605 and
                            dt = '2016-12-20'
                    )   m
                left join
                    (
                        -- 取平均的销量
                        select
                            wid as sku_id,
                            avg(total_sales)   as sale_all
                        from
                            dev.dev_tmp_lgb_saleForecast
                        where
                            fdcid = 605     and
                            dt >= '2016-12-20'
                        group by
                            wid
                    )   n
                on
                    m.sku_id = n.sku_id
            )  b
        left join
            (
                -- 取inv的前一天数据
                select
                    wid as sku_id,
                    stock_qtty
                from
                    dev.dev_tmp_lgb_saleForecast
                where
                    fdcid = 605 and
                    dt = '2016-12-19'
            )  c
        on
            b.sku_id = c.sku_id
    ) d
on
    a.sku_id = d.sku_id


-- 3、计算s-S
select
    sku_id,
    stock_qtty,
    plan_allocation_qtty,
    forecast_sales_mean,
    ceil((stock_qtty + plan_allocation_qtty) / forecast_sales_mean) as plan_S,
    ceil((stock_qtty + actual_allocation_qtty) / forecast_sales_mean) as actual_S,
    ceil(stock_qtty / forecast_sales_mean) as s,
    sale,
    sale_all  as sale_avg,
    date_s
from
    dev.dev_tmp_lgb_combine_allocation_Forecast
where
    date_s = '2016-12-20'  and
    plan_allocation_qtty  > 0
-- 将数据导出，使用 python 画出分布图


-- 查询有S-s为负数的问题
SELECT
    *
FROM
    dev.dev_tmp_lgb_combine_allocation_Forecast
WHERE
    sku_id IN ('1140618', '1205323', '1859302', '2832688', '3206126', '3220844', '3466659', '3921082', '885157', '911618', '915879')


-- 最后一张大表
create table dev.dev_tmp_lgb_combine_allocation_Forecast_price as
select
    a.*,
    b.stk_prc as price
from
    (
        select
            *
        from
            dev.dev_tmp_lgb_combine_allocation_Forecast
    )   a
left join
    (
        select
            sku_id,
            stk_prc
        from
            gdm.gdm_m03_item_sku_price_da
        where
            type = 1 AND
            dt = sysdate(-1)
    )   b
on
    a.sku_id = b.sku_id
