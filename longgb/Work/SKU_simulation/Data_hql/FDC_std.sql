
-- dc_id = '605'
-- start_date = '2016-07-01'
-- end_date = '2016-11-01'


-- ==================================================================
-- =                         first、建立临时表                       =
-- ==================================================================
-- 1、临时表01 - 建立交叉表
drop table if exists dev.tmp_zjs_fdc_std_diff_01;
create table dev.tmp_zjs_fdc_std_diff_01 as
select
    f.dt,
    g.fdc_id,
    g.sku_id
from
    (
        SELECT
            day_string as dt
        FROM
            dev.dev_dim_date
        WHERE
            day_string between '2016-07-01' and '2016-11-01'
    )  f
cross join
    (
       SELECT
            distinct
            wid  as sku_id,
            fdcid   as  fdc_id
        FROM
            fdm.fdm_fdc_whitelist_chain 			-- 白名单表
        WHERE
            start_date <= '2016-11-01'
            AND end_date > '2016-11-01'
            AND yn = 1                               -- 限制条件，有效的白名单
            AND fdcid = '605'
    ) g;


-- 2、临时表02 - 销量预测
drop table if exists dev.tmp_zjs_fdc_std_diff_02;
create table dev.tmp_zjs_fdc_std_diff_02 as
SELECT
    dt,
    dc_id  as  fdc_id,
    sku_id,
    forecast_daily_override_sales
FROM
    app.app_pf_forecast_result_fdc_di
WHERE
    dt >= '2016-07-01'
    AND dt <= '2016-11-01'
    And dc_id = '605'


-- 3、临时表03 - 真实销量
drop table if exists dev.tmp_zjs_fdc_std_diff_03;
create table dev.tmp_zjs_fdc_std_diff_03 as
select
    dt,
    dc_id  as  fdc_id,
    sku_id,
    total_sales
from
    app.app_sfs_sales_dc
where
    dt >= '2016-07-01'
    AND dt <= '2016-11-01'
    And dc_id = '605'


-- 4、临时表04 - sku上下柜状态
drop table if exists dev.tmp_zjs_fdc_std_diff_04;
create table dev.tmp_zjs_fdc_std_diff_04 as
select
    -- 它有重复记录，可能存在 同一个dt、fdc_id、sku_id，但是库存、sku_status_cd不相同的情况
    dt,
    dim_delv_center_num  as fdc_id,
    sku_id,
    max(sku_status_cd)  as  sku_status_cd
from
    app.app_sfs_vendibility
where
    dt >= '2016-07-01'
    AND dt <= '2016-11-01'
    and dim_delv_center_num = '605'
group by
    dt,
    dim_delv_center_num,
    sku_id


-- 5、临时表05 - 库存
drop table if exists dev.tmp_zjs_fdc_std_diff_05;
create table dev.tmp_zjs_fdc_std_diff_05 as
select
    dt,
    delv_center_num  as fdc_id,
    sku_id,
    sum(stock_qtty + in_transit_qtty + inner_in_qtty - inner_out_qtty - order_transfer_num - no_sale_num - app_booking_num - sale_reserve_qtty) as stock_qtty
from
    app.app_sim_act_inventory
where
    dt >= '2016-07-01'
    AND dt <= '2016-11-01'
    and delv_center_num = '605'
group by
    dt,
    delv_center_num,
    sku_id


-- ==================================================================
-- =                          last、建立总表                         =
-- ==================================================================
drop table if exists dev.tmp_zjs_fdc_std_diff;
create table dev.tmp_zjs_fdc_std_diff as
select
    b.dt,
    a.fdc_id,
    a.sku_id,
    b.forecast_daily_override_sales,
    c.total_sales,
    d.sku_status_cd,
    e.stock_qtty,
    lag(e.stock_qtty,1)over(parititon by a.sku_id order by b.dt) as stock_qtty_start,           -- 取 lag 上一天的数据
    sum(c.total_sales)over(parititon by a.sku_id order by b.dt rows BETWEEN ROWS BETWEEN CURRENT ROW AND FOLLOWING 3) as sales_sum3
from
    (
        select
            dt,
            fdc_id,
            sku_id
        from
            dev.tmp_zjs_fdc_std_diff_01
    ) a
left join
    (
        SELECT
            dt,
            fdc_id,
            sku_id,
            forecast_daily_override_sales
        FROM
            dev.tmp_zjs_fdc_std_diff_02
    ) b
on
    a.dt = b.dt
    and a.fdc_id = b.fdc_id
    and a.sku_id = b.sku_id
left join
    (
        select
            dt,
            fdc_id,
            sku_id,
            total_sales
        from
            dev.tmp_zjs_fdc_std_diff_03
    ) c
on
    a.dt = c.dt
    and a.fdc_id = c.fdc_id
    and a.sku_id = c.sku_id
left join
    (
        select
            dt,
            fdc_id,
            sku_id,
            sku_status_cd
        from
            dev.tmp_zjs_fdc_std_diff_04
    ) d
on
    a.dt = d.dt
    and a.sku_id = d.sku_id
    and a.fdc_id = d.fdc_id
left join
    (
        select
            dt,
            fdc_id,
            sku_id,
            stock_qtty
        from
            dev.tmp_zjs_fdc_std_diff_05
    ) e
on
    a.dt = e.dt
    and a.sku_id = e.sku_id
    and a.fdc_id = e.fdc_id;




