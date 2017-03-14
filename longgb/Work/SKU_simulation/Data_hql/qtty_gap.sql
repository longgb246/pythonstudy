-- 按照abs_gap绝对的gap排序
DROP TABLE if EXISTS dev.dev_diaobo_simulation_bp_mid03_qtty_gap;
CREATE TABLE dev.dev_diaobo_simulation_bp_mid03_qtty_gap AS
SELECT
    a.sku_id,
    avg(a.abs_gap)  as avg_abs_qtty_gap,
    avg(a.gap)  as avg_qtty_gap,
    avg(a.gap_rate)  as avg_qtty_gap_rate
FROM
    (
        SELECT
            b.sku_id,
            b.dt,
            b.bp_quantity ,
            b.actual_allocation_qtty,
            abs(b.bp_quantity - b.actual_allocation_qtty) as abs_gap,
            b.bp_quantity - b.actual_allocation_qtty  as gap,
            (b.bp_quantity - b.actual_allocation_qtty)/ b.actual_allocation_qtty  as gap_rate
        FROM
        	 (
                SELECT
                    sku_id,
                    dt,
                    CASE WHEN bp_quantity is NULL THEN 0 ELSE bp_quantity end as bp_quantity,
                    CASE WHEN actual_allocation_qtty is NULL THEN 0 ELSE actual_allocation_qtty end as actual_allocation_qtty
                FROM
                    dev.dev_diaobo_simulation_bp_mid03
            ) b
        WHERE
            dt > '2016-12-01'
    ) a
GROUP BY
	a.sku_id
ORDER BY
	avg_abs_qtty_gap   DESC



-- ==============================================================
-- =                        继续昨天的东西                       =
-- ==============================================================
-- ------------------------------ 建立04表 ------------------------------
create table dev.dev_diaobo_simulation_bp_mid04 as
SELECT
    b.sku_id,
    b.dt,
    b.bp_quantity,
    b.actual_allocation_qtty,
    abs(b.bp_quantity - b.actual_allocation_qtty) as abs_gap,
    b.bp_quantity - b.actual_allocation_qtty  as gap,
    (b.bp_quantity - b.actual_allocation_qtty)/ b.actual_allocation_qtty  as gap_rate
FROM
     (
        SELECT
            sku_id,
            dt,
            CASE WHEN bp_quantity is NULL THEN 0 ELSE bp_quantity end as bp_quantity,
            CASE WHEN actual_allocation_qtty is NULL THEN 0 ELSE actual_allocation_qtty end as actual_allocation_qtty
        FROM
            dev.dev_diaobo_simulation_bp_mid03
    ) b
WHERE
    dt > '2016-12-01'


-- ------------------------------ 建立05表 ------------------------------
-- 三级品类的不参与补货
create table dev.dev_diaobo_simulation_bp_mid05
as
select
    a.*,
    case when b.item_third_cate_cd is null then  1 else 0 end as   third_cate_cd_include,
    case when c.include is null then 1 else 0 end as   sku_include
from
    dev.dev_diaobo_simulation_bp_mid04  a
left join
    (
        select
            d.sku_id,
            d.item_third_cate_cd
        from
            (
                select
                    item_sku_id  as sku_id ,
                    item_third_cate_cd
                from
                    gdm.gdm_m03_item_sku_da
                WHERE
                    dt= sysdate( - 1)
             ) d
        inner join
            (
                select
                    wpid3
                from
                    fdm.fdm_fdc_auto_calc_category_chain
                WHERE
                    fdcid = '605'
                    and dp='ACTIVE'
                    and include=0
            ) e
        on
            d.item_third_cate_cd = e.wpid3
    )  b
on
    a.sku_id = b.sku_id
left join
    (
        select
            item_code  as sku_id,
            include
        from
            dev.dev_diaobo_simulation_bp_mid04_01
        where
            include = '否'
    )  c
on
    a.sku_id = c.sku_id


-- 建表
drop table dev.dev_diaobo_simulation_bp_mid04_01;
create table dev.dev_diaobo_simulation_bp_mid04_01
(
    busi_type       string,
    com_name        string,
    item_code       string,
    item_name       string,
    include         string,
    maintian        string,
    maintian_date   string
)
row format delimited fields terminated by '\t';
-- 三级品类---》商品宽表----》sku_id
-- 禁止调拨的三级品类：fdm_fdc_auto_calc_category_chain
-- 前台会过滤掉一级品类为空的三级品类
-- 禁止调拨的SKU：fdm_fdc_auto_calc_ware_chain（有问题）


-- 1、在 actual_allocation_qtty=0 和 bp_quantity>0 的情况下。
select
    round(bp_quantity/10),
    count(DISTINCT sku_id),
    sum(bp_quantity)
from
    dev.dev_diaobo_simulation_bp_mid05
where
    dt = '2016-12-03'
    and sku_include=1
    and third_cate_cd_include=1
    and bp_quantity>0
    and actual_allocation_qtty=0
group by
    round(bp_quantity/10)


-- 2、在 bp_quantity=0 和 actual_allocation_qtty>0 的情况下。
select
    round(actual_allocation_qtty/10),
    count(DISTINCT sku_id),
    sum(actual_allocation_qtty)
from
    dev.dev_diaobo_simulation_bp_mid05
where
    dt = '2016-12-03'
    and sku_include=1
    and third_cate_cd_include=1
    and bp_quantity=0
    and actual_allocation_qtty>0
group by
    round(actual_allocation_qtty/10)


-- 3、在 bp_quantity>0 和 actual_allocation_qtty>0 的情况下。
select
    round((bp_quantity - actual_allocation_qtty)/10),
    count(DISTINCT sku_id),
    sum(bp_quantity - actual_allocation_qtty)
from
    dev.dev_diaobo_simulation_bp_mid05
where
    dt = '2016-12-03'
    and sku_include=1
    and third_cate_cd_include=1
    and bp_quantity>0
    and actual_allocation_qtty>0
group by
    round((bp_quantity - actual_allocation_qtty)/10)



-- ==============================================================
-- =                        分析原因                             =
-- ==============================================================
-- 查找极值
select
    sku_id,
    round(bp_quantity/10)
from
    dev.dev_diaobo_simulation_bp_mid05
where
    dt = '2016-12-03'
    and sku_include=1
    and third_cate_cd_include=1
    and bp_quantity>0
    and actual_allocation_qtty=0
    and round(bp_quantity/10) >= 100
-- 档位如下：
-- 4007944 : 124
-- 329867  : 103
SELECT
    delv_center_num as rdc_id,
    dt,
    sku_id,
    sum(in_transit_qtty) AS open_po,
    sum(stock_qtty) AS inv
FROM
    gdm.gdm_m08_item_stock_day_sum
WHERE
    dt = '2016-12-03'  and
    delv_center_num in ('4', '605')  and
    sku_id in (4007944, 329867)
group by
    delv_center_num,
    dt,
    sku_id
-- 发现极值点的影响为库存为空，将范围放大，如下：



-- ============================== 分析第一类 ==============================
-- actual_allocation_qtty = 0, bp_quantity>0
select
    a.*,
    b.level_bp,
    b.actual_allocation_qtty,
    c.safestock,
    c.maxstock,
    d.forecast_daily_override_sales,
    d.forecast_sales_mean
from
    (
        SELECT
            delv_center_num as rdc_id,
            dt,
            sku_id,
            sum(in_transit_qtty) AS open_po,
            sum(stock_qtty) AS inv
        FROM
            gdm.gdm_m08_item_stock_day_sum
        WHERE
            dt = '2016-12-02'  and
            delv_center_num in ('4', '605')
            )
        group by
            delv_center_num,
            dt,
            sku_id
    ) a
join
    (
        select
            sku_id
        from
            dev.dev_diaobo_simulation_bp_mid05
        where
            dt = '2016-12-03'
            and sku_include=1
            and third_cate_cd_include=1
            and bp_quantity>0
            and actual_allocation_qtty=0
            and round(bp_quantity/10) >= 20
    ) b
on
    a.sku_id = b.sku_id
left join
    (
        select
            sku_id   ,
            safestock,
            maxstock
        from
            dev.dev_inv_opt_simulation_data_pre_mid02
        where
            dt = '2016-12-03'
    ) c
on
    b.sku_id = c.sku_id
left join
    (
        select
            sku_id,
            dc_id,
            forecast_daily_override_sales,
            (forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])/7  as forecast_sales_mean
        from
            app.app_pf_forecast_result_fdc_di
        where
            dt = '2016-12-03'  and
            dc_id = '605'
    ) d
on
    b.sku_id = d.sku_id and
    a.rdc_id = d.dc_id




-- ============================== 分析第二类 ==============================
-- actual_allocation_qtty > 0, bp_quantity = 0
select
    a.*,
    b.level_bp,
    b.actual_allocation_qtty,
    c.safestock,
    c.maxstock,
    d.forecast_daily_override_sales,
    d.forecast_sales_mean
from
    (
        SELECT
            delv_center_num as rdc_id,
            dt,
            sku_id,
            sum(in_transit_qtty) AS open_po,
            sum(stock_qtty) AS inv
        FROM
            gdm.gdm_m08_item_stock_day_sum
        WHERE
            dt = '2016-12-02'  and
            delv_center_num in ('4', '605')
        group by
            delv_center_num,
            dt,
            sku_id
    ) a
join
    (
        select
            sku_id,
            round(actual_allocation_qtty/10) as level_bp,
            actual_allocation_qtty
        from
            dev.dev_diaobo_simulation_bp_mid05
        where
            dt = '2016-12-03'
            and sku_include=1
            and third_cate_cd_include=1
            and bp_quantity=0
            and actual_allocation_qtty>0
            and round(actual_allocation_qtty/10) >= 10
    ) b
on
    a.sku_id = b.sku_id
left join
    (
        select
            sku_id   ,
            safestock,
            maxstock
        from
            dev.dev_inv_opt_simulation_data_pre_mid02
        where
            dt = '2016-12-03'
    ) c
on
    b.sku_id = c.sku_id
left join
    (
        select
            sku_id,
            dc_id,
            forecast_daily_override_sales,
            (forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])/7  as forecast_sales_mean
        from
            app.app_pf_forecast_result_fdc_di
        where
            dt = '2016-12-03'  and
            dc_id = '605'
    ) d
on
    b.sku_id = d.sku_id and
    a.rdc_id = d.dc_id


-- ============================== 分析第三类 ==============================
-- actual_allocation_qtty > 0, bp_quantity > 0
select
    a.*,
    b.level_bp,
    b.bp_quantity,
    b.actual_allocation_qtty,
    c.safestock,
    c.maxstock,
    d.forecast_daily_override_sales,
    d.forecast_sales_mean
from
    (
        SELECT
            delv_center_num as rdc_id,
            dt,
            sku_id,
            sum(in_transit_qtty) AS open_po,
            sum(stock_qtty) AS inv
        FROM
            gdm.gdm_m08_item_stock_day_sum
        WHERE
            dt = '2016-12-02'  and
            delv_center_num in ('4', '605')
        group by
            delv_center_num,
            dt,
            sku_id
    ) a
join
    (
        select
            sku_id,
            round((bp_quantity - actual_allocation_qtty)/10) as level_bp,
            bp_quantity,
            actual_allocation_qtty
        from
            dev.dev_diaobo_simulation_bp_mid05
        where
            dt = '2016-12-03'
            and sku_include=1
            and third_cate_cd_include=1
            and bp_quantity>0
            and actual_allocation_qtty>0
            and round((bp_quantity - actual_allocation_qtty)/10) >= 8
    ) b
on
    a.sku_id = b.sku_id
left join
    (
        select
            sku_id   ,
            safestock,
            maxstock
        from
            dev.dev_inv_opt_simulation_data_pre_mid02
        where
            dt = '2016-12-03'
    ) c
on
    b.sku_id = c.sku_id
left join
    (
        select
            sku_id,
            dc_id,
            forecast_daily_override_sales,
            (forecast_daily_override_sales[0]+forecast_daily_override_sales[1]+forecast_daily_override_sales[2]+forecast_daily_override_sales[3]+forecast_daily_override_sales[4]+forecast_daily_override_sales[5]+forecast_daily_override_sales[6])/7  as forecast_sales_mean
        from
            app.app_pf_forecast_result_fdc_di
        where
            dt = '2016-12-03' and
            dc_id = '605'
    ) d
on
    b.sku_id = d.sku_id  and
    a.rdc_id = d.dc_id



-- 真实的sS。
select
    sku_id   ,
    safestock,
    maxstock
from
    dev.dev_inv_opt_simulation_data_pre_mid02
where
    dt = '2016-12-03'
-- 拿销量预测的
select
    sku_id,
    forecast_daily_override_sales
from
    app.app_pf_forecast_result_fdc_di
where
    dt = '2016-12-03'

