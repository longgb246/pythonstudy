-- 获取数据

-- 保持时间的统一："2016-12-03"。
-- 1、FDC白名单
SELECT
    wid  as sku_id,
    fdcid   as  fdc_id
FROM
    fdm.fdm_fdc_whitelist_chain
WHERE
    start_date <= '2016-12-03'
    AND end_date >= '2016-12-03'
    AND yn = 1
-- 2、配送中心信息
SELECT
    dc_id,                          -- 配送中心id
    dc_type,                        -- 类型
    org_dc_id,                      -- 所属配送中心id
    dc_name
FROM
    dim.dim_dc_info
-- 3、RDC的库存
SELECT
    delv_center_num  as rdc_id,             -- 配送中心，RDC
    sku_id,
    sum(stock_qtty) AS inv                  -- 库存数量
FROM
    gdm.gdm_m08_item_stock_day_sum	     	-- 商品库存日汇总
WHERE
    dt = '2016-12-02'
    AND delv_center_num in (3, 4, 5, 6, 9, 10, 316, 682)
group by
    delv_center_num,
    sku_id



-- ================ 合并以上信息建表 ================
drop table if exists dev.tmp_lgb_rdc_inv_diff;
create table dev.tmp_lgb_rdc_inv_diff  as
select
    c.rdc_id,
    c.fdc_id,
    c.sku_id,
    d.inv
from
    (
        select
            a.rdc_id,
            a.fdc_id,
            b.sku_id
        from
            (
                SELECT
                    dc_id  as  fdc_id,                          -- 配送中心id
                    org_dc_id  as  rdc_id                       -- 所属配送中心id
                FROM
                    dim.dim_dc_info
                where
                    org_dc_id  in  (3, 4, 5, 6, 9, 10, 316, 682)        -- 不取 772	德州
                    and dc_type in  (0, 1)           -- 取 RDC 或者 FDC
                    and dc_id  not in  (3, 4, 5, 6, 9, 10, 316, 682)
            )  a
        join
            (
                SELECT
                    wid  as sku_id,
                    fdcid   as  fdc_id
                FROM
                    fdm.fdm_fdc_whitelist_chain
                WHERE
                    start_date <= '2016-12-03'
                    AND end_date >= '2016-12-03'
                    AND yn = 1
            )  b
        on
            a.fdc_id = b.fdc_id
    )  c
join
    (
        SELECT
            delv_center_num  as rdc_id,
            sku_id,
            sum(stock_qtty) AS inv
        FROM
            gdm.gdm_m08_item_stock_day_sum
        WHERE
            dt = '2016-12-02'
            AND delv_center_num in (3, 4, 5, 6, 9, 10, 316, 682)
        group by
            delv_center_num,
            sku_id
    )  d
on
    c.rdc_id = d.rdc_id
    and c.sku_id = d.sku_id



-- ========================= FDC 维度 =========================
drop table if exists dev.tmp_lgb_rdc_inv_diff_count_fdc;
create table dev.tmp_lgb_rdc_inv_diff_count_fdc as
select
    a.rdc_id,
    a.fdc_id,
    c.count_inv_neg0,
    a.count_inv0,
    b.count_inv12
from
    (
        select
            rdc_id,
            fdc_id,
            count(1)  as  count_inv0
        from
            dev.tmp_lgb_rdc_inv_diff
        where
            inv = 0
        group by
            rdc_id,
            fdc_id
    )  a
join
    (
        select
            rdc_id,
            fdc_id,
            count(1)  as  count_inv12
        from
            dev.tmp_lgb_rdc_inv_diff
        where
            inv <= 12  and
            inv > 0
        group by
            rdc_id,
            fdc_id
    )  b
on
    a.rdc_id = b.rdc_id
    and a.fdc_id = b.fdc_id
join
    (
        select
            rdc_id,
            fdc_id,
            count(1)  as  count_inv_neg0
        from
            dev.tmp_lgb_rdc_inv_diff
        where
            inv < 0
        group by
            rdc_id,
            fdc_id
    )  c
on
    a.rdc_id = c.rdc_id
    and a.fdc_id = c.fdc_id
-- 查询
select
    *
from
    dev.tmp_lgb_rdc_inv_diff_count_fdc
order by
    rdc_id,
    fdc_id


-- ========================= RDC 维度 =========================
drop table if exists dev.tmp_lgb_rdc_inv_diff_rdc;
create table dev.tmp_lgb_rdc_inv_diff_rdc as
select distinct
    rdc_id,
    sku_id,
    inv
from
    dev.tmp_lgb_rdc_inv_diff
-- 总的
drop table if exists dev.tmp_lgb_rdc_inv_diff_count_rdc;
create table dev.tmp_lgb_rdc_inv_diff_count_rdc as
select
    a.rdc_id,
    c.count_inv_neg0,
    a.count_inv0,
    b.count_inv12
from
    (
        select
            rdc_id,
            count(1)  as  count_inv0
        from
            dev.tmp_lgb_rdc_inv_diff_rdc
        where
            inv = 0
        group by
            rdc_id
    )  a
join
    (
        select
            rdc_id,
            count(1)  as  count_inv12
        from
            dev.tmp_lgb_rdc_inv_diff_rdc
        where
            inv <= 12  and
            inv > 0
        group by
            rdc_id
    )  b
on
    a.rdc_id = b.rdc_id
join
    (
        select
            rdc_id,
            count(1)  as  count_inv_neg0
        from
            dev.tmp_lgb_rdc_inv_diff_rdc
        where
            inv < 0
        group by
            rdc_id
    )  c
on
    a.rdc_id = c.rdc_id
-- 查询
select
    *
from
    dev.tmp_lgb_rdc_inv_diff_count_rdc
order by
    rdc_id

