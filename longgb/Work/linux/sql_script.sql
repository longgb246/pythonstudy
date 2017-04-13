-- 1、简单验证
-- easy_sql.sql
select
    count(*)
from
    gdm.gdm_m03_item_sku_da
where
    dt = '$this_date'


-- sql_script_2.sql
-- 为了验证 sku 都是唯一的，没有多余的。
-- 【结论】初步验证是没有重复的！
select
    sku_count,
    count(sku_count) as all_sku_count
from
    (
        select
            item_sku_id,
            count(item_sku_id) as sku_count
        from
            gdm.gdm_m03_item_sku_da
        where
            dt = '$this_date'
        group by
            item_sku_id
    ) a
group by
    sku_count


-- 复杂验证
-- complex.sql
-- 假设没有多余的，那么。
select
    a.item_sku_id,
    a.dt1,
    b.dt2
from
    (
        select
            dt as dt1,
            item_sku_id
        from
            gdm.gdm_m03_item_sku_da
        where
            dt = '$this_date'
    )  a
left join
    (
        select
            dt as dt2,
            item_sku_id
        from
            gdm.gdm_m03_item_sku_da
        where
            dt = '$next_date'
    )  b
on
    a.item_sku_id = b.item_sku_id
where
    b.dt2 is null

