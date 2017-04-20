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






select
    dt,
    sku_id,
    fdc_id,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * std1,0)
--        else
--            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std1) end lop1,
--    case when forecast_daily_override_sales is null then coalesce(1.96 *s td2,0)
--        else
--            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std2) end lop2,
    case when forecast_daily_override_sales is null then null
        else
            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std3) end lop3,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * std4,0)
--        else
--            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std4) end lop4,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * std5,0)
--        else
--            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std5) end lop5,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * std6,0)
--        else
--            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std6) end lop6,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * std7,0)
--        else
--            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std7) end lop7,


--    case when forecast_daily_override_sales is null then coalesce(1.96 * sqrt(8/7) * std1,0)
--        else
--            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
--            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
--            1.96 * sqrt(8/7) * std1) end target_qtty1,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * sqrt(8/7) * std2,0)
--        else
--            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
--            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
--            1.96 * sqrt(8/7) * std2) end target_qtty2,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * sqrt(8/7) * std3,0)
--        else
--            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
--            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
--            1.96 * sqrt(8/7) * std3) end target_qtty3,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * sqrt(8/7) * std4,0)
--        else
--            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
--            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
--            1.96 * sqrt(8/7) * std4) end target_qtty4,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * sqrt(8/7) * std5,0)
--        else
--            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
--            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
--            1.96 * sqrt(8/7) * std5) end target_qtty5,
--    case when forecast_daily_override_sales is null then coalesce(1.96 * sqrt(8/7) * std6,0)
--        else
--            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
--            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
--            1.96 * sqrt(8/7) * std6) end target_qtty6,
    case when forecast_daily_override_sales is null then null
        else
            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) +
            1.96 * sqrt(8/7) * std7) as target_qtty7
from
    (
        select
            dt,
            sku_id,
            fdc_id,
            std1,
            std2,
            std3,
            std4,
            std5,
            std6,
            std7
        from
            app.app_ioa_iaa_std
        where
            dt = 'active'
    )  a
join
    (
        select
            dt,
            dc_id  as  fdc_id,
            sku_id,
            forecast_daily_override_sales
        from
            app.app_pf_forecast_result_fdc_di
        where
            dt = 'active'
            and dc_type='1'
    )  b
on
    a.sku_id = b.sku_id
    and a.fdc_id = b.fdc_id




drop table if exists dev.dev_tmp_check_app_ioa_iaa_std;
CREATE table dev.dev_tmp_check_app_ioa_iaa_std as
select
    a.dt,
    a.sku_id,
    a.fdc_id,
    case when forecast_daily_override_sales is null then null
        else
            (b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + 1.96 * std3) end lop3,
    case when forecast_daily_override_sales is null then null
        else
            ((b.forecast_daily_override_sales[0] + b.forecast_daily_override_sales[1] + b.forecast_daily_override_sales[2] + b.forecast_daily_override_sales[3] +
            b.forecast_daily_override_sales[4] + b.forecast_daily_override_sales[5] + b.forecast_daily_override_sales[6]) /7 * 8 +
            1.96 * sqrt(8/7) * std7) end target_qtty7
from
    (
        select
            dt,
            sku_id,
            fdc_id,
            std1,
            std2,
            std3,
            std4,
            std5,
            std6,
            std7
        from
            app.app_ioa_iaa_std
        where
            dt = 'active'
    )  a
join
    (
        select
            dt,
            dc_id  as  fdc_id,
            sku_id,
            forecast_daily_override_sales
        from
            app.app_pf_forecast_result_fdc_di
        where
--            dt = 'active'
            dt = '2017-04-18'
            and dc_type='1'
    )  b
on
    a.sku_id = b.sku_id
    and a.fdc_id = b.fdc_id;
