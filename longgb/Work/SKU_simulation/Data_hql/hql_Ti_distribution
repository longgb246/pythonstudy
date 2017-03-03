-- 1、选取sku和价格
select
    sku_id,
    stk_prc
from
    gdm.gdm_m03_item_sku_price_da
where
    type = 1 AND
    dt = sysdate(-1)



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

