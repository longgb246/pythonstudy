

dev.dev_inv_opt_simulation_sku_list_select

select
    *
from
    dev.dev_inv_opt_simulation_sku_list_select
limit
    1000



select
    dt,
    fdcid,
    sku_id,
    band
from
    dev.dev_inv_opt_simulation_sku_list_select






select
    count(dt) as count,
    fdcid,
    sku_id,
    band
from
    dev.dev_inv_opt_simulation_sku_list_select
group by
    fdcid,
    sku_id,
    band
order by
    count(dt) desc



dev.dev_inv_opt_fdc_sku_allocation_data


-- 1、取出 SKU
select DISTINCT
    SKU_id
from
    dev.dev_inv_opt_fdc_sku_allocation_data
where
    rdc_id is not null AND
    (plan_num_auto + delivered_num_manual)>0

-- 2、将取出的 SKU 与原来的进行相交

select
    a.band,
    a.fdcid,
    --发生了调拨
    sum(case when b.sku_id is not null then 1 else 0 end) allocation_sku,--发生了调拨的sku
    sum(case when b.sku_id is not null and a.sku_id is not null  then 1 else 0 end) allocation_whitelist_sku,--发生了调拨且在白名单，
    --白名单数量
    sum(case when a.sku_id is not null then 1 else 0 end) whitelist, --白名单数量
    sum(case when a.sku_id is not null and b.sku_id is null then 1 else 0 end) white_no_alloca,--在白名单里面未发生调拨
    count(1)
from
    (
        select
            *
        from
            dev.dev_inv_opt_simulation_sku_list_select
        where
            dt = '2017-02-19'
    )  a
full join
    (
        select DISTINCT
            SKU_id  as sku_id
        from
            dev.dev_inv_opt_fdc_sku_allocation_data
        where
            rdc_id is not null AND
            (plan_num_auto + delivered_num_manual)>0  and
            dt > '2017-01-18' AND
            dt < '2017-02-19'
    )  b
on
    a.sku_id = b.sku_id
group by
    a.band,
    a.fdcid


