-- 根据三级品类名称查询三级品类代码
select
    dim_item_gen_third_cate_id,
    dim_item_gen_third_cate_name
from
    dim.dim_item_gen_third_cate
where
    dim_item_gen_third_cate_name like '%米面杂粮%' and
    valid_flag = 1 and
    otc_utc_flag = 1;

-- 查询三级品类最新的上柜SKU数量
select
    item_third_cate_cd,
    min(item_third_cate_name) as item_third_cate_name,
    count(*) as cnt
from
    gdm.gdm_m03_item_sku_da
where
    dt = '2016-12-07' and
    data_type = 1 and
    sku_valid_flag = 1 and
    sku_status_cd = '3001' and
    item_third_cate_cd in ('1655', '13783', '13785', '13781', '13784', '13782', '1661', '11977')
group by
    item_third_cate_cd;

-- 查询三级品类每天上柜SKU数量
select
    dt,
    item_third_cate_cd,
    min(item_third_cate_name) as item_third_cate_name,
    count(distinct sku_id) as cnt
from
    app.app_s09_pv_stock_sum
where
    dt between '2016-11-01' and '2016-12-31' and
    item_third_cate_cd in ('11977') and
    sku_status_cd == 3001
group by
    dt,
    item_third_cate_cd
order by
    dt;
