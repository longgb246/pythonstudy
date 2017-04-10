
-- 1、加工表：dev.dev_inv_opt_simulation_sku_list_fdcall 调拨中的 band 表。
select
    dt,
    fdcid,
    sku_id,
    band
from
    dev.dev_inv_opt_simulation_sku_list_fdcall

-- 2、原始表 adm.adm_s03_item_band_region
-- 按照日期 dt 的分区表
-- 该表有：全国、北京、上海、广州、武汉、成都、沈阳、西安 的销售量band、销售金额band。
select
        sku_id,
        item_third_cate_cd,
        org_chengdu_sale_num_band       -- 【成都】加入band信息，三级band
from
        adm.adm_s03_item_band_region
where
        stat_ct_type_cd  = 28
        and stat_type_cd = 3
        and dt           = '2017-02-17'

