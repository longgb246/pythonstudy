-- ==============================================================
-- =                          FDC 调拨量数据                     =
-- ==============================================================

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
    co.dp = "ACTIVE"
    and ck.dp = "ACTIVE"
    and ck.yn in (1, 3, 5)                                                  -- 1---正常，3---删除处理中， 5---删除失败
    and ck.org_from = 4                                                     -- 配出机构 RDC 为 4 的
    and ck.org_to = 605                                                     -- 配入机构 FDC 为 605 的
    and to_date(ck.create_date) = '$this_date'                              -- 日期
group by
    to_date(ck.create_date),
    co.art_no,
    ck.org_to,
    ck.org_from




-- ==============================================================
-- =                          FDC 库存数据                       =
-- ==============================================================
SELECT
    delv_center_num  as  fdc_id,
    sku_id,
    sum(in_transit_qtty) AS fdc_open_po,
    sum(stock_qtty) AS fdc_inv, --库存数量
    dt
FROM
    gdm.gdm_m08_item_stock_day_sum	     	        -- 【主表】商品库存日汇总
WHERE
    dt >= '${start_date}'
    AND dt <= '${end_date}'
    AND delv_center_num = '${dc_id}'
group by
    delv_center_num,
    dt,
    sku_id

