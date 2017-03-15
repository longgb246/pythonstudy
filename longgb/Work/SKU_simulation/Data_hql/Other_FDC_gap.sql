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


-- RDC的库存
SELECT
    delv_center_num  as rdc_id,             -- 配送中心，RDC
    dt,
    sku_id,
    sum(in_transit_qtty) AS open_po,        -- 在途
    sum(stock_qtty) AS inv                  -- 库存数量
FROM
    gdm.gdm_m08_item_stock_day_sum	     	-- 商品库存日汇总
WHERE
    dt = '2016-12-02'
    AND delv_center_num in (3, 4, 5, 6, 9, 10, 316, 682)
group by
    delv_center_num,
    dt,
    sku_id



