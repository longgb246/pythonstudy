-- ==============================================================
-- =                          RDC 库存数据                       =
-- ==============================================================
-- RDC ： 3, 4, 5, 6, 9, 10, 316, 682

-- 1	6	0	6	北京
-- 2	682	0	682	固安
-- 3	3	0	3	上海
-- 4	10	0	10	广州
-- 5	4	0	4	成都
-- 6	5	0	5	武汉
-- 7	9	0	9	沈阳
-- 8	316	0	316	西安
-- 9	772	0	772	德州



-- （1）选 每天、每个rdc、每个sku的在途量和、库存和
SELECT
    delv_center_num  as rdc_id,             -- 配送中心，RDC
    dt,
    sku_id,
    sum(in_transit_qtty) AS open_po,        -- 在途
    sum(stock_qtty) AS inv                  -- 库存数量
FROM
    gdm.gdm_m08_item_stock_day_sum	     	-- 商品库存日汇总              后改为 app.app_sim_act_inventory 数据，来源：旭波。
WHERE
    dt>='${start_date}'
    AND dt<='${end_date}'
    AND delv_center_num='${org_dc_id}'
group by
    delv_center_num,
    dt,
    sku_id


-- ==============================================================
-- =                          RDC 销量数据                       =
-- ==============================================================
create table dev.tmp_gaoyun_sale as
select
    sku_id,
    dc_id  as rdc_id,
    order_date,
    total_sales,                            -- 销量
    dt
from
    app.app_sfs_sales_region
where
    dt >= '${start_date}'
    and dt <= '${end_date}'
    and dc_id in (3, 4, 5, 6, 9, 10, 316, 682)