-- ==============================================================
-- =                            白名单表                        =
-- ==============================================================
-- 白名单表 ： fdm.fdm_fdc_whitelist_chain
-- 取出 FDC 的白名单数据

SELECT
    --    modify_time,
    --    create_time,
    wid  as sku_id,
    fdcid   as  fdc_id
FROM
    fdm.fdm_fdc_whitelist_chain 			-- 白名单表
WHERE
    start_date <= '${end_date}'
    AND end_date >= '${end_date}'
    AND yn= 1                               -- 限制条件，有效的白名单
    AND fdcid='${dc_id}'



