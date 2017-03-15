
-- 0-RDC，1-FDC，2-大件中心仓，3-大件卫星仓，10-保税仓 ，11-miniRDC

SELECT
    dc_id,                          -- 配送中心id
    dc_type,                        -- 类型
    org_dc_id,                      -- 所属配送中心id
    dc_name
FROM
    dim.dim_dc_info


