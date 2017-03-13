-- 看前1000个
SELECT
	*
FROM
	dev.dev_diaobo_simulation_bp_mid03
LIMIT
	1000



-- 描述
DESC dev.dev_diaobo_simulation_bp_mid03



-- 按照abs_gap绝对的gap排序
DROP TABLE if EXISTS dev.dev_diaobo_simulation_bp_mid03_qtty_gap;

CREATE TABLE dev.dev_diaobo_simulation_bp_mid03_qtty_gap AS
SELECT
	a.sku_id,
   avg(a.abs_gap)  as avg_abs_qtty_gap,
   avg(a.gap)  as avg_qtty_gap,
   avg(a.gap_rate)  as avg_qtty_gap_rate
FROM
    (
        SELECT
           b.sku_id,
           b.dt,
           b.bp_quantity,
           b.actual_allocation_qtty,
           abs(b.bp_quantity - b.actual_allocation_qtty) as abs_gap,
        	 b.bp_quantity - b.actual_allocation_qtty  as gap,
        	 (b.bp_quantity - b.actual_allocation_qtty)/ b.actual_allocation_qtty  as gap_rate
        FROM
        	 (
            SELECT
               sku_id,
               dt,
               CASE WHEN bp_quantity is NULL THEN 0 ELSE bp_quantity end as bp_quantity,
               CASE WHEN actual_allocation_qtty is NULL THEN 0 ELSE actual_allocation_qtty end as actual_allocation_qtty
            FROM
                dev.dev_diaobo_simulation_bp_mid03
            ) b
        WHERE
            dt > '2016-12-01'
    ) a
GROUP BY
	a.sku_id
ORDER BY
	avg_abs_qtty_gap   DESC



-- 查询
SELECT
	*
FROM
	dev.dev_diaobo_simulation_bp_mid03_qtty_gap
