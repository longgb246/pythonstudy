-- 建立日期交叉表
create table dev.dev_inv_opt_simulation_sku_list as
select
    f.dt,
    g.fdcid,
    g.sku_id
from
    (
        SELECT
            day_string as dt
        FROM
            dev.dev_dim_date
        WHERE
            day_string between '2017-01-01' and '2017-03-20'
    )  f
cross join
    (
        select
            dt,
            wid as sku_id ,
            fdcid
        from
            dev.dev_inv_opt_fdc_sku_daily_summary_mid01
        where
	    	dt = '2017-02-19' and
		    fdcid in ('605', '634', '633')
    ) g