select
    *,
    row_number() over(partition by wid,war_hou_id,dt order by reple_id desc)as cnt
from
    fdm.fdm_pbs_replenishmentall
where
    dt between '2016-10-01' and '2016-12-31'
    and wpid=7057