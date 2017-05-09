-- dt = current_date - 1   昨天的时间 "%Y-%m-%d"
-- beginYearDate = current_date.year - 1     去年的时间 "%Y-%m-%d"
-- orderCount = 10

select

from 
	gdm.gdm_m04_pur_det_basic_sum
where
	dt=dt 
	and valid_flag='1' 
	AND cgdetail_yn=1 
	AND create_tm BETWEEN 


