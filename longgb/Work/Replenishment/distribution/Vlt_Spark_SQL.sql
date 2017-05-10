-- dt = current_date - 1   昨天的时间 "%Y-%m-%d"
-- beginYearDate = current_date.year - 1     去年的时间 "%Y-%m-%d"
-- orderCount = 10


-- =============================================================
-- 1、extractPurChasedOrder
-- =============================================================
-- 1.1 sourceOrder
create table dev.lgb_tmp_sourceOrder_52 as 
	select
		item_third_cate_cd,		-- 三级品类
		pur_bill_id,			-- 采购单编号
		sku_id,
		supp_brevity_cd,		-- 供应商简码
		int_org_num,			-- 内部机构编号
		create_tm,				-- 创建时间
		complete_dt				-- 完成日期
	from 
		gdm.gdm_m04_pur_det_basic_sum
	where
		dt='2017-05-01'
		and valid_flag='1' 
		AND cgdetail_yn=1 
		AND create_tm >= '2016-05-02' 
		and create_tm <= '2017-05-01'
		AND complete_dt != '' 
		AND pur_bill_src_cd IN (2, 3, 4, 10)		-- 自动下单
[ DROP TABLE IF EXISTS dev.lgb_tmp_sourceOrder_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_sourceOrder_53; ]
[ create table dev.lgb_tmp_sourceOrder_52 as select item_third_cate_cd, pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from gdm.gdm_m04_pur_det_basic_sum where dt='2017-05-01'  and valid_flag='1'  AND cgdetail_yn=1  AND create_tm >= '2016-05-02' and create_tm <='2017-05-01' AND complete_dt !=''  AND pur_bill_src_cd IN (2, 3, 4, 10); ]
[ create table dev.lgb_tmp_sourceOrder_53 as select item_third_cate_cd, pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from gdm.gdm_m04_pur_det_basic_sum where dt='2017-05-02'  and valid_flag='1'  AND cgdetail_yn=1  AND create_tm >= '2016-05-03' and create_tm <='2017-05-02' AND complete_dt !=''  AND pur_bill_src_cd IN (2, 3, 4, 10); ]

-- 1.2 autoPoOrder
create table dev.lgb_tmp_autoPoOrder_52 as 
	select
		item_third_cate_cd,
		pur_bill_id,
		sku_id,
		supp_brevity_cd,
		int_org_num,
		create_tm,
		complete_dt
	from
		gdm.gdm_m04_pur_det_basic_sum
	where
		dt='2017-05-01'
		and valid_flag='1' 
		AND cgdetail_yn=1 
		AND create_tm >= '2016-05-02' 
		and create_tm <= '2017-05-01'
		AND complete_dt !='' 
		AND pur_bill_src_cd = 15					-- 人工下单
[ DROP TABLE IF EXISTS dev.lgb_tmp_autoPoOrder_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_autoPoOrder_53; ]
[ create table dev.lgb_tmp_autoPoOrder_52 as  select item_third_cate_cd, pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from gdm.gdm_m04_pur_det_basic_sum where dt='2017-05-01' and valid_flag='1'  AND cgdetail_yn=1  AND create_tm >='2016-05-02' and create_tm <='2017-05-01' AND complete_dt !=''  AND pur_bill_src_cd = 15; ]
[ create table dev.lgb_tmp_autoPoOrder_53 as  select item_third_cate_cd, pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from gdm.gdm_m04_pur_det_basic_sum where dt='2017-05-02' and valid_flag='1'  AND cgdetail_yn=1  AND create_tm >='2016-05-03' and create_tm <='2017-05-02' AND complete_dt !=''  AND pur_bill_src_cd = 15; ]

-- 1.3 orders
create table dev.lgb_tmp_orders_52 as 		-- 合并到一起
	select
		*
	from
		dev.lgb_tmp_sourceOrder_52 a
	union all
	select
		*
	from
		dev.lgb_tmp_autoPoOrder_52 b
[ DROP TABLE IF EXISTS dev.lgb_tmp_orders_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_orders_53; ]
[ create table dev.lgb_tmp_orders_52 as select * from dev.lgb_tmp_sourceOrder_52 a union all select * from dev.lgb_tmp_autoPoOrder_52 b; ]
[ create table dev.lgb_tmp_orders_53 as select * from dev.lgb_tmp_sourceOrder_53 a union all select * from dev.lgb_tmp_autoPoOrder_53 b; ]

-- =============================================================
-- 2、removeNeipei
-- =============================================================
create table dev.lgb_tmp_unqiueOrder_52 as  -- 仅仅要采购单编号去重
	select distinct
		pur_bill_id
	from	
		dev.orders
[ DROP TABLE IF EXISTS dev.lgb_tmp_unqiueOrder_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_unqiueOrder_53; ]
[ create table dev.lgb_tmp_unqiueOrder_52 as select distinct pur_bill_id from dev.lgb_tmp_orders_52; ]		
[ create table dev.lgb_tmp_unqiueOrder_53 as select distinct pur_bill_id from dev.lgb_tmp_orders_53; ]		

create table dev.lgb_tmp_fdm_scm_cgfenpei_chain_52 as 
	select
		rfid as pur_bill_id,
		idcompany				-- 新机构(配送中心)
	from
		dev.lgb_tmp_unqiueOrder_52 a
	join 
		fdm.fdm_scm_cgfenpei_chain b
	on 
		a.pur_bill_id = b.rfid
[ DROP TABLE IF EXISTS dev.lgb_tmp_fdm_scm_cgfenpei_chain_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_fdm_scm_cgfenpei_chain_53; ]		
[ create table dev.lgb_tmp_fdm_scm_cgfenpei_chain_52 as select rfid as pur_bill_id, idcompany from dev.lgb_tmp_unqiueOrder_52 a join  fdm.fdm_scm_cgfenpei_chain b on a.pur_bill_id = b.rfid; ]		
[ create table dev.lgb_tmp_fdm_scm_cgfenpei_chain_53 as select rfid as pur_bill_id, idcompany from dev.lgb_tmp_unqiueOrder_53 a join  fdm.fdm_scm_cgfenpei_chain b on a.pur_bill_id = b.rfid; ]		

create table dev.lgb_tmp_fdm_scm_cgtable_chain_52 as 
	select
		id as pur_bill_id,
		idcompany				-- 新机构(配送中心)
	from
		dev.lgb_tmp_unqiueOrder_52 a
	join 
		fdm.fdm_scm_cgtable_chain b
	on 
		a.pur_bill_id = b.id
[ DROP TABLE IF EXISTS dev.lgb_tmp_fdm_scm_cgtable_chain_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_fdm_scm_cgtable_chain_53; ]		
[ create table dev.lgb_tmp_fdm_scm_cgtable_chain_52 as select id as pur_bill_id, idcompany from dev.lgb_tmp_unqiueOrder_52 a join fdm.fdm_scm_cgtable_chain b on a.pur_bill_id = b.id; ]
[ create table dev.lgb_tmp_fdm_scm_cgtable_chain_53 as select id as pur_bill_id, idcompany from dev.lgb_tmp_unqiueOrder_53 a join fdm.fdm_scm_cgtable_chain b on a.pur_bill_id = b.id; ]

-- 【saveAsTable】 lgb_neipeiOrder_1
create table dev.lgb_tmp_neipeiOrder_52 as
	select distinct
		a.pur_bill_id
	from
		dev.lgb_tmp_fdm_scm_cgfenpei_chain_52 a
	join
		dev.lgb_tmp_fdm_scm_cgtable_chain_52 b
	on
		a.pur_bill_id = b.pur_bill_id
	where
		a.idcompany != b.idcompany
[ DROP TABLE IF EXISTS dev.lgb_tmp_neipeiOrder_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_neipeiOrder_53; ]	
[ create table dev.lgb_tmp_neipeiOrder_52 as select distinct a.pur_bill_id from dev.lgb_tmp_fdm_scm_cgfenpei_chain_52 a join dev.lgb_tmp_fdm_scm_cgtable_chain_52 b on a.pur_bill_id = b.pur_bill_id where a.idcompany != b.idcompany; ]
[ create table dev.lgb_tmp_neipeiOrder_53 as select distinct a.pur_bill_id from dev.lgb_tmp_fdm_scm_cgfenpei_chain_53 a join dev.lgb_tmp_fdm_scm_cgtable_chain_53 b on a.pur_bill_id = b.pur_bill_id where a.idcompany != b.idcompany; ]

-- sourceOrder
create table dev.lgb_tmp_sourceOrder_52_2 as
select 
	item_third_cate_cd,
	a.pur_bill_id,
	sku_id,
	supp_brevity_cd,
	int_org_num,
	create_tm,
	complete_dt
from
	dev.lgb_tmp_sourceOrder_52 a
left join
	dev.lgb_tmp_neipeiOrder_52 b
on
	a.pur_bill_id = b.pur_bill_id
where
	b.pur_bill_id is Null
[ DROP TABLE IF EXISTS dev.lgb_tmp_sourceOrder_52_2; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_sourceOrder_53_2; ]
[ create table dev.lgb_tmp_sourceOrder_52_2 as select  item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_sourceOrder_52 a left join dev.lgb_tmp_neipeiOrder_52 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null; ]
[ create table dev.lgb_tmp_sourceOrder_53_2 as select  item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_sourceOrder_53 a left join dev.lgb_tmp_neipeiOrder_53 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null; ]

-- autoPoOrder
create table dev.lgb_tmp_autoPoOrder_52_2 as
	select 
		item_third_cate_cd,
		a.pur_bill_id,
		sku_id,
		supp_brevity_cd,
		int_org_num,
		create_tm,
		complete_dt
	from
		dev.lgb_tmp_autoPoOrder_52 a 
	left join
		dev.lgb_tmp_neipeiOrder_52 b
	on
		a.pur_bill_id = b.pur_bill_id
	where
		b.pur_bill_id is Null
[ DROP TABLE IF EXISTS dev.lgb_tmp_autoPoOrder_52_2; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_autoPoOrder_53_2; ]
[ create table dev.lgb_tmp_autoPoOrder_52_2 as select item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_autoPoOrder_52 a  left join dev.lgb_tmp_neipeiOrder_52 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null; ]
[ create table dev.lgb_tmp_autoPoOrder_53_2 as select item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_autoPoOrder_53 a  left join dev.lgb_tmp_neipeiOrder_53 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null; ]

-- orders
create table dev.lgb_tmp_orders_52_2 as
	select 
		item_third_cate_cd,
		a.pur_bill_id,
		sku_id,
		supp_brevity_cd,
		int_org_num,
		create_tm,
		complete_dt
	from
		dev.lgb_tmp_orders_52 a
	left join
		dev.lgb_tmp_neipeiOrder_52 b
	on
		a.pur_bill_id = b.pur_bill_id
	where
		b.pur_bill_id is Null
[ DROP TABLE IF EXISTS dev.lgb_tmp_orders_52_2; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_orders_53_2; ]
[ create table dev.lgb_tmp_orders_52_2 as select  item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_orders_52 a left join dev.lgb_tmp_neipeiOrder_52 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null; ]
[ create table dev.lgb_tmp_orders_53_2 as select  item_third_cate_cd, a.pur_bill_id, sku_id, supp_brevity_cd, int_org_num, create_tm, complete_dt from dev.lgb_tmp_orders_53 a left join dev.lgb_tmp_neipeiOrder_53 b on a.pur_bill_id = b.pur_bill_id where b.pur_bill_id is Null; ]

-- =============================================================
-- 3、extract_into_wareHouse
-- =============================================================
create table dev.lgb_tmp_into_wareHouse_52 as 
	select
		pur_bill_id, 
		sku_id,
		min(into_wh_tm) as into_wh_tm			-- 入库时间
	from
		gdm.gdm_m04_pur_recv_det_basic_sum
	where
		dt='2017-05-01'
		AND cgdetail_yn=1 
		AND into_wh_qtty>0 
	group by
		pur_bill_id, 
		sku_id
[ DROP TABLE IF EXISTS dev.lgb_tmp_into_wareHouse_52; ]
[ DROP TABLE IF EXISTS dev.lgb_tmp_into_wareHouse_53; ]
[ create table dev.lgb_tmp_into_wareHouse_52 as select pur_bill_id, sku_id, min(into_wh_tm) as into_wh_tm from gdm.gdm_m04_pur_recv_det_basic_sum where dt='2017-05-01' AND cgdetail_yn=1 AND into_wh_qtty>0 group by pur_bill_id, sku_id; ]
[ create table dev.lgb_tmp_into_wareHouse_53 as select pur_bill_id, sku_id, min(into_wh_tm) as into_wh_tm from gdm.gdm_m04_pur_recv_det_basic_sum where dt='2017-05-02' AND cgdetail_yn=1 AND into_wh_qtty>0 group by pur_bill_id, sku_id; ]

create table dev.lgb_tmp_into_wareHouse_52_2 as 
	select
		*
	from
		dev.lgb_tmp_orders_52_2 a
	join
		dev.lgb_tmp_into_wareHouse_52 b
	on
		a.pur_bill_id = b.pur_bill_id
		AND	a.sku_id = b.sku_id
->[ DROP TABLE IF EXISTS dev.lgb_tmp_into_wareHouse_52_2; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_into_wareHouse_53_2; ]
->[ create table dev.lgb_tmp_into_wareHouse_52_2 as  select * from dev.lgb_tmp_orders_52_2 a join dev.lgb_tmp_into_wareHouse_52 b on a.pur_bill_id = b.pur_bill_id AND	a.sku_id = b.sku_id;]
->[ create table dev.lgb_tmp_into_wareHouse_53_2 as  select * from dev.lgb_tmp_orders_53_2 a join dev.lgb_tmp_into_wareHouse_53 b on a.pur_bill_id = b.pur_bill_id AND	a.sku_id = b.sku_id;]

-- =============================================================
-- 4、extract_t_6
-- =============================================================
create table dev.lgb_tmp_t_6_Old_52 as 
	select
		po_id,									-- 采购单号
		max(create_time) as t_6
	from		
		fdm.fdm_procurement_po_process
	where
		po_yn=1 
		AND po_state=6 
		AND process_desc LIKE '%采购单提交成功，启动审核工作流%' 
	group by
		po_id				
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_Old_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_Old_53; ]
->[ create table dev.lgb_tmp_t_6_Old_52 as select po_id, max(create_time) as t_6 from fdm.fdm_procurement_po_process where po_yn=1 AND po_state=6 AND process_desc LIKE '%采购单提交成功，启动审核工作流%' group by po_id;]
->[ create table dev.lgb_tmp_t_6_Old_53 as select po_id, max(create_time) as t_6 from fdm.fdm_procurement_po_process where po_yn=1 AND po_state=6 AND process_desc LIKE '%采购单提交成功，启动审核工作流%' group by po_id;]

create table dev.lgb_tmp_t_6_New_52 as 
	select
		poid,									-- 采购单号
		max(createtime) as t_6
	from		
		fdm.fdm_procurement_lifecycle_chain
	where
		dt='4712-12-31' 
		and actiontype=104 
		and yn=1
	group by
		poid
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_New_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_New_53; ]
->[ create table dev.lgb_tmp_t_6_New_52 as select poid, max(createtime) as t_6 from fdm.fdm_procurement_lifecycle_chain where dt='4712-12-31' and actiontype=104 and yn=1 group by poid;]
->[ create table dev.lgb_tmp_t_6_New_53 as select poid, max(createtime) as t_6 from fdm.fdm_procurement_lifecycle_chain where dt='4712-12-31' and actiontype=104 and yn=1 group by poid;]

create table dev.lgb_tmp_t_6_52 as
	select
		*
	from
		dev.lgb_tmp_t_6_Old_52 a
	union all
	select
		*
	from
		dev.lgb_tmp_t_6_New_52 b
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_New_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_New_53; ]
->[ create table dev.lgb_tmp_t_6_52 as select * from dev.lgb_tmp_t_6_Old_52 a union all select * from dev.lgb_tmp_t_6_New_52 b;]
->[ create table dev.lgb_tmp_t_6_53 as select * from dev.lgb_tmp_t_6_Old_53 a union all select * from dev.lgb_tmp_t_6_New_53 b;]


create table dev.lgb_tmp_sourcePo_52 as
	select distinct
		pur_bill_id
	from
		lgb_tmp_sourceOrder_52
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sourcePo_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sourcePo_53; ]
->[ create table dev.lgb_tmp_sourcePo_52 as select distinct pur_bill_id from lgb_tmp_sourceOrder_52;]
->[ create table dev.lgb_tmp_sourcePo_53 as select distinct pur_bill_id from lgb_tmp_sourceOrder_53;]

create table dev.lgb_tmp_t_6_source_52 as 
	select
		a.pur_bill_id,
		t_6
	from
		dev.lgb_tmp_sourcePo_52 a
	join
		dev.lgb_tmp_t_6_52 b
	on
		a.pur_bill_id = b.po_id
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_source_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_source_53; ]
->[ create table dev.lgb_tmp_t_6_source_52 as  select a.pur_bill_id, t_6 from dev.lgb_tmp_sourcePo_52 a join dev.lgb_tmp_t_6_52 b on a.pur_bill_id = b.po_id;]
->[ create table dev.lgb_tmp_t_6_source_53 as  select a.pur_bill_id, t_6 from dev.lgb_tmp_sourcePo_53 a join dev.lgb_tmp_t_6_53 b on a.pur_bill_id = b.po_id;]

create table dev.lgb_tmp_t_6_autoPo_52 as 
	select	
		pur_bill_id,
		max(create_tm) as t_6
	from
		dev.lgb_tmp_autoPoOrder_52_2 
	group by 
		pur_bill_id
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_autoPo_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t_6_autoPo_53; ]
->[ create table dev.lgb_tmp_t_6_autoPo_52 as select pur_bill_id, max(create_tm) as t_6 from dev.lgb_tmp_autoPoOrder_52_2 group by pur_bill_id;]
->[ create table dev.lgb_tmp_t_6_autoPo_53 as select pur_bill_id, max(create_tm) as t_6 from dev.lgb_tmp_autoPoOrder_53_2 group by pur_bill_id;]
->-- 最大的疑问！！！！这个地方是否有问题！！！！


-- dev.t6
create table dev.lgb_tmp_t6_52 as
	select
		pur_bill_id,
		max(t_6) as t_6
	from
		dev.lgb_tmp_t_6_source_52
	group by 
		pur_bill_id
	union all
	select
		pur_bill_id,
		max(t_6) as t_6
	from
		dev.lgb_tmp_t_6_autoPo_52
	group by 
		pur_bill_id
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t6_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_t6_53; ]
->[ create table dev.lgb_tmp_t6_52 as select pur_bill_id, max(t_6) as t_6 from dev.lgb_tmp_t_6_source_52 group by  pur_bill_id union all select pur_bill_id, max(t_6) as t_6 from dev.lgb_tmp_t_6_autoPo_52 group by  pur_bill_id;]
->[ create table dev.lgb_tmp_t6_53 as select pur_bill_id, max(t_6) as t_6 from dev.lgb_tmp_t_6_source_53 group by  pur_bill_id union all select pur_bill_id, max(t_6) as t_6 from dev.lgb_tmp_t_6_autoPo_53 group by  pur_bill_id;]

-- =============================================================
-- 5、extract_combine_Data
-- =============================================================
-- 【saveAsTable】lgb_vlt_jobCombinedData_1
create table dev.lgb_tmp_combinedData_52 as
	select
		*
	from
		dev.lgb_tmp_into_wareHouse_52_2 a
	join 
		dev.lgb_tmp_t6_52 b
	on
		a.pur_bill_id = b.pur_bill_id
->[ DROP TABLE IF EXISTS dev.lgb_tmp_combinedData_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_combinedData_53; ]
->[ create table dev.lgb_tmp_combinedData_52 as select * from dev.lgb_tmp_into_wareHouse_52_2 a join  dev.lgb_tmp_t6_52 b on a.pur_bill_id = b.pur_bill_id;]
->[ create table dev.lgb_tmp_combinedData_53 as select * from dev.lgb_tmp_into_wareHouse_53_2 a join  dev.lgb_tmp_t6_53 b on a.pur_bill_id = b.pur_bill_id;]

create table dev.lgb_tmp_combinedData_52_2 as 
	select
	 	c.pur_bill_id,
	 	c.sku_id,
	 	c.item_third_cate_cd,
	 	c.supp_brevity_cd,
	 	c.int_org_num,
	 	c.into_wh_tm,
	 	c.t_6,
	 	c.round(vlt) as vlt
	 from
	 	(
	 		select
				pur_bill_id,
				sku_id,
				item_third_cate_cd,
				supp_brevity_cd,
				int_org_num,
				into_wh_tm,
				t_6,
				round((unix_timestamp(into_wh_tm)-unix_timestamp(t_6))/86400.0,2) as vlt
			from
				dev.lgb_tmp_combinedData_52
			where
				vlt >= 0.5 
				and vlt <= 60 
	 	) c
->[ DROP TABLE IF EXISTS dev.lgb_tmp_combinedData_52_2; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_combinedData_53_2; ]	
->[ create table dev.lgb_tmp_combinedData_52_2 as select c.pur_bill_id, c.sku_id, c.item_third_cate_cd, c.supp_brevity_cd, c.int_org_num, c.into_wh_tm, c.t_6, c.round(vlt) as vlt from ( select pur_bill_id, sku_id, item_third_cate_cd, supp_brevity_cd, int_org_num, into_wh_tm, t_6, round((unix_timestamp(into_wh_tm)-unix_timestamp(t_6))/86400.0,2) as vlt from dev.lgb_tmp_combinedData_52 where vlt >= 0.5 and vlt <= 60 ) c;]
->[ create table dev.lgb_tmp_combinedData_53_2 as select c.pur_bill_id, c.sku_id, c.item_third_cate_cd, c.supp_brevity_cd, c.int_org_num, c.into_wh_tm, c.t_6, c.round(vlt) as vlt from ( select pur_bill_id, sku_id, item_third_cate_cd, supp_brevity_cd, int_org_num, into_wh_tm, t_6, round((unix_timestamp(into_wh_tm)-unix_timestamp(t_6))/86400.0,2) as vlt from dev.lgb_tmp_combinedData_53 where vlt >= 0.5 and vlt <= 60 ) c;]

-- =============================================================
-- 6、transform_data
-- =============================================================
create table dev.lgb_tmp_sku_slice_vlt_count_52 as 
	select
		sku_id,
		supp_brevity_cd,
		int_org_num,
		vlt,
		count(distinct pur_bill_id, sku_id) as sku_vlt_OrderCount
	from
		dev.lgb_tmp_combinedData_52_2
	group by
		sku_id,
		supp_brevity_cd,
		int_org_num,
		vlt
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sku_slice_vlt_count_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sku_slice_vlt_count_53; ]	
->[ create table dev.lgb_tmp_sku_slice_vlt_count_52 as  select sku_id, supp_brevity_cd, int_org_num, vlt, count(distinct pur_bill_id, sku_id) as sku_vlt_OrderCount from dev.lgb_tmp_combinedData_52_2 group by sku_id, supp_brevity_cd, int_org_num, vlt;]
->[ create table dev.lgb_tmp_sku_slice_vlt_count_53 as  select sku_id, supp_brevity_cd, int_org_num, vlt, count(distinct pur_bill_id, sku_id) as sku_vlt_OrderCount from dev.lgb_tmp_combinedData_53_2 group by sku_id, supp_brevity_cd, int_org_num, vlt;]	

create table dev.lgb_tmp_sku_slice_count_52 as
	select
		sku_id,
		supp_brevity_cd,
		int_org_num,
		max(item_third_cate_cd) as item_third_cate_cd,
		count(Distinct pur_bill_id, sku_id) as sku_Ordercount,
		round(mean(vlt),2) as sku_vlt_mean,
		round(stddev(vlt),2) as sku_vlt_stdev
	from
		dev.lgb_tmp_combinedData_52_2
	group by
		sku_id,
		supp_brevity_cd,
		int_org_num
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sku_slice_count_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sku_slice_count_53; ]	
->[ create table dev.lgb_tmp_sku_slice_count_52 as select sku_id, supp_brevity_cd, int_org_num, max(item_third_cate_cd) as item_third_cate_cd, count(Distinct pur_bill_id, sku_id) as sku_Ordercount, round(mean(vlt),2) as sku_vlt_mean, round(stddev(vlt),2) as sku_vlt_stdev from dev.lgb_tmp_combinedData_52_2 group by sku_id, supp_brevity_cd, int_org_num;]
->[ create table dev.lgb_tmp_sku_slice_count_53 as select sku_id, supp_brevity_cd, int_org_num, max(item_third_cate_cd) as item_third_cate_cd, count(Distinct pur_bill_id, sku_id) as sku_Ordercount, round(mean(vlt),2) as sku_vlt_mean, round(stddev(vlt),2) as sku_vlt_stdev from dev.lgb_tmp_combinedData_53_2 group by sku_id, supp_brevity_cd, int_org_num;]

create table dev.lgb_tmp_skuData_52 as
	select
		sku_id,
		supp_brevity_cd,
		int_org_num,
		item_third_cate_cd,
		vlt,
		sku_vlt_OrderCount,
		sku_Ordercount,
		round(sku_vlt_OrderCount*1.0/sku_Ordercount,2) as sku_vlt_prob,
		sku_vlt_mean,
		sku_vlt_stdev
	from
		dev.lgb_tmp_sku_slice_vlt_count_52
	join
		dev.lgb_tmp_sku_slice_count_52
	on
		sku_id,
		supp_brevity_cd,
		int_org_num
->[ DROP TABLE IF EXISTS dev.lgb_tmp_skuData_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_skuData_53; ]	
->[ create table dev.lgb_tmp_skuData_52 as select sku_id, supp_brevity_cd, int_org_num, item_third_cate_cd, vlt, sku_vlt_OrderCount, sku_Ordercount, round(sku_vlt_OrderCount*1.0/sku_Ordercount,2) as sku_vlt_prob, sku_vlt_mean, sku_vlt_stdev from dev.lgb_tmp_sku_slice_vlt_count_52 join dev.lgb_tmp_sku_slice_count_52 on sku_id, supp_brevity_cd, int_org_num;]
->[ create table dev.lgb_tmp_skuData_53 as select sku_id, supp_brevity_cd, int_org_num, item_third_cate_cd, vlt, sku_vlt_OrderCount, sku_Ordercount, round(sku_vlt_OrderCount*1.0/sku_Ordercount,2) as sku_vlt_prob, sku_vlt_mean, sku_vlt_stdev from dev.lgb_tmp_sku_slice_vlt_count_53 join dev.lgb_tmp_sku_slice_count_53 on sku_id, supp_brevity_cd, int_org_num;]

-- 【saveAsTable】lgb_sku_result_1
create table dev.lgb_tmp_sku_result_52 as
	select
		sku_id,
		supp_brevity_cd,
		int_org_num,
		max(item_third_cate_cd) as item_third_cate_cd,
		collect_set(concat(cast(vlt as string), cast(sku_vlt_prob as string), ":")) as sku_vlt_dist,
		max(sku_Ordercount) as sku_Ordercount,
		max(sku_vlt_mean) as sku_vlt_mean,
		max(sku_vlt_stdev) as sku_vlt_stdev
	from
		dev.lgb_tmp_skuData_52
	group by
		sku_id,
		supp_brevity_cd,
		int_org_num
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sku_result_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_sku_result_53; ]			
->[ create table dev.lgb_tmp_sku_result_52 as select sku_id, supp_brevity_cd, int_org_num, max(item_third_cate_cd) as item_third_cate_cd, collect_set(concat(cast(vlt as string), cast(sku_vlt_prob as string), ":")) as sku_vlt_dist, max(sku_Ordercount) as sku_Ordercount, max(sku_vlt_mean) as sku_vlt_mean, max(sku_vlt_stdev) as sku_vlt_stdev from dev.lgb_tmp_skuData_52 group by sku_id, supp_brevity_cd, int_org_num;]
->[ create table dev.lgb_tmp_sku_result_53 as select sku_id, supp_brevity_cd, int_org_num, max(item_third_cate_cd) as item_third_cate_cd, collect_set(concat(cast(vlt as string), cast(sku_vlt_prob as string), ":")) as sku_vlt_dist, max(sku_Ordercount) as sku_Ordercount, max(sku_vlt_mean) as sku_vlt_mean, max(sku_vlt_stdev) as sku_vlt_stdev from dev.lgb_tmp_skuData_53 group by sku_id, supp_brevity_cd, int_org_num;]

create table dev.lgb_tmp_order_lt_10_52 as
	select
		supp_brevity_cd,
		int_org_num,
		item_third_cate_cd
	from
		dev.lgb_tmp_sku_slice_count_52
	where
		sku_Ordercount< 10
->[ DROP TABLE IF EXISTS dev.lgb_tmp_order_lt_10_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_order_lt_10_53; ]	
->[ create table dev.lgb_tmp_order_lt_10_52 as select supp_brevity_cd, int_org_num, item_third_cate_cd from dev.lgb_tmp_sku_slice_count_52 where sku_Ordercount< 10;]
->[ create table dev.lgb_tmp_order_lt_10_52 as select supp_brevity_cd, int_org_num, item_third_cate_cd from dev.lgb_tmp_sku_slice_count_52 where sku_Ordercount< 10;]

create table dev.lgb_tmp_cid3_slice_vlt_count_52 as
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		vlt,
		count(Distinct pur_bill_id, sku_id) as cid3_vlt_Count
	from
		dev.lgb_tmp_combinedData_52_2
	group by
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		vlt
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_slice_vlt_count_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_slice_vlt_count_53; ]	
->[ create table dev.lgb_tmp_cid3_slice_vlt_count_52 as select item_third_cate_cd, supp_brevity_cd, int_org_num, vlt, count(Distinct pur_bill_id, sku_id) as cid3_vlt_Count from dev.lgb_tmp_combinedData_52_2 group by item_third_cate_cd, supp_brevity_cd, int_org_num, vlt;]
->[ create table dev.lgb_tmp_cid3_slice_vlt_count_53 as select item_third_cate_cd, supp_brevity_cd, int_org_num, vlt, count(Distinct pur_bill_id, sku_id) as cid3_vlt_Count from dev.lgb_tmp_combinedData_53_2 group by item_third_cate_cd, supp_brevity_cd, int_org_num, vlt;]

create table dev.lgb_tmp_cid3_slice_count_52 as
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		count(Distinct pur_bill_id, sku_id) as cid3_Ordercount,
		round(mean(vlt), 2) as cid3_vlt_mean,
		round(stddev("vlt"), 2) as cid3_vlt_stdev
	from
		dev.lgb_tmp_combinedData_52_2
	group by
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_slice_count_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_slice_count_53; ]	
->[ create table dev.lgb_tmp_cid3_slice_count_52 as select item_third_cate_cd, supp_brevity_cd, int_org_num, count(Distinct pur_bill_id, sku_id) as cid3_Ordercount, round(mean(vlt), 2) as cid3_vlt_mean, round(stddev("vlt"), 2) as cid3_vlt_stdev from dev.lgb_tmp_combinedData_52_2 group by item_third_cate_cd, supp_brevity_cd, int_org_num;]
->[ create table dev.lgb_tmp_cid3_slice_count_53 as select item_third_cate_cd, supp_brevity_cd, int_org_num, count(Distinct pur_bill_id, sku_id) as cid3_Ordercount, round(mean(vlt), 2) as cid3_vlt_mean, round(stddev("vlt"), 2) as cid3_vlt_stdev from dev.lgb_tmp_combinedData_53_2 group by item_third_cate_cd, supp_brevity_cd, int_org_num;]

create table dev.lgb_tmp_cid3_data_52 as
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		vlt,
		cid3_vlt_Count,
		cid3_Ordercount,
		round(cid3_vlt_Count*1.0/cid3_Ordercount,2) as cid3_vlt_prob,
		cid3_vlt_mean,
		cid3_vlt_stdev
	from
		dev.lgb_tmp_cid3_slice_vlt_count_52
	join
		dev.lgb_tmp_cid3_slice_count_52
	on
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_data_52; ]	
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_data_53; ]	
[ create table dev.lgb_tmp_cid3_data_52 as select item_third_cate_cd, supp_brevity_cd, int_org_num, vlt, cid3_vlt_Count, cid3_Ordercount, round(cid3_vlt_Count*1.0/cid3_Ordercount,2) as cid3_vlt_prob, cid3_vlt_mean, cid3_vlt_stdev from dev.lgb_tmp_cid3_slice_vlt_count_52 join dev.lgb_tmp_cid3_slice_count_52 on item_third_cate_cd, supp_brevity_cd, int_org_num;]
[ create table dev.lgb_tmp_cid3_data_53 as select item_third_cate_cd, supp_brevity_cd, int_org_num, vlt, cid3_vlt_Count, cid3_Ordercount, round(cid3_vlt_Count*1.0/cid3_Ordercount,2) as cid3_vlt_prob, cid3_vlt_mean, cid3_vlt_stdev from dev.lgb_tmp_cid3_slice_vlt_count_53 join dev.lgb_tmp_cid3_slice_count_53 on item_third_cate_cd, supp_brevity_cd, int_org_num;]


-- 【saveAsTable】lgb_cid3_result_1
create table dev.lgb_tmp_cid3_result_52 as 
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		collect_set(concat(cast(vlt as string), cast(cid3_vlt_prob as string), ":")) as cid3_vlt_dist,
		max(cid3_Ordercount) as cid3_Ordercount,
		max(cid3_vlt_mean) as cid3_vlt_mean,
		max(cid3_vlt_stdev) as cid3_vlt_stdev
	from
		dev.lgb_tmp_order_lt_10_52 a
	join
		dev.lgb_tmp_cid3_data_52 b
	on
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num
	group by 
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_result_52; ]
->[ DROP TABLE IF EXISTS dev.lgb_tmp_cid3_result_53; ]

-- =============================================================
-- 7、save
-- =============================================================
--【按分区 insertInto】app.app_vlt_distribution   lgb_result_1
create table dev.result as
	select
		[dt] as dt,
		item_third_cate_cd,
        supp_brevity_cd,
        int_org_num,
        sku_id,
        sku_vlt_dist,
        sku_Ordercount,
        sku_vlt_mean,
        sku_vlt_stdev,
        case when a.sku_Ordercount<10 then b.cid3_vlt_dist else a.sku_vlt_dist end dist,
        case when a.sku_Ordercount<10 then b.cid3_Ordercount else a.sku_Ordercount end Ordercount,
        case when a.sku_Ordercount<10 then b.cid3_vlt_mean else a.sku_vlt_mean end vlt_mean,
        case when a.sku_Ordercount<10 then b.cid3_vlt_stdev else a.sku_vlt_stdev end vlt_stdev
	from
		dev.sku_result a
	left join
		dev.cid3_result b
	on
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num







-- 中间表
-- 1、lgb_neipeiOrder_2
-- 2、lgb_vlt_jobCombinedData_2
-- 3、lgb_sku_result_2
-- 4、lgb_cid3_result_2
-- 5、lgb_result_2
