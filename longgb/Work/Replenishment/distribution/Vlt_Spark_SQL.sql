-- dt = current_date - 1   昨天的时间 "%Y-%m-%d"
-- beginYearDate = current_date.year - 1     去年的时间 "%Y-%m-%d"
-- orderCount = 10


-- =============================================================
-- 1、extractPurChasedOrder
-- =============================================================
-- 1.1 sourceOrder
create table dev.sourceOrder as 
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
		dt=dt 
		and valid_flag='1' 
		AND cgdetail_yn=1 
		AND create_tm BETWEEN beginYearDate and dt
		AND complete_dt <>'' 
		AND pur_bill_src_cd IN (2, 3, 4, 10)		-- 自动下单

-- 1.2 autoPoOrder
create table dev.autoPoOrder as 
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
		dt=dt 
		and valid_flag='1' 
		AND cgdetail_yn=1 
		AND create_tm BETWEEN beginYearDate AND dt
		AND complete_dt <>'' 
		AND pur_bill_src_cd = 15					-- 人工下单

-- 1.3 orders
create table dev.orders as 		-- 合并到一起
	select
		dev.sourceOrder
	unionall
		dev.autoPoOrder


-- =============================================================
-- 2、removeNeipei
-- =============================================================
create table dev.unqiueOrder as  -- 仅仅要采购单编号去重
	select distinct
		pur_bill_id
	from	
		dev.orders

create table dev.fdm_scm_cgfenpei_chain as 
	select
		b.rfid as pur_bill_id,
		idcompany				-- 新机构(配送中心)
	from
		dev.unqiueOrder a
	join 
		fdm.fdm_scm_cgfenpei_chain b
	on 
		a.pur_bill_id = b.pur_bill_id

create table dev.fdm_scm_cgtable_chain as 
	select
		id as pur_bill_id,
		idcompany				-- 新机构(配送中心)
	from
		dev.unqiueOrder a
	join 
		fdm.fdm_scm_cgtable_chain b
	on 
		a.pur_bill_id = b.pur_bill_id

-- 【saveAsTable】
create table dev.neipeiOrder as
	select
		a.pur_bill_id
	from
		dev.fdm_scm_cgfenpei_chain a
	join
		dev.fdm_scm_cgtable_chain b
	on
		a.pur_bill_id = b.pur_bill_id
		AND a.idcompany <> b.idcompany

-- sourceOrder
select 
	item_third_cate_cd,
	a.pur_bill_id,
	sku_id,
	supp_brevity_cd,
	int_org_num,
	create_tm,
	complete_dt
from
	dev.sourceOrder a
left join
	dev.neipeiOrder b
on
	a.pur_bill_id = b.pur_bill_id
where
	b.pur_bill_id is Null

-- autoPoOrder
select 
	item_third_cate_cd,
	a.pur_bill_id,
	sku_id,
	supp_brevity_cd,
	int_org_num,
	create_tm,
	complete_dt
from
	dev.autoPoOrder a
left join
	dev.neipeiOrder b
on
	a.pur_bill_id = b.pur_bill_id
where
	b.pur_bill_id is Null

-- orders
select 
	item_third_cate_cd,
	a.pur_bill_id,
	sku_id,
	supp_brevity_cd,
	int_org_num,
	create_tm,
	complete_dt
from
	dev.orders a
left join
	dev.neipeiOrder b
on
	a.pur_bill_id = b.pur_bill_id
where
	b.pur_bill_id is Null


-- =============================================================
-- 3、extract_into_wareHouse
-- =============================================================
create table dev.into_wareHouse as 
	select
		pur_bill_id, 
		sku_id,
		min(into_wh_tm) as into_wh_tm			-- 入库时间
	from
		gdm.gdm_m04_pur_recv_det_basic_sum
	where
		dt=dt
		AND cgdetail_yn=1 
		AND into_wh_qtty>0 
	group by
		pur_bill_id, 
		sku_id

create table dev.into_wareHouse as 
	select
		*
	from
		dev.orders a
	join
		dev.into_wareHouse b
	on
		a.pur_bill_id = b.pur_bill_id
		AND	a.sku_id = b.sku_id


-- =============================================================
-- 4、extract_t_6
-- =============================================================
create table dev.t_6_Old as 
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

create table dev.t_6_New as 
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

create table dev.t_6 as
	select
		dev.t_6_Old
	unionall
		dev.t_6_New

create table dev.t_6_source as 
	select
		a.pur_bill_id,
		t_6
	from
		dev.sourcePo a
	join
		dev.t_6 b
	on
		a.pur_bill_id = b.po_id

create table dev.t_6_autoPo as 
	select	
		pur_bill_id,
		max(create_tm) as t_6
	from
		dev.autoPoOrder 
	group by 
		pur_bill_id

-- dev.t6
select
	pur_bill_id,
	max(t_6) as t_6
from
	dev.t_6_source
unionall
	dev.t_6_autoPo
group by 
	pur_bill_id


-- =============================================================
-- 5、extract_combine_Data
-- =============================================================
-- 【saveAsTable】vlt_jobCombinedData
create table dev.combinedData as
	select
		*
	from
		dev.into_wareHouse a
	join 
		dev.t6 b
	on
		a.pur_bill_id = b.pur_bill_id


create table dev.combinedData as 
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
				dev.vlt_jobCombinedData
			where
				vlt between 0.5 and 60 
	 	) c
	

-- =============================================================
-- 6、transform_data
-- =============================================================
create table dev.sku_slice_vlt_count as 
	select
		sku_id,
		supp_brevity_cd,
		int_org_num,
		vlt,
		count(distinct pur_bill_id, sku_id) as sku_vlt_OrderCount
	from
		dev.combinedData
	group by
		sku_id,
		supp_brevity_cd,
		int_org_num,
		vlt

create table dev.sku_slice_count as
	select
		sku_id,
		supp_brevity_cd,
		int_org_num,
		max(item_third_cate_cd) as item_third_cate_cd,
		count(Distinct pur_bill_id, sku_id) as sku_Ordercount,
		round(mean(vlt),2) as sku_vlt_mean,
		round(stddev(vlt),2) as sku_vlt_stdev
	from
		dev.combinedData
	group by
		sku_id,
		supp_brevity_cd,
		int_org_num

create table dev.skuData as
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
		dev.sku_slice_vlt_count
	join
		dev.sku_slice_count
	on
		sku_id,
		supp_brevity_cd,
		int_org_num

-- 【saveAsTable】sku_result
create table dev.sku_result as
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
		dev.skuData
	group by
		sku_id,
		supp_brevity_cd,
		int_org_num

create table dev.order_lt_10 as
	select
		supp_brevity_cd,
		int_org_num,
		item_third_cate_cd
	from
		dev.sku_slice_count
	where
		sku_Ordercount< 10

create table dev.cid3_slice_vlt_count as
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		vlt,
		count(Distinct pur_bill_id, sku_id) as cid3_vlt_Count
	from
		dev.combinedData
	group by
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		vlt

create table dev.cid3_slice_count as
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		count(Distinct pur_bill_id, sku_id) as cid3_Ordercount,
		round(mean(vlt), 2) as cid3_vlt_mean,
		round(stddev("vlt"), 2) as cid3_vlt_stdev
	from
		dev.combinedData
	group by
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num

create table dev.cid3_data as
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
		dev.cid3_slice_vlt_count
	join
		dev.cid3_slice_count
	on
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num

-- 【saveAsTable】cid3_result
create table dev.cid3_result as 
	select
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num,
		collect_set(concat(cast(vlt as string), cast(cid3_vlt_prob as string), ":")) as cid3_vlt_dist,
		max(cid3_Ordercount) as cid3_Ordercount,
		max(cid3_vlt_mean) as cid3_vlt_mean,
		max(cid3_vlt_stdev) as cid3_vlt_stdev
	from
		dev.order_lt_10
	join
		dev.cid3_data
	on
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num
	group by 
		item_third_cate_cd,
		supp_brevity_cd,
		int_org_num


-- =============================================================
-- 7、save
-- =============================================================
--【按分区 insertInto】app.app_vlt_distribution
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


