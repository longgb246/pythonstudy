-- 创建训练数据中新品列表，选出有用的字段
drop table if exists dev.dev_lzt_train_new_sku_list;
create table dev.dev_lzt_train_new_sku_list as
select
    item_sku_id_hashed,
    main_sku_id_hashed,
    sku_name,
    brand_codes,            -- 原表中这个字段拼错了
    barndname_full,
    item_third_cate_cd,
    item_third_cate_name,
    shelves_dt,             -- 上架时间
    otc_tm,                 -- 上柜时间
    item_type,              -- 商品型号
    size,                   -- 尺寸
    size_rem,               -- 尺寸备注
    size_seq,               -- 尺寸次序
    len,
    width,
    height,
    calc_volume,
    wt,                     -- 重量
    colour
from
    dev.dev_lzt_train_sku_basic_info
where
    shelves_dt >='2015-01-01';


SELECT count(*) FROM dev.dev_lzt_train_new_sku_list;  -- 32924个


-- 给sku的首次上柜且可售日打标，方便后续计算首次上柜且可售后7天内可售天的平均销量
drop table if EXISTS dev.dev_lzt_train_new_sku_daily;
create table dev.dev_lzt_train_new_sku_daily as
select
    d.*,
    ROW_NUMBER() OVER(PARTITION BY d.item_sku_id_hashed, d.shelves_dt ORDER BY d.dt) as rnk
from
    (
        select
            a.*,
            b.dt,
            b.allow_flag,               -- 上柜且可售
            b.sku_status_cd,            -- 上柜状态
            b.allow_reserve_flag,       -- 可售
            b.stock_qtty,
            case when c.total_sales is not null then c.total_sales else 0 end as total_sales
        FROM
            dev.dev_lzt_train_new_sku_list a
        left JOIN
            (
                SELECT
                    *,
                    case when (sku_status_cd = 1 AND (allow_reserve_flag = 1 or stock_qtty >0)) THEN 1 else 0 end as allow_flag
                FROM
                    dev.dev_lzt_train_sku_vendibility_daily
            ) b
        ON
            a.item_sku_id_hashed = b.item_sku_id_hashed
        left join
            (
                select
                    item_sku_id_hashed,
                    dt,
                    total_sales
                from
                    dev.dev_lzt_train_sku_sales_daily
            ) c
        on
            b.item_sku_id_hashed = c.item_sku_id_hashed
            and b.dt = c.dt
        where
            b.sku_status_cd = 1
            and b.allow_flag = 1  -- 选出上柜且可售天，实际上可售就够了
    ) d;


-- 计算首次上柜可售后7天内可售天的平均销量
drop table if EXISTS dev.dev_lzt_train_new_sku_daily_avg_sales;
create table dev.dev_lzt_train_new_sku_daily_avg_sales as
select
    b.item_sku_id_hashed,
    b.shelves_dt,
    sum(b.allow_flag) as total_keshou_days,
    sum(b.total_sales) as total_7days_sales,
    sum(b.total_sales)/sum(b.allow_flag) as avg_keshou_sales,
    sum(case when b.total_sales >0 then 1 else 0 end) as sales_beyond0_days
from
    (
        select
            *
        from
            dev.dev_lzt_train_new_sku_daily
        where
            rnk = 1
    ) a  -- 选出第一个上柜天(之前是选出rnk<=7的，没达目的,因为这个daily表已经是按照上柜=1筛选过了)
join
    (
        select
            *
        from
            dev.dev_lzt_train_new_sku_daily
    ) b
on
    a.item_sku_id_hashed = b.item_sku_id_hashed
where
   b.dt <= date_add(a.dt,6)
  -- and allow_flag = 1
group by
    b.item_sku_id_hashed,
    b.shelves_dt;

