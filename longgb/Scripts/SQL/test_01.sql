drop table if exists dev.dev_lgb_test_stddev;
create table dev.dev_lgb_test_stddev
(
    sku_id  string,
    dt      date,
    sales   double
)
row format delimited
fields terminated by '\t'
;
load data local inpath 'testData.txt' into table dev.dev_lgb_test_stddev;


select
    sku_id,
    dt,
    stddev_pop(sales) as std_sales
from
    dev.dev_lgb_test_stddev
group by
    sku_id,
    dt;


-- 【测试 1】sum() over(partition by __ order by __ rows between current row and __ following)
-- 测试结果：sum(sales) over(partition by sku_id order by dt rows between current row and 6 following) as sales_all
-- 使用上面的不会出现结果为空的情况。


-- 【测试 2】stddev_pop()
-- 测试结果：当有空值的时候，直接不把空值拉入计算，拿所有的非空值进行计算。


