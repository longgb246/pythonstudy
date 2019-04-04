-- upload

-- 1.train_sku_basic_info.csv
drop table if exists dev.dev_lzt_train_sku_basic_info;
create table dev.dev_lzt_train_sku_basic_info
(
sku_name string,
item_name string,
item_desc string,
data_type int,
brand_codes string,
barndname_en string,
barndname_cn string,
barndname_full string,
item_origin string,
qgp int,
sku_valid_flag int,
item_valid_flag int,
sku_status_cd int,
item_status_cd string,
item_first_cate_cd int,
item_first_cate_name string,
item_second_cate_cd int,
item_second_cate_name string,
item_third_cate_cd int,
item_third_cate_name string,
shelves_tm string,
shelves_dt string,
otc_tm string,
utc_tm string,
support_cash_on_deliver_flag string,
vender_direct_delv_flag string,
slogan string,
sale_qtty_lim string,
first_into_wh_tm string,
item_type string,
size string,
size_rem string,
size_seq string,
len string,
width string,
height string,
calc_volume string,
wt string,
colour string,
pac_propt string,
pac_spec string,
free_goods_flag string,
item_sku_id_hashed string,
main_sku_id_hashed string
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' TBLPROPERTIES ("skip.header.line.count"="1");
LOAD DATA LOCAL INPATH 'train_sku_basic_info.csv' OVERWRITE INTO TABLE dev.dev_lzt_train_sku_basic_info;
alter table dev.dev_lzt_train_sku_basic_info set serdeproperties ('serialization.encoding'='GBK');

-- 2.train_sku_flow_city_daily.csv  这个表好像有点问题。先不管
drop table if exists dev.dev_lzt_train_sku_flow_city_daily;
create table dev.dev_lzt_train_sku_flow_city_daily
(
dt string,
subd_num int,
province_id	city_id int,
item_first_cate_id int,
item_second_cate_id int,
item_third_cate_id int,
pv int,
upv int,
avg_page_rt float,
bounces int,
item_sku_id_hashed string,
test1 string
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' TBLPROPERTIES ("skip.header.line.count"="1");
LOAD DATA LOCAL INPATH 'train_sku_flow_city_daily.csv' OVERWRITE INTO TABLE dev.dev_lzt_train_sku_flow_city_daily;
alter table dev.dev_lzt_train_sku_flow_city_daily set serdeproperties ('serialization.encoding'='GBK');

-- 3.train_sku_price_daily
drop table if exists dev.dev_lzt_train_sku_price_daily;
create table dev.dev_lzt_train_sku_price_daily
(
xxx_prc float,
mkt_prc float,
dt string,
item_sku_id_hashed string
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','  TBLPROPERTIES ("skip.header.line.count"="1");
LOAD DATA LOCAL INPATH 'train_sku_price_daily.csv' OVERWRITE INTO TABLE dev.dev_lzt_train_sku_price_daily;
alter table dev.dev_lzt_train_sku_price_daily set serdeproperties ('serialization.encoding'='GBK');

-- 4.train_sku_sales_daily.csv
drop table if exists dev.dev_lzt_train_sku_sales_daily;
create table dev.dev_lzt_train_sku_sales_daily
(
dt string,
dc_id int,
order_date string,
total_sales float,
item_sku_id_hashed string
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','  TBLPROPERTIES ("skip.header.line.count"="1");
LOAD DATA LOCAL INPATH 'train_sku_sales_daily.csv' OVERWRITE INTO TABLE dev.dev_lzt_train_sku_sales_daily;
alter table dev.dev_lzt_train_sku_sales_daily set serdeproperties ('serialization.encoding'='GBK');

-- 5.train_sku_vendibility_daily

drop table if exists dev.dev_lzt_train_sku_vendibility_daily;
create table dev.dev_lzt_train_sku_vendibility_daily
(
stat_date string,
dim_delv_center_num int,
stock_qtty int,
allow_reserve_flag int,
sku_status_cd int,
dt string,
item_sku_id_hashed string
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','  TBLPROPERTIES ("skip.header.line.count"="1");
LOAD DATA LOCAL INPATH 'train_sku_vendibility_daily.csv' OVERWRITE INTO TABLE dev.dev_lzt_train_sku_vendibility_daily;
alter table dev.dev_lzt_train_sku_vendibility_daily set serdeproperties ('serialization.encoding'='GBK');

-- 6.test_sku_basic_info
drop table if exists dev.dev_lzt_test_sku_basic_info;
create table dev.dev_lzt_test_sku_basic_info
(
sku_name string,
item_name string,
item_desc string,
data_type int,
brand_codes string,
barndname_en string,
barndname_cn string,
barndname_full string,
item_origin string,
qgp int,
sku_valid_flag int,
item_valid_flag int,
sku_status_cd int,
item_status_cd string,
item_first_cate_cd int,
item_first_cate_name string,
item_second_cate_cd int,
item_second_cate_name string,
item_third_cate_cd int,
item_third_cate_name string,
shelves_tm string,
shelves_dt string,
otc_tm string,
utc_tm string,
support_cash_on_deliver_flag string,
vender_direct_delv_flag string,
slogan string,
sale_qtty_lim string,
first_into_wh_tm string,
item_type string,
size string,
size_rem string,
size_seq string,
len string,
width string,
height string,
calc_volume string,
wt string,
colour string,
pac_propt string,
pac_spec string,
free_goods_flag string,
item_sku_id_hashed string,
main_sku_id_hashed string
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' TBLPROPERTIES ("skip.header.line.count"="1");
LOAD DATA LOCAL INPATH '../xxxata_test/test_sku_basic_info.csv' OVERWRITE INTO TABLE dev.dev_lzt_test_sku_basic_info;
alter table dev.dev_lzt_test_sku_basic_info set serdeproperties ('serialization.encoding'='GBK');

-- 7.




