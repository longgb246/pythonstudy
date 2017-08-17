drop table if exists dev.dev_lgb_train_sku_basic_info;
create table dev.dev_lgb_train_sku_basic_info
(
    sku_name    varchar(255),
    item_name   varchar(255),
    item_desc   varchar(255),
    data_type   varchar(255),
    brand_code  varchar(255),
    barndname_en    varchar(255),
    barndname_cn    varchar(255),
    barndname_full  varchar(255),
    item_origin     varchar(255),
    qgp         varchar(255),
    sku_valid_flag      varchar(255),
    item_valid_flag     varchar(255),
    sku_status_cd       varchar(255),
    item_status_cd      varchar(255),
    item_first_cate_cd      varchar(255),
    item_first_cate_name    varchar(255),
    item_second_cate_cd     varchar(255),
    item_second_cate_name   varchar(255),
    item_third_cate_cd      varchar(255),
    item_third_cate_name    varchar(255),
    shelves_tm          date,
    shelves_dt          date,
    otc_tm              date,
    utc_tm              date,
    support_cash_on_deliver_flag        varchar(255),
    vender_direct_delv_flag             varchar(255),
    slogan              varchar(255),
    sale_qtty_lim       double,
    first_into_wh_tm    date,
    item_type           date,
    size                double,
    size_remsize_seq    double,
    len                 double,
    width               double,
    height              double,
    calc_volume         double,
    wt                  double,
    colour              varchar(255),
    pac_propt           varchar(255),
    pac_spec            varchar(255),
    free_goods_flag     varchar(255),
    item_sku_id_hashed  varchar(255),
    main_sku_id_hashed  varchar(255)
)
row format delimited
fields terminated by '\t';

load data local inpath 'train_sku_basic_info.csv' into table dev.dev_lgb_train_sku_basic_info;

