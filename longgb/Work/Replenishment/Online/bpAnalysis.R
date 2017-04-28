library(data.table)

# ====================================================
# =                     数据处理                     =
# ====================================================
# 读取数据
bpData          <- fread("bp_origin.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))

# 基本处理
bpData[, dt:=as.Date(dt)];
bpData[, rdcSkuid:=paste(int_org_num, sku_id, sep='-')]
rdcSkuid        <- unique(bpData[,rdcSkuid])


# ====================================================
# =                   缺失值填充                     =
# ====================================================
# missing data [ bp vlt_ref nrt ]
# 使用均值向上取整，填补
# x <- "10-1093041"
bpDataMissList      <- lapply(rdcSkuid, function(x){
  subData     <- bpData[rdcSkuid==x,]
  # 创建日期的list以判断缺失
  date_start      <- as.Date(min(subData[,dt]))
  date_end        <- as.Date(max(subData[,dt]))
  date_list       <- data.table(dt=seq(from=date_start, to=date_end, by=1))
  subData <- subData[date_list, on='dt']
  # 取均值向上取整，填补
  mean_bp <- as.integer(ceiling(mean(subData[,bp], na.rm = TRUE)))
  mean_vlt_ref <- as.integer(ceiling(mean(subData[,vlt_ref], na.rm = TRUE)))
  mean_nrt <- as.integer(ceiling(mean(subData[,nrt], na.rm = TRUE)))
  subData[is.na(bp), bp:=mean_bp]
  subData[is.na(vlt_ref), vlt_ref:=mean_vlt_ref]
  subData[is.na(nrt), nrt:=mean_nrt]
  subData[is.na(rdcSkuid), rdcSkuid:=x]
  subData
})
bpDataMissing <- rbindlist(bpDataMissList)


# 检查缺失情况。
# checkDataList      <- lapply(rdcSkuid, function(x){
#   subData     <- bpData[rdcSkuid==x,]
#   # 创建日期的list以判断缺失
#   date_start      <- as.Date(min(subData[,dt]))
#   date_end        <- as.Date(max(subData[,dt]))
#   date_list       <- data.table(dt=seq(from=date_start, to=date_end, by=1))
#   check <- date_list[!subData, on="dt"]
#   check$rdcSkuid = x
#   # 取均值向上取整，填补
#   check
# })
# checkData_test <- rbindlist(checkDataList)
# 发现缺失的日期存在有5个日期。
# unique(checkData_test[,dt])  # "2016-10-18" "2016-10-19" "2016-10-29" "2016-10-30" "2016-10-31"


# ====================================================
# =                     异常值处理                   =
# vendor-rdc-cid3 median+2sigma
# ====================================================
# outlier   3 sd [ strong outlier ]
bpDataOutList      <- lapply(rdcSkuid, function(x){
  subData     <- bpDataMissing[rdcSkuid==x,]
  # 取3倍标准差以外的，强异常值修改，使用中位数进行替代值。
  mean_vlt_ref <- mean(subData[,vlt_ref], na.rm = TRUE)
  median_vlt_ref <- median(subData[,vlt_ref], na.rm = TRUE)
  sd_vlt_ref <- sd(subData[,vlt_ref], na.rm = TRUE)
  upper_vlt_ref <- as.integer(ceiling(mean_vlt_ref + 2*sd_vlt_ref))
  # lower_vlt_ref <- as.integer(floor(mean_vlt_ref - 2*sd_vlt_ref))
  # subData[(vlt_ref>upper_vlt_ref)|(vlt_ref<lower_vlt_ref), vlt_ref:=median_vlt_ref]
  subData[vlt_ref > upper_vlt_ref, vlt_ref:=median_vlt_ref]
  subData
})
bpDataFinal <- rbindlist(bpDataOutList)

# 检查
# bpDataFinal[order(vlt_ref, decreasing = TRUE),.(rdcSkuid,vlt_ref)]

# 数据进行存储
write.table(bpDataFinal, file="bp.csv", sep=",", row.names=F)

