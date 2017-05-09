library(data.table)
help("data.table")


# =======================================================
# =                       文档的例子                    =
# =======================================================

# 1 基本信息 ####
df1 = data.frame(x=rep(c("b", "a", "c"), each=3), y=c(1,3,6), v=1:9)
dt1 = data.table(x=rep(c("b", "a", "c"), each=3), y=c(1,3,6), v=1:9)
identical(dim(df1), dim(dt1))     # TRUE
identical(df1$a, dt1$a)           # TRUE
is.list(df1)                      # TRUE
is.list(dt1)                      # TRUE
is.data.frame(dt1)                # TRUE
tables()

  
# 2 索引、切片 ####  
dt1[2,]
dt1[3:2,]
dt1[order(y),]                    # 按照x进行排序，语法逻辑:1、先对x进行序列的排序，2、序列引用于原data.table上面
dt1[y>2 & v>5,][order(y, v),]     # 怎么进行按照2个顺序进行排序，python可以
dt1[!(2:4)]
dt1[,list(v)]                     # 使用list()，使得返回结果依然是data.table
dt1[,sum:=sum(v)]                 # 使用计算函数
dt1[,sum:=NULL]                   # 删掉某一行
dt1[,list(a=sum(v), b=v^2)]       # 返回两列，并且命名为a,b


# 3 聚合函数 ####
dt1[, sum(v), by=x]               # 按照x进行聚合，计算每个x的sum
dt1[, sum(v), by=list(x, sm)]     # 多个groupby


# 4 交并 ####
dt2 = data.table(x=c("c", "b"), v=8:7, foo=c(4,2))
dt1[dt2, on="x"]                  # right join，但是左边的字段是保留
dt2[dt1, on="x"]                  # left join，空的值为NA
dt1[dt2, on="x", nomatch=0]       # inner join
dt1[!dt2, on="x"]                 # 差交
dt1[dt2, on=c("x", "v"), nomatch=0]          # 多个字段的交集


# 5 问题 ####
dt1[nrow(dt1),]                   # 最后一行 
dt1[, ncol(dt1)]                  # 怎么进行iloc类似的索引？


# 6 高级 ####
dt3 = data.table(x=rep(c("b","a","c"),each=3), v=c(1,1,1,2,2,1,1,2,2), y=c(1,3,6), a=1:9, b=9:1)
dt3[, sum(v), by=.(y%%2)]         # 根据y%%2的计算结果进行分类统计
dt3[, list(sm=sum(v)), by=.(bool = y%%2)]   # 改变列名字
dt3[, list(MySum=sum(v), MyMin=min(v), MyMax=max(v)), by=list(x, y%%2)]     # 多列计算
dt3[, .(seq=min(a):max(b)), by=x]
dt3[, {tmp <- mean(y);tmp <- tmp-1;.(a=a-tmp, b=b-tmp)}, by=x]              # 使用{进行中间计算} 










2353268

# =======================================================
# =                      R In Action                    =
# =======================================================

vars <- c("mpg", "hp", "wt")
mtcars[vars]
mm <- data.table(mtcars[vars])
mm$index <- row.names(mtcars)
names(mm)             # 取列名
row.names(mtcars)     # 取index名
summary(subset(mm, select = vars))      # 使用subset()进行选取

# 143.
# 应该直接学习原理，然后用python直接实现，否则还是黑箱子。




DT = data.table(A=5:1,B=letters[5:1])
setkey(DT,B)    # re-orders table and marks it sorted.
DT[J("b")]      # returns the 2nd row
DT[.("b")]      # same. Style of package plyr.
DT[list("b")]   # same

# CJ usage examples
CJ(c(5,NA,1), c(1,3,2)) # sorted and keyed data.table
do.call(CJ, list(c(5,NA,1), c(1,3,2))) # same as above
CJ(c(5,NA,1), c(1,3,2), sorted=FALSE) # same order as input, unkeyed
# use for 'unique=' argument
x = c(1,1,2)
y = c(4,6,4)
CJ(x, y, unique=TRUE) # unique(x) and unique(y) are computed automatically










path <- "D:/Lgb/data_sz/sample_data_2017"
bpData          <- fread(paste(path,"bp.csv", sep = "/"), integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));




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
# ====================================================
# outlier   3 sd [ strong outlier ]
bpDataOutList      <- lapply(rdcSkuid, function(x){
  subData     <- bpDataMissing[rdcSkuid==x,]
  # 取3倍标准差以外的，强异常值修改，使用中位数进行替代值。
  mean_vlt_ref <- mean(subData[,vlt_ref], na.rm = TRUE)
  median_vlt_ref <- median(subData[,vlt_ref], na.rm = TRUE)
  sd_vlt_ref <- sd(subData[,vlt_ref], na.rm = TRUE)
  upper_vlt_ref <- as.integer(ceiling(mean_vlt_ref + 3*sd_vlt_ref))  
  lower_vlt_ref <- as.integer(floor(mean_vlt_ref - 3*sd_vlt_ref))  
  subData[(vlt_ref>upper_vlt_ref)|(vlt_ref<lower_vlt_ref), vlt_ref:=median_vlt_ref]
  subData
})
bpDataFinal <- rbindlist(bpDataOutList)

# 检查
# bpDataFinal[order(vlt_ref, decreasing = TRUE),.(rdcSkuid,vlt_ref)]

# 数据进行存储
write.table(bpDataFinal, file="bp.csv", sep=",", row.names=F)





help("colMeans")
library(data.table)
help("data.table")

help(mean)
mean()

help(sort)

