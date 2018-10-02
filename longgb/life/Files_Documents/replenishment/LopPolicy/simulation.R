library(data.table)
library(zoo)

# ========================================================
# =             Load the data and preprocessing          =
# ========================================================
# 0. Configure ####
path <- "D:/Lgb/data_sz/sample_data_2017"

# 1. Read Data ####
vltDD           <- fread(paste(path,"vltDDInput.csv", sep = "/"), integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));
forecastResult  <- fread(paste(path,"forecastInput.csv", sep = "/"), integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));
bpData          <- fread(paste(path,"bp.csv", sep = "/"), integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));


# 2. Preprocessing ####
# 2-1. The forecastResult checking ####
forecastResult[, salesFill:=list(mean(sales,na.rm=T)), by="rdcSkuid"]
forecastResult[is.na(sales), sales:=salesFill]
forecastResult[is.na(sales), sales:=0]                  # 那个skurdc为空的，则用0来填充
forecastResult      <- unique(forecastResult, by=c("rdcSkuid", "curDate"));      # 取rdcSkuid、curDate去重,取第一条？
# forecastResult[, length(sales), by=.(rdcSkuid, curDate)][order(V1, decreasing = TRUE)]
# forecastResult[rdcSkuid=='9-1103183' & curDate=='2016-11-03', ]  为什么会出现同一rdc、sku、date下销量预测有2条记录的情况


# 2-2. The vltDD preprocessing ####
vltDD[, rdc:=tstrsplit(metaFlag,'-')[3]];
vltDD[, supplier:=tstrsplit(metaFlag,'-')[2]];
vltDD[, rdcSkuid:=paste(rdc, sku_id, sep='-')];
vltDD   <- unique(vltDD, by=c("rdcSkuid", "intVlt"));   # 就是去重而已,已经验证均是一样的数据
# vltDD[, length(newDD), by=.(rdcSkuid, intVlt)][order(V1, decreasing = TRUE)]
# vltDD[rdcSkuid=='4-695467' & intVlt==1, ]
# 取两个数据集都有的rdc、sku
intersectRdcSkuid   <- intersect(vltDD$rdcSkuid, forecastResult$rdcSkuid);       # 取rdcSkuid重合的
vltDD               <- vltDD[rdcSkuid %in% intersectRdcSkuid,];
forecastResult      <- forecastResult[rdcSkuid %in% intersectRdcSkuid,];


# 2-3. The bpData preprocess ####
bpData[, rdcSkuid:=paste(int_org_num, sku_id, sep='-')];
bpData[, dt:=as.Date(dt)];
bpData$index    <- 1:nrow(bpData);
duplicatedCheck <- bpData[,list(recordCount=length(index)), by=c("rdcSkuid", "dt")][order(recordCount, decreasing=TRUE)];
bpData  <- bpData[,.(rdcSkuid, dt, bp)];


# ========================================================
# =                   Simulation framework               =
# ========================================================
# 3. LOP Simulation ####
simulateInitialDate   <- as.Date("2016-10-01");

# 3-1. sales data preprocessing ####
simulationData  <- forecastResult[curDate>=simulateInitialDate,];                          # 取"2016-10-01"后的日期进行仿真
beginRdcSkuidList   <- simulationData[curDate==simulateInitialDate, unique(rdcSkuid)];     # 仿真开始日期的rdcSkuid
simulationData  <- simulationData[rdcSkuid %in% beginRdcSkuidList,];                       # 取仿真开始日期的rdcSkuid的数据
simulationData[, c("sumSales","sumInventory"):=list(sum(sales,na.rm=T), sum(inventory,na.rm=T)), by=rdcSkuid];  # 取rdcSkuid的销量、库存之和

# 3-2. Remove skulist ####
removeRdcSku1   <- unique(simulationData[curDate==simulateInitialDate & is.na(inventory),]$rdcSkuid)            # 初始库存为0
removeRdcSku2   <- unique(simulationData[curDate==simulateInitialDate & is.na(predMean1), ]$rdcSkuid)           # 初始销量预测第一天为0
removeRdcSku3   <- unique(simulationData[sumInventory==0,]$rdcSkuid)                       # rdcSkuid的库存之和为0
removeRdcSku4   <- unique(simulationData[sumSales==0,]$rdcSkuid);                          # rdcSkuid的销量之和为0
removeRdSkuList     <- unique(c(removeRdcSku1, removeRdcSku2, removeRdcSku3,removeRdcSku4));
simulationData  <- simulationData[!rdcSkuid %in% removeRdSkuList,];
simulationData  <- simulationData[order(rdcSkuid, curDate),];

# 3-3. Start Simulation ####
# For every rdcSkuid pair do simulation
simulationKeyList   <- unique(simulationData$rdcSkuid)
LOPCalculationList  <- lapply(simulationKeyList, function(x){
        # x <- simulationKeyList[1]
        print(x);
        subVltData          <- vltDD[rdcSkuid==x & newDD >0.0, ];                          # newDD是vlt的概率
        subForecastData     <- simulationData[rdcSkuid==x,];
        
        # 3-3-1. sample vtl from the vlt distribution ####
        if(nrow(subVltData)>1){ 
            # 一些问题，inventory、sales为什么都相同
            subForecastData$sampleVlt  <- sample(subVltData$intVlt, size=nrow(subForecastData), prob=subVltData$newDD, replace=T)  # 抽取vlt
        }else{
           # 验证过一定有intVlt
            subForecastData$sampleVlt  <- subVltData$intVlt
        }
        
        # 3-3-2. The Expectation  求预测D_sales的期望 ####
        LOPMeanList     <- sapply(subVltData$intVlt, function(y){
            # y <- subVltData$intVlt[1]
            # The left part of the LOP
            subProb     <- subVltData[intVlt==y,]$newDD;
            vltKey      <- ifelse(y<=28, y, 28);                                           # 限制vlt不超过28
            expectationNames    <- paste('predMean', 1:vltKey, sep='');
            subMeanResult   <- rowSums(subset(subForecastData, select=expectationNames))*subProb;              # 求和乘以概率？
            subMeanResult   <- subMeanResult*y/vltKey;                                     # 结果乘以*y/vltKey
            subMeanResult;
        })
        subForecastData$LOPMean <- rowSums(LOPMeanList)                                    # 求的是预测D_sales的期望
        
        # 3-3-3. The Variance 求预测D_sales的方差 ####
        expectationOfVarianceList         <- sapply(subVltData$intVlt, function(y){
            # y <- subVltData$intVlt[1]
            subProb     <- subVltData[intVlt==y,]$newDD;
            vltKey      <- ifelse(y<=28, y, 28);
            expectationNames    <- paste('predMean', 1:vltKey, sep='');
            subMeanResult   <- rowSums(subset(subForecastData, select=expectationNames));
            subMeanResult   <- subMeanResult*y/vltKey;
            subMeanResult   <- subMeanResult-subForecastData$LOPMean;
            conditionVariance1  <- subMeanResult^2*subProb;
            conditionVariance1;
        })
        subForecastData$conditionVariance1 <- rowSums(expectationOfVarianceList)

        # The second part of the condition vairance, expectation of the variance
        varianceOfConditionMeanList    <- sapply(subVltData$intVlt, function(y){
            vltKey      <- ifelse(y<=28, y, 28);
            sdNames         <- paste('predSd', 1:vltKey, sep='');
            subSdResult     <- rowSums(subset(subForecastData, select=sdNames)^2);
            subSdResult     <- subSdResult*y/vltKey;
            subSdResult;
        })
        subForecastData$conditionVariance2  <- rowSums(varianceOfConditionMeanList);
        
        subForecastData[, LOPSd:=sqrt(conditionVariance1+conditionVariance2)];
        
    })

LOPCalculationData <- rbindlist(LOPCalculationList)


# ========================================================
# =                       The BP logic                   =
# ========================================================
# 3. BP Simulation ####
# check if the LOPCalculationData is redundant
LOPCalculationData  <- unique(LOPCalculationData, by=c("rdcSkuid", "curDate"));
LOPCalculationData[, curDate:=as.Date(curDate)];
# 交
setkeyv(LOPCalculationData, c("rdcSkuid", "curDate")); 
setkeyv(bpData, c("rdcSkuid", "dt"));
joinData    <- bpData[LOPCalculationData]
# 填充
joinData[, fillBp:=mean(bp, na.rm=T), by=rdcSkuid];
joinData[is.na(bp), bp:=as.integer(fillBp)];
# bp
bpList  <- unique(joinData$bp)
bpCalculationList   <- lapply(bpList, function(x){
    subData     <- joinData[bp==x, ];
    bpKey       <- ifelse(x<=28, x, 28);
    expectationNames    <- paste('predMean', 1:bpKey, sep='');
    sdNames             <- paste('predSd', 1:bpKey, sep='');
    bpMean          <- rowSums(subset(subData,select=expectationNames));
    bpSd            <- sqrt(rowSums(subset(subData, select=sdNames)^2));
    subData$bpMean  <- bpMean;
    subData$bpSd    <- bpSd;
    subData;
})
bpCalculationData   <- rbindlist(bpCalculationList)
save(LOPCalculationData, bpCalculationData, file="temp.rda")
# load("temp.rda")

# calculate the 0.95 quantile
finalData   <- subset(bpCalculationData, select=c(1:10, 69,70,73,74,75,76))
finalData[, LOP95:=round(qnorm(0.95, LOPMean, LOPSd))];
finalData[, bp95:=round(qnorm(0.95, bpMean, bpSd))];


# ========================================================
# =                       Do simulation                 =
# ========================================================
# (1) Initial value of the simulation data
# 初始化 Initialized the data
simulateInitialDate   <- as.Date("2016-10-01");
simulateEndDate	<- as.Date("2016-12-31");
finalData[,c("AQ", "simuOpenpo", "simuInv","simuSales","pur_qtty"):=0];
finalData[dt == simulateInitialDate, simuInv:=inventory];
finalData[is.na(sales), sales:=0];

# 读取 load vlt from test set
vltDD2 <- fread("vltDDInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));
vltData <- fread("vltData.csv",integer64='numeric',sep="\t",na.strings=c("NULL","NA", "", "\\N"));
setnames(vltData, c("pur_bill_id","sku_id","item_third_cate_cd","supp_brevity_cd","int_org_num","store_id","create_tm","complete_dt","into_wh_tm","t_6","new_vlt","int_new_vlt","flag","isautopo"));
skus <-  unique(vltDD2$sku_id);

# 初步处理
vltDataList <- vltData[(sku_id %in% skus) & (create_tm>= '2016-10-01'),.(pur_bill_id,sku_id,supp_brevity_cd,int_org_num,create_tm,int_new_vlt)];
vltDataList[,rdcSkuidSupp:=paste(int_org_num,sku_id,supp_brevity_cd,sep="-")];
vltDataList[,create_tm:=as.Date(create_tm)];
vltDataList <- vltDataList[order(sku_id,int_org_num,create_tm),];

# vlt
vlt <- lapply(unique(vltDataList$rdcSkuidSupp), function(x){	
    print(x);
		sbset <- vltDataList[rdcSkuidSupp==x,];
		data.table(rdcSkuidSupp=x,vlt=paste(sbset$int_new_vlt,collapse=","))
});

# 处理vlt
vltList <- rbindlist(vlt);
vltList[,rdc:=tstrsplit(rdcSkuidSupp,"-")[1]];
vltList[,sku:=tstrsplit(rdcSkuidSupp,"-")[2]];
vltList[,rdcSku:=paste(rdc,sku,sep="-")];
vltList[,sku:=NULL];
vltList[,rdc:=NULL];
vltList[,rdcSkuidSupp:=NULL];
setkey(vltList,rdcSku);

# 交
rdcskuList <-  intersect(finalData$rdcSkuid,vltList$rdcSku);
finalData <-  finalData[rdcSkuid %in% rdcskuList,];
setkey(finalData,rdcSkuid);

testData <- vltList[finalData];
testData[,rdcSkuid:=rdcSku];
testData[,rdcSku:=NULL];

###simulation
#testData    <- copy(finalData);

totalDates <-  as.Date(simulateInitialDate:simulateEndDate);

testData <-  testData[dt %in% totalDates,];
testData[,amt:=length(unique(dt)),by="rdcSkuid"];
testData <-  testData[amt==88,];
testData[,amt:=NULL];


totalSkus <-  unique(testData$rdcSkuid);
fullData <-  data.table(expand.grid(totalDates,totalSkus));
setnames(fullData,c("dt","rdcSkuid"));
setkeyv(fullData,c("rdcSkuid","dt"));
setkeyv(testData,c("rdcSkuid","dt"));
testData <-  testData[fullData];
testData <- testData[order(rdcSkuid,dt),];

testData[,c("AQ", "simuOpenpo", "simuInv","simuSales","pur_qtty"):=0];
testData[dt == simulateInitialDate, simuInv:=inventory];
testData[is.na(sales), sales:=0];

#testData[,bp2:= as.integer(mean(bp,na.rm=T)), by=rdcSkuid];
#testData[is.na(bp),bp:=bp2]
#testData[,c("bp2"):=NULL]

testData[,LOP952:=as.integer(mean(LOP95,na.rm=T)),by=rdcSkuid];
testData[is.na(LOP95),LOP95:=as.integer(LOP952)];
testData[,LOP952:=NULL];

testData[,bp952:=as.integer(mean(bp95,na.rm=T)),by=rdcSkuid];
testData[is.na(bp95),bp95:=as.integer(bp952)];
testData[,bp952:=NULL];
###vectorized by date

#testData[,sampleVlt2:=mean(sampleVlt,na.rm=T),by=rdcSkuid];
#testData[is.na(sampleVlt),sampleVlt:=sampleVlt2];
#testData[,sampleVlt2:=NULL];


testData[,sampleVLT:=max(vlt,na.rm=T),by=rdcSkuid];
testData[,vlt:=0];
testData[,vltIndex:=1];


testData2 <- copy(testData)
testData <- copy(testData2);

# testData=testData[rdcSkuid %in% c("10-1187576","10-1187581"),];


# Final Simulation ####
for(x in totalDates){
  	print(as.Date(x));
    # 更新inv，计算sales
	  subData<- testData[dt==x, ];   
    subData[, simuInv:=simuInv+AQ];               
    subData[, 		simuSales	:=sales];subData[sales>simuInv, simuSales:=simuInv]; 
  	testData[dt==x,	simuSales	:=sales];testData[sales>simuInv, simuSales:=simuInv];
  	testData[dt==x, simuInv:=simuInv+AQ]; 
  	RQData  <- subData[simuInv+simuOpenpo<LOP95,];
    RQData[, DQ:=LOP95+bp95-(simuOpenpo+simuInv)];
	  for(y in RQData$rdcSkuid){
    		print(y);
		    i <-  testData[rdcSkuid==y & dt== x ,vltIndex];
		    DQ  <- RQData[rdcSkuid==y,]$DQ;
    		if(DQ>0) testData[rdcSkuid==y & dt> x ,vltIndex:= vltIndex+1]; 
         realVltList <- as.integer(unlist(strsplit(RQData[rdcSkuid==y,]$sampleVLT,",")));
		    len <-  length(realVltList);
		    realVlt	<- 	ifelse(i%%len==0,len,realVltList[i%%len]);
            testData[rdcSkuid==y & dt %in% (1:realVlt+x), simuOpenpo:=simuOpenpo+DQ];  # 更新在途
            testData[rdcSkuid==y & dt %in% (realVlt+1+x), c("AQ","simuInv"):=list(AQ+DQ,simuInv+DQ)]; # 更新到达
    		testData[rdcSkuid==y & dt== x ,pur_qtty:= DQ];
    		testData[rdcSkuid==y & dt== x ,vlt:= realVlt];
  	};
	  subData[, simuInvNext:=simuInv-simuSales];
    subData[, dt:=dt+1];
    subData <- subset(subData, select=c("rdcSkuid", "dt", "simuInvNext"));
    setkeyv(subData, c("rdcSkuid", "dt"));
    setkeyv(testData, c("rdcSkuid", "dt"));
    testData    <- subData[testData];
    testData[!is.na(simuInvNext), simuInv:=simuInvNext];
  	testData[,simuInvNext:=NULL];
}
write.table(testData,"testData.txt",sep="\t",row.names=F);
dat <-  fread("testData.txt",sep="\t");
dat[,inv:=0];
dat[simuInv>0,inv:=1];
dat[,act_inv:=0];
dat[inventory>0,act_inv:=1];

kpi <-  dat[, list(
                  simuCr=sum(inv,na.rm=T)/length(inv),
                  simuIto=sum(simuInv,na.rm=T)/sum(simuSales,na.rm=T),
                  Cr=sum(act_inv,na.rm=T)/length(act_inv),
                  Ito=sum(inventory,na.rm=T)/sum(sales,na.rm=T)
              ),
              by=rdcSkuid]
