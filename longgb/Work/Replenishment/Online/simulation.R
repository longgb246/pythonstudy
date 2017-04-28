library(data.table)
################################################
####  Load the data and preprocessing
################################################
forecastResult  <- fread("forecastInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
bpData          <- fread("bp.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
vltDD           <- fread("vltDDInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))


####  The forecast data preprocessing
forecastResult[, salesFill:=as.integer(ceiling(mean(sales,na.rm=T))), by="rdcSkuid"]
forecastResult[is.na(sales), sales:=salesFill]
forecastResult[is.na(sales), sales:=0]
forecastResult[, curDate:=as.Date(curDate)]
forecastResult$index    <- 1:nrow(forecastResult)
forecastResult      <- unique(forecastResult, by=c("rdcSkuid", "curDate"))


####  The vlt data preprocessing
vltDD[, rdc:=tstrsplit(metaFlag,'-')[3]]
vltDD[, supplier:=tstrsplit(metaFlag,'-')[2]]
vltDD[, rdcSkuid:=paste(rdc, sku_id, sep='-')]
vltDD   <- unique(vltDD, by=c("rdcSkuid", "intVlt"))


####  The bp data preprocess
bpData[, rdcSkuid:=paste(int_org_num, sku_id, sep='-')]
bpData[, dt:=as.Date(dt)]
# Check if oneday get multiple record
# bpData$index    <- 1:nrow(bpData)
# duplicatedCheck <- bpData[,list(recordCount=length(index)), by=c("rdcSkuid", "dt")]
bpData  <- bpData[, .(rdcSkuid, dt,vlt_ref, nrt, bp)]


####  Get the same rdcSkuid
intersectRdcSkuid   <- intersect(unique(vltDD$rdcSkuid), unique(forecastResult$rdcSkuid))
forecastResult      <- forecastResult[rdcSkuid %in% intersectRdcSkuid,]
intersectRdcSkuid   <- intersect(unique(bpData$rdcSkuid), unique(forecastResult$rdcSkuid))
bpData              <- bpData[rdcSkuid %in% intersectRdcSkuid,]
vltDD               <- vltDD[rdcSkuid %in% intersectRdcSkuid,]
forecastResult      <- forecastResult[rdcSkuid %in% intersectRdcSkuid,]


####  If pred<0, assign 0 to pred
# 1-10
forecastResult[predMean1<0, predMean1:=0];forecastResult[predMean2<0, predMean2:=0];
forecastResult[predMean3<0, predMean3:=0];forecastResult[predMean4<0, predMean4:=0];
forecastResult[predMean5<0, predMean5:=0];forecastResult[predMean6<0, predMean6:=0];
forecastResult[predMean7<0, predMean7:=0];forecastResult[predMean8<0, predMean8:=0];
forecastResult[predMean9<0, predMean9:=0];forecastResult[predMean10<0, predMean10:=0];
# 11-20
forecastResult[predMean11<0, predMean11:=0];forecastResult[predMean12<0, predMean12:=0];
forecastResult[predMean13<0, predMean13:=0];forecastResult[predMean14<0, predMean14:=0];
forecastResult[predMean15<0, predMean15:=0];forecastResult[predMean16<0, predMean16:=0];
forecastResult[predMean17<0, predMean17:=0];forecastResult[predMean18<0, predMean18:=0];
forecastResult[predMean19<0, predMean19:=0];forecastResult[predMean20<0, predMean20:=0];
# 21-28
forecastResult[predMean21<0, predMean21:=0];forecastResult[predMean22<0, predMean22:=0];
forecastResult[predMean23<0, predMean23:=0];forecastResult[predMean24<0, predMean24:=0];
forecastResult[predMean25<0, predMean25:=0];forecastResult[predMean26<0, predMean26:=0];
forecastResult[predMean27<0, predMean27:=0];forecastResult[predMean28<0, predMean28:=0];


################################################
####  Simulation framework
#  (1) Forecast data preprocessing
################################################
simulateInitialDate   <- as.Date("2016-10-01")
simulationData  <- forecastResult[curDate>=simulateInitialDate,]

#### Make the intial date get the data
beginRdcSkuidList   <- simulationData[curDate==simulateInitialDate, unique(rdcSkuid)]
simulationData      <- simulationData[rdcSkuid %in% beginRdcSkuidList,]


#### Delete some data
simulationData[, c("sumSales","sumInventory"):=list(sum(sales,na.rm=T), sum(inventory,na.rm=T)), by=rdcSkuid]
# The initial inventory is zero
removeRdcSku1   <- unique(simulationData[curDate==simulateInitialDate & is.na(inventory),]$rdcSkuid)
# The initial forecast result is zero
removeRdcSku2   <- unique(simulationData[curDate==simulateInitialDate & is.na(predMean1), ]$rdcSkuid)
# The total inventory is zero
removeRdcSku3   <- unique(simulationData[sumInventory==0,]$rdcSkuid)
# The total sales is zero
removeRdcSku4   <- unique(simulationData[sumSales==0,]$rdcSkuid)
removeRdSkuList     <- unique(c(removeRdcSku1, removeRdcSku2, removeRdcSku3,removeRdcSku4))

simulationData  <- simulationData[!rdcSkuid %in% removeRdSkuList,]
simulationData  <- simulationData[order(rdcSkuid, curDate),]


#### The calculation of safe vlt
bpData[, safeVLT:=vlt_ref+nrt]
bpData[safeVLT<14, safeVLT:=14]

setkeyv(bpData, c("rdcSkuid", "dt"))
setkeyv(simulationData, c("rdcSkuid", "curDate"))

simulationData  <- bpData[simulationData]
simulationData  <- simulationData[!is.na(bp),]


# ==================================================================
# 10-1093041 2016-10-18
bpData[rdcSkuid=="10-1093041" & dt=="2016-10-18", ]
bpData[rdcSkuid=="10-1093041", .(rdcSkuid, dt)]
bpDataFinal[rdcSkuid=="10-1093041", .(rdcSkuid, dt)]
bpDataFinal[rdcSkuid=="10-1093041" & dt=="2016-10-18", .(rdcSkuid, dt, bp, vlt_ref, nrt)]
bpDataMissing[rdcSkuid=="10-1093041", .(rdcSkuid, dt)]
subData[rdcSkuid=="10-1093041", .(rdcSkuid, dt, bp, vlt_ref, nrt)]
# ==================================================================


safeVLTList     <- unique(simulationData$safeVLT)
LOPList     <- lapply(safeVLTList, function(x){
        subData <- simulationData[safeVLT==x,]
        safeVltKey      <- ifelse(x<=28, x, 28)
        ###The expectation
        expectationNames    <- paste('predMean', 1:safeVltKey ,sep='')
        subMeanResult   <- rowSums(subset(subData,select=expectationNames))
        subMeanResult   <- subMeanResult*x/safeVltKey
        ###The variance
	sdNames         <- paste('predSd', 1:safeVltKey, sep='')
        subSdResult     <- sd(rowSums(subset(subData,select=sdNames)^2,na.rm=T))
        subSdResult     <- subSdResult*x/safeVltKey
	subData$LOPMean	<- subMeanResult
	subData$LOPSd	<- subSdResult
	subData
})

LOPData <- rbindlist(LOPList)
###
bpList  <- unique(LOPData$bp)
bpCalculationList   <- lapply(bpList, function(x){
    subData     <- LOPData[bp==x, ]
    bpKey       <- ifelse(x<=28, x, 28);
    expectationNames    <- paste('predMean', 1:bpKey, sep='');
    sdNames             <- paste('predSd', 1:bpKey, sep='')
    bpMean          <- rowSums(subset(subData,select=expectationNames))
    bpSd            <- sqrt(rowSums(subset(subData, select=sdNames)^2))
    subData$bpMean  <- bpMean
    subData$bpSd    <- bpSd
    subData
})

bpCalculationData   <- rbindlist(bpCalculationList)
finalData   <- subset(bpCalculationData, select=c(1:13,73:76))

finalData[, LOP95:=round(qnorm(0.95, LOPMean, LOPSd))]
finalData[, bp95:=round(qnorm(0.95, bpMean, bpSd))]


##########################################################
### Do simulation
### (1) Initial value of the simulation data
###########################################################
##### Initialized the data
simulateInitialDate   <- as.Date("2016-10-01")
finalData[,c("AQ", "simuOpenpo", "simuInv"):=0]
finalData[dt == simulateInitialDate, simuInv:=as.numeric(inventory)]
finalData[is.na(sales), sales:=0]

testData    <- copy(finalData)
totalDates  <- sort(unique(finalData$dt))
# Final Simulation ####
for(x in totalDates){
    print(x);
    subData     <- testData[dt==x, ];   
    subData[, simuInv:=simuInv+AQ];               
    subData[,simuSales:=sales];
    subData[sales>simuInv, simuSales:=simuInv];  #In case no inventory sales
    ###Interate in the testData
    testData[dt==x, simuSales:=sales];
    testData[sales>simuInv, simuSales:=simuInv];
    testData[dt==x, simuInv:=simuInv+AQ]; 
    
    RQData  <- subData[simuInv+simuOpenpo < LOP95,];
    RQData[, DQ:=LOP95 + bp95 -( simuOpenpo + simuInv)];
    
    for(y in RQData$rdcSkuid){
        print(y);
        i   <-  testData[rdcSkuid==y & dt== x ,vltIndex];
        DQ  <-  RQData[rdcSkuid==y,]$DQ;
        if(DQ>0) testData[rdcSkuid==y & dt> x ,vltIndex:= vltIndex+1]; 
        realVltList <-  as.integer(unlist(strsplit(RQData[rdcSkuid==y,]$sampleVLT,",")));
	    len         <-  length(realVltList);
	    realVlt	    <- 	ifelse(i%%len==0,len,realVltList[i%%len]);
        testData[rdcSkuid==y & dt %in% (1:realVlt+x), simuOpenpo:=simuOpenpo+DQ];
        testData[rdcSkuid==y & dt %in% (realVlt+1+x), c("AQ","simuInv"):=list(AQ+DQ,simuInv+DQ)];
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

