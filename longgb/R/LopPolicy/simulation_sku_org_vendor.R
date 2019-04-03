library(data.table);
library(zoo);
############################################
####Load the data and preprocessing
################################################
vltDD           <- fread("C:/Users/Administrator/Downloads/vltDDInput_sku_org_vendor.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));
forecastResult  <- fread("C:/Users/Administrator/Downloads/forecastInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));
bpData          <- fread("C:/Users/Administrator/Downloads/bp.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));

### preprocessing
####The forecast result checking


#duplicatedCheck     <- forecastResult[,list(recordCount=length(index)), by=c("rdcSkuid", "curDate")]
#duplicatedCheck[,table(recordCount)]
forecastResult      <- unique(forecastResult, by=c("rdcSkuid", "curDate"));

###The vlt data preprocessing
vltDD[, rdc:=tstrsplit(metaFlag,'-')[3]];
vltDD[, supplier:=tstrsplit(metaFlag,'-')[2]];
vltDD[,rdcSkuid:=paste(rdc, sku_id, sep='-')];
vltDD   <- unique(vltDD, by=c("rdcSkuid", "intVlt"));

intersectRdcSkuid   <- intersect(vltDD$rdcSkuid, forecastResult$rdcSkuid);
vltDD               <- vltDD[rdcSkuid %in% intersectRdcSkuid,];
forecastResult      <- forecastResult[rdcSkuid %in% intersectRdcSkuid,];

## bp data preprocess
bpData[, rdcSkuid:=paste(int_org_num, sku_id, sep='-')];
bpData[, dt:=as.Date(dt)];
bpData$index    <- 1:nrow(bpData);
duplicatedCheck <- bpData[,list(recordCount=length(index)), by=c("rdcSkuid", "dt")];

bpData  <- bpData[,.(rdcSkuid, dt, bp)];

######################################################3
#### Simulation framework
###One. sales data preprocessing
######################################################3
simulateInitialDate   <- as.Date("2016-10-01");
simulationData  <- forecastResult[curDate>=simulateInitialDate,];

####make the intial date get the data
beginRdcSkuidList   <- simulationData[curDate==simulateInitialDate, unique(rdcSkuid)];
simulationData  <- simulationData[rdcSkuid %in% beginRdcSkuidList,];

simulationData[, c("sumSales","sumInventory"):=list(sum(sales,na.rm=T), sum(inventory,na.rm=T)), by=rdcSkuid];
#The initial inventory is zero
removeRdcSku1   <- unique(simulationData[curDate==simulateInitialDate & is.na(inventory),]$rdcSkuid);
#The initial forecast result is zero
removeRdcSku2   <- unique(simulationData[curDate==simulateInitialDate & is.na(predMean1), ]$rdcSkuid);
#The total inventory is zero
removeRdcSku3   <- unique(simulationData[sumInventory==0,]$rdcSkuid);
# The total sales is zero
removeRdcSku4   <- unique(simulationData[sumSales==0,]$rdcSkuid);
removeRdSkuList     <- unique(c(removeRdcSku1, removeRdcSku2, removeRdcSku3,removeRdcSku4));

simulationData  <- simulationData[!rdcSkuid %in% removeRdSkuList,];
simulationData  <- simulationData[order(rdcSkuid, curDate),];

#####For every rdcSkuid pair do simulation
simulationKeyList   <- unique(simulationData$rdcSkuid)
LOPCalculationList <- lapply(simulationKeyList, function(x){
    print(x);
    subVltData          <- vltDD[rdcSkuid==x & skuDD >0.0, ];
    subForecastData     <- simulationData[rdcSkuid==x,];
    #sample vtl from the vlt distribution
    if(nrow(subVltData)>1)
    { subForecastData$sampleVlt  <- sample(subVltData$intVlt, size=nrow(subForecastData), prob=subVltData$skuDD, replace=T)}
    else{subForecastData$sampleVlt  <- subVltData$intVlt};
    ###The left part of the VLT, the expectation
  	LOPMeanList     <- sapply(subVltData$intVlt, function(y){
        ###The left part of the LOP
        subProb     <- subVltData[intVlt==y,]$skuDD;
        vltKey      <- ifelse(y<=28, y, 28);
        expectationNames    <- paste('predMean', 1:vltKey, sep='');
        subMeanResult   <- rowSums(subset(subForecastData,select=expectationNames))*subProb;
        subMeanResult   <- subMeanResult*y/vltKey;
        subMeanResult;
});
     subForecastData$LOPMean <- rowSums(LOPMeanList);
    ###The first part of the condition vairance, expectation of the variance
     expectationOfVarianceList         <- sapply(subVltData$intVlt, function(y){
        ###The left part of the LOP
        subProb     <- subVltData[intVlt==y,]$skuDD;
        vltKey      <- ifelse(y<=28, y, 28);
        expectationNames    <- paste('predMean', 1:vltKey, sep='');
        subMeanResult   <- rowSums(subset(subForecastData,select=expectationNames));
        subMeanResult   <- subMeanResult*y/vltKey;
        subMeanResult   <- subMeanResult-subForecastData$LOPMean;
        conditionVariance1  <- subMeanResult^2*subProb;
        conditionVariance1;
});
     subForecastData$conditionVariance1 <- rowSums(expectationOfVarianceList);

      ###The second part of the condition vairance, expectation of the variance
    varianceOfConditionMeanList    <- sapply(subVltData$intVlt, function(y){
        vltKey      <- ifelse(y<=28, y, 28);
        sdNames         <- paste('predSd', 1:vltKey, sep='');
        subSdResult     <- rowSums(subset(subForecastData,select=sdNames)^2);
        subSdResult     <- subSdResult*y/vltKey;
        subSdResult;
});
    subForecastData$conditionVariance2  <- rowSums(varianceOfConditionMeanList);
    subForecastData[, LOPSd:=sqrt(conditionVariance1+conditionVariance2)];
})

LOPCalculationData <- rbindlist(LOPCalculationList)


#3################################################################
# The BP logic 
#3################################################################
### check if the LOPCalculationData is redundant
LOPCalculationData  <- unique(LOPCalculationData, by=c("rdcSkuid", "curDate"));
LOPCalculationData[, curDate:=as.Date(curDate)];

setkeyv(LOPCalculationData, c("rdcSkuid", "curDate"));
setkeyv(bpData, c("rdcSkuid", "dt"));

joinData    <- bpData[LOPCalculationData];

###
joinData[, fillBp:=mean(bp, na.rm=T), by=rdcSkuid];
joinData[is.na(bp), bp:=as.integer(fillBp)];

bpList  <- unique(joinData$bp);
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

bpCalculationData   <- rbindlist(bpCalculationList);

save(LOPCalculationData,bpCalculationData, file="temp_skudd.rda");


finalData   <- subset(bpCalculationData, select=c(1:10, 69,70,73,74,75,76));

#load("temp.rda")
###calculate the 0.95 quantile
finalData[, LOP95:=round(qnorm(0.95, LOPMean, LOPSd))];
finalData[, bp95:=round(qnorm(0.95, bpMean, bpSd))];

##########################################################
### Do simulation
### (1) Initial value of the simulation data
###########################################################
##### Initialized the data
simulateInitialDate   <- as.Date("2016-10-01");
simulateEndDate	<- as.Date("2016-12-31");

finalData[,c("AQ", "simuOpenpo", "simuInv","simuSales","pur_qtty"):=0];
finalData[dt == simulateInitialDate, simuInv:=inventory];
finalData[is.na(sales), sales:=0];


###load vlt from test set
###
###
vltDD2=fread("vltDDInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"));
vltData <- fread("vltData.csv",integer64='numeric',sep="\t",na.strings=c("NULL","NA", "", "\\N"));
setnames(vltData, c("pur_bill_id","sku_id","item_third_cate_cd","supp_brevity_cd","int_org_num","store_id","create_tm","complete_dt","into_wh_tm","t_6","new_vlt","int_new_vlt","flag","isautopo"));
skus= unique(vltDD2$sku_id);
vltDataList=vltData[(sku_id %in% skus) & (create_tm>= '2016-10-01'),.(pur_bill_id,sku_id,supp_brevity_cd,int_org_num,create_tm,int_new_vlt)];
vltDataList[,rdcSkuidSupp:=paste(int_org_num,sku_id,supp_brevity_cd,sep="-")];
vltDataList[,create_tm:=as.Date(create_tm)];
vltDataList=vltDataList[order(sku_id,int_org_num,create_tm),];

vlt=lapply(unique(vltDataList$rdcSkuidSupp),function(x)
	{	print(x);
		sbset = vltDataList[rdcSkuidSupp==x,];
		data.table(rdcSkuidSupp=x,vlt=paste(sbset$int_new_vlt,collapse=","))});
vltList=rbindlist(vlt);
vltList[,rdc:=tstrsplit(rdcSkuidSupp,"-")[1]];
vltList[,sku:=tstrsplit(rdcSkuidSupp,"-")[2]];
vltList[,rdcSku:=paste(rdc,sku,sep="-")];
vltList[,sku:=NULL];
vltList[,rdc:=NULL];
vltList[,rdcSkuidSupp:=NULL];
setkey(vltList,rdcSku);

rdcskuList = intersect(finalData$rdcSkuid,vltList$rdcSku);
finalData = finalData[rdcSkuid %in% rdcskuList,];
setkey(finalData,rdcSkuid);

testData=vltList[finalData];
testData[,rdcSkuid:=rdcSku];
testData[,rdcSku:=NULL];


###simulation
#testData    <- copy(finalData);

totalDates = as.Date(simulateInitialDate:simulateEndDate);

testData = testData[dt %in% totalDates,];
testData[,amt:=length(unique(dt)),by="rdcSkuid"];
testData = testData[amt==88,];
testData[,amt:=NULL];


totalSkus = unique(testData$rdcSkuid);
fullData = data.table(expand.grid(totalDates,totalSkus));
setnames(fullData,c("dt","rdcSkuid"));
setkeyv(fullData,c("rdcSkuid","dt"));
setkeyv(testData,c("rdcSkuid","dt"));
testData = testData[fullData];

testData=testData[order(rdcSkuid,dt),];

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


testData2=copy(testData)


testData=copy(testData2);

#testData=testData[rdcSkuid %in% c("10-1187576","10-1187581"),];

for(x in totalDates)
{
	
	print(as.Date(x));
	subData<- testData[dt==x, ];   
    subData[, simuInv:=simuInv+AQ];               
    subData[, 		simuSales	:=sales];subData[sales>simuInv, simuSales:=simuInv]; 
	testData[dt==x,	simuSales	:=sales];testData[sales>simuInv, simuSales:=simuInv];
	testData[dt==x, simuInv:=simuInv+AQ]; 
	RQData  <- subData[simuInv+simuOpenpo<LOP95,];
    RQData[, DQ:=LOP95+bp95-(simuOpenpo+simuInv)];
	for(y in RQData$rdcSkuid)
	{
		print(y);
		i = testData[rdcSkuid==y & dt== x ,vltIndex];
		DQ          <- RQData[rdcSkuid==y,]$DQ;
		if(DQ>0) testData[rdcSkuid==y & dt> x ,vltIndex:= vltIndex+1]; 
        realVltList   <- as.integer(unlist(strsplit(RQData[rdcSkuid==y,]$sampleVLT,",")));
		len = length(realVltList);
		realVlt	=	ifelse(i%%len==0,len,realVltList[i%%len]);
        testData[rdcSkuid==y &  dt %in% (1:realVlt+x), simuOpenpo:=simuOpenpo+DQ]; 
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
write.table(testData,"testData_skuDD.txt",sep="\t",row.names=F);

dat= fread("testData.txt",sep="\t");
dat[,inv:=0];
dat[simuInv>0,inv:=1];
dat[,act_inv:=0];
dat[inventory>0,act_inv:=1];

dat_skuDD= fread("testData_skuDD.txt",sep="\t");
dat_skuDD[,inv:=0];
dat_skuDD[simuInv>0,inv:=1];
dat_skuDD[,act_inv:=0];
dat_skuDD[inventory>0,act_inv:=1];

kpi_total_new = dat[,list(simuCr=sum(inv,na.rm=T)/length(inv),simuIto=sum(simuInv,na.rm=T)/sum(simuSales,na.rm=T),Cr=sum(act_inv,na.rm=T)/length(act_inv),Ito=sum(inventory,na.rm=T)/sum(sales,na.rm=T))];
setnames(kpi_total_new,c("simuCr_new","simuIto_new","Cr","Ito"));
kpi_total_skuDD = dat_skuDD[,list(simuCr=sum(inv,na.rm=T)/length(inv),simuIto=sum(simuInv,na.rm=T)/sum(simuSales,na.rm=T),Cr=sum(act_inv,na.rm=T)/length(act_inv),Ito=sum(inventory,na.rm=T)/sum(sales,na.rm=T))];
setnames(kpi_total_skuDD,c("simuCr_sku","simuIto_sku","Cr","Ito"));
kpi_total=subset(cbind(kpi_total_new,kpi_total_skuDD),select=c(3,5,1,4,6,2));
write.table(kpi_total,"kpi_total.csv",sep=",",row.names=F);

kpi = dat[,list(simuCr=sum(inv,na.rm=T)/length(inv),simuIto=sum(simuInv,na.rm=T)/sum(simuSales,na.rm=T),Cr=sum(act_inv,na.rm=T)/length(act_inv),Ito=sum(inventory,na.rm=T)/sum(sales,na.rm=T)),by=rdcSkuid];
kpi_skuDD = dat_skuDD[,list(simuCr=sum(inv,na.rm=T)/length(inv),simuIto=sum(simuInv,na.rm=T)/sum(simuSales,na.rm=T),Cr=sum(act_inv,na.rm=T)/length(act_inv),Ito=sum(inventory,na.rm=T)/sum(sales,na.rm=T)),by=rdcSkuid];

setnames(kpi_skuDD,c("rdcSkuid","simuCr_Skudd","simuIto_Skudd","Cr","Ito"));

write.table(kpi,"kpi.csv",sep=",",row.names=F);
write.table(kpi_skuDD,"kpi_skuDD.csv",sep=",",row.names=F);


setkey(kpi_skuDD,rdcSkuid);
setkey(kpi,rdcSkuid);
kpi_result=kpi_skuDD[kpi][,.(rdcSkuid,Cr,simuCr,simuCr_Skudd,Ito,simuIto,simuIto_Skudd)];
write.table(kpi_skuDD[kpi],"/home/neidian/gaoyun/test/kpi_result.csv",sep=",",row.names=F);