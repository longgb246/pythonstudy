library(data.table)
############################################
####Load the data and preprocessing
################################################
vltDD           <- fread("vltDDInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
forecastResult  <- fread("forecastInput.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))
bpData          <- fread("bp.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))

### preprocessing
####The forecast result checking
forecastResult      <- forecastResult[rdcSkuid %in% intersectRdcSkuid,]
forecastResult$index    <- 1:nrow(forecastResult)
#duplicatedCheck     <- forecastResult[,list(recordCount=length(index)), by=c("rdcSkuid", "curDate")]
#duplicatedCheck[,table(recordCount)]
forecastResult      <- unique(forecastResult, by=c("rdcSkuid", "curDate"))

###The vlt data preprocessing
vltDD[, rdc:=tstrsplit(metaFlag,'-')[3]]
vltDD[, supplier:=tstrsplit(metaFlag,'-')[2]]
vltDD[,rdcSkuid:=paste(rdc, sku_id, sep='-')]
vltDD   <- unique(vltDD, by=c("rdcSkuid", "intVlt"))

intersectRdcSkuid   <- intersect(vltDD$rdcSkuid, forecastResult$rdcSkuid)
vltDD               <- vltDD[rdcSkuid %in% intersectRdcSkuid,]
forecastResult      <- forecastResult[rdcSkuid %in% intersectRdcSkuid,]

## bp data preprocess
bpData[, rdcSkuid:=paste(int_org_num, sku_id, sep='-')]
bpData[, dt:=as.Date(dt)]
bpData$index    <- 1:nrow(bpData)
duplicatedCheck <- bpData[,list(recordCount=length(index)), by=c("rdcSkuid", "dt")]

bpData  <- bpData[,.(rdcSkuid, dt, bp)]

######################################################3
#### Simulation framework
###One. sales data preprocessing
######################################################3
simulateInitialDate   <- as.Date("2016-10-01")
simulationData  <- forecastResult[curDate>=simulateInitialDate,]

####make the intial date get the data
beginRdcSkuidList   <- simulationData[curDate==simulateInitialDate, unique(rdcSkuid)]
simulationData  <- simulationData[rdcSkuid %in% beginRdcSkuidList,]

simulationData[, c("sumSales","sumInventory"):=list(sum(sales,na.rm=T), sum(inventory,na.rm=T)), by=rdcSkuid]
#The initial inventory is zero
removeRdcSku1   <- unique(simulationData[curDate==simulateInitialDate & is.na(inventory),]$rdcSkuid)
#The initial forecast result is zero
removeRdcSku2   <- unique(simulationData[curDate==simulateInitialDate & is.na(predMean1), ]$rdcSkuid)
#The total inventory is zero
removeRdcSku3   <- unique(simulationData[sumInventory==0,]$rdcSkuid)
# The total sales is zero
removeRdcSku4   <- unique(simulationData[sumSales==0,]$rdcSkuid)
removeRdSkuList     <- unique(c(removeRdcSku1, removeRdcSku2, removeRdcSku3,removeRdcSku4))

simulationData  <- simulationData[!rdcSkuid %in% removeRdSkuList,]
simulationData  <- simulationData[order(rdcSkuid, curDate),]

#####For every rdcSkuid pair do simulation
simulationKeyList   <- unique(simulationData$rdcSkuid)
LOPCalculationList <- lapply(simulationKeyList, function(x){
                          print(x)
    subVltData          <- vltDD[rdcSkuid==x & newDD >0.0, ]
    subForecastData     <- simulationData[rdcSkuid==x,]

    #sample vtl from the vlt distribution
    if(nrow(subVltData)>1)
    { subForecastData$sampleVlt  <- sample(subVltData$intVlt, size=nrow(subForecastData), prob=subVltData$newDD, replace=T)}
    else{subForecastData$sampleVlt  <- subVltData$intVlt}
    ###The left part of the VLT, the expectation
  	LOPMeanList     <- sapply(subVltData$intVlt, function(y){
        ###The left part of the LOP
        subProb     <- subVltData[intVlt==y,]$newDD
        vltKey      <- ifelse(y<=28, y, 28)
        expectationNames    <- paste('predMean', 1:vltKey, sep='')
        subMeanResult   <- rowSums(subset(subForecastData,select=expectationNames))*subProb
        subMeanResult   <- subMeanResult*y/vltKey
        subMeanResult
})
     subForecastData$LOPMean <- rowSums(LOPMeanList)
    ###The first part of the condition vairance, expectation of the variance
     expectationOfVarianceList         <- sapply(subVltData$intVlt, function(y){
        ###The left part of the LOP
        subProb     <- subVltData[intVlt==y,]$newDD
        vltKey      <- ifelse(y<=28, y, 28)
        expectationNames    <- paste('predMean', 1:vltKey, sep='')
        subMeanResult   <- rowSums(subset(subForecastData,select=expectationNames))
        subMeanResult   <- subMeanResult*y/vltKey
        subMeanResult   <- subMeanResult-subForecastData$LOPMean
        conditionVariance1  <- subMeanResult^2*subProb
        conditionVariance1
})
     subForecastData$conditionVariance1 <- rowSums(expectationOfVarianceList)

      ###The second part of the condition vairance, expectation of the variance
    varianceOfConditionMeanList    <- sapply(subVltData$intVlt, function(y){
        vltKey      <- ifelse(y<=28, y, 28)
        sdNames         <- paste('predSd', 1:vltKey, sep='')
        subSdResult     <- rowSums(subset(subForecastData,select=sdNames)^2)
        subSdResult     <- subSdResult*y/vltKey
        subSdResult
})
    subForecastData$conditionVariance2  <- rowSums(varianceOfConditionMeanList)
    subForecastData[, LOPSd:=sqrt(conditionVariance1+conditionVariance2)]

})

LOPCalculationData <- rbindlist(LOPCalculationList)


#3################################################################
# The BP logic 
#3################################################################
### check if the LOPCalculationData is redundant
LOPCalculationData  <- unique(LOPCalculationData, by=c("rdcSkuid", "curDate"))
LOPCalculationData[, curDate:=as.Date(curDate)]

setkeyv(LOPCalculationData, c("rdcSkuid", "curDate"))
setkeyv(bpData, c("rdcSkuid", "dt"))

joinData    <- bpData[LOPCalculationData]

###
joinData[, fillBp:=mean(bp, na.rm=T), by=rdcSkuid]
joinData[is.na(bp), bp:=as.integer(fillBp)]

bpList  <- unique(joinData$bp)
bpCalculationList   <- lapply(bpList, function(x){
    subData     <- joinData[bp==x, ]
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

save(LOPCalculationData,bpCalculationData, file="temp.rda")

finalData   <- subset(bpCalculationData, select=c(1:10, 70,71,74,76,77))

#load("temp.rda")
###calculate the 0.95 quantile
finalData[, LOP95:=round(qnorm(0.95, LOPMean, LOPSd))]
finalData[, bp95:=round(qnorm(0.95, bpMean, bpSd))]

##########################################################
### Do simulation
### (1) Initial value of the simulation data
###########################################################
##### Initialized the data
simulateInitialDate   <- as.Date("2016-10-01")

finalData[,c("AQ", "simuOpenpo", "simuInv"):=0]
finalData[dt == simulateInitialDate, simuInv:=inventory]
finalData[is.na(sales), sales:=0]

###simulation
testData    <- copy(finalData)
###vectorized by date
sapply(sort(unique(testData$dt)), function(x){
           print(x)
    subData      <- testData[dt==x, ]   #The select date
    subData[, simuInv:=simuInv+AQ]               #simuInvi = simuInvi + AQi;
    subData[,simuSales:=sales];  subData[sales>simuInv, simuSales:=simuInv]  # min(simuInvi, salesi);

    RQData  <- subData[simuInv+simuOpenpo<LOP95,]   #for those need do replensihment
    RQData[, DQ:=LOP95+bp95-(simuOpenpo+simuInv)]
    ##add simuopenpo for the later sampleVlt day
    sapply(RQData$rdcSkuid, function(y){
               print(y)
        DQ          <- RQData[rdcSkuid==y,]$DQ
        sampleVLT   <- RQData[rdcSkuid==y,]$sampleVlt
        testData[rdcSkuid==y &  dt %in% (1:sampleVLT+x), simuOpenpo:=simuOpenpo+DQ] #for 1:alpha 
        testData[rdcSkuid==y & dt %in% (sampleVLT+1+x), AQ:=AQ+DQ]
        #testData[rdcSkuid==y & dt %in%(x+1), simuInv:=tmpInv-tmpSales]
    })
    
    ##The simu refresh
    subData[, simuInvNext:=simuInv-simuSales]
    subData[, dt:=dt+1]
    subData <- subset(subData, select=c("rdcSkuid", "dt", "simuInvNext"))
    
    setkeyv(subData, c("rdcSkuid", "dt"))
    setkeyv(testData, c("rdcSkuid", "dt"))
    testData    <- subData[testData]
    testData[!is.na(simuInvNext), simuInv:=simuInvNext]
})

