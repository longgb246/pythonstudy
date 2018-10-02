library(data.table)

# ------------------------------------------------------------------
# 1. Read the Data
# ------------------------------------------------------------------
forecastResult  <- fread("forecastInput_origin.csv", integer64="numeric", na.strings=c("NULL","NA", "", "\\N"))


# ------------------------------------------------------------------
# 2. Same handle with the forecastResult
# ------------------------------------------------------------------
forecastResult[, salesFill:=as.integer(ceiling(mean(sales,na.rm=T))), by="rdcSkuid"]
forecastResult[is.na(sales), sales:=salesFill]
forecastResult[is.na(sales), sales:=0]
forecastResult[, curDate:=as.Date(curDate)]
forecastResult      <- unique(forecastResult, by=c("rdcSkuid", "curDate"))
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
# same
simulateInitialDate   <- as.Date("2016-10-01")
forecastResult  <- forecastResult[curDate>=simulateInitialDate,]


# ------------------------------------------------------------------
# 3. Get the missing date, Calculate the missRatio of date
# ------------------------------------------------------------------
forecastResult[, recordNum:=length(curDate), by='rdcSkuid']
forecastResult[, minDate:=min(curDate), by='rdcSkuid']
forecastResult[, maxDate:=max(curDate), by='rdcSkuid']
forecastResult[, dateGap:=maxDate-minDate+1]
forecastResult[, missRatio:=as.numeric(recordNum)/as.numeric(dateGap)]
# 3.1 Only use the data (missRatio>0.7)
forecastResult    <- forecastResult[missRatio>0.7,]
rdcSkuidList    <- unique(forecastResult[missRatio<1,]$rdcSkuid)


# ------------------------------------------------------------------
# 4. Impute the missing data
# ------------------------------------------------------------------
imputeList      <- lapply(rdcSkuidList, function(x){
    print(x)
    subData     <- forecastResult[rdcSkuid==x,]
    subData     <- subData[!is.na(predSd1),]
    subData     <- subData[!is.na(predMean1),]
    fullDateSeq <- seq(unique(subData$minDate), unique(subData$maxDate),1)
    missDate    <- fullDateSeq[!fullDateSeq %in% subData$curDate]
    notSelect   <- c('rdc', 'skuid', 'cid2', 'cid3', 'curDate', 'rdcSkuid', 'minDate', 'maxDate', 'dateGap')
    for (i in 1:length(missDate)){
        y   <- missDate[i]
        imputeDate  <- c(y-1, y-2, y+1, y+2);
        imputeData  <- subData[curDate %in% imputeDate, ];
        imputeDataF <- data.table(rdc=imputeData$rdc[1]);
        for (m in  names(imputeData)[2:length(names(imputeData))]){
            if (!m %in% notSelect){
                imputeDataF[, c(m):=colMeans(subset(imputeData, select=m), na.rm = TRUE)]
            }else if(m == 'curDate'){
                imputeDataF[, c(m):=y]
            }else{
                imputeDataF[, c(m):=subset(imputeData, select=m)[1]]
            }};
        subData     <- rbind(imputeDataF, subData)
        }
    subData <- subData[order(curDate),]
    subData
})
forecastResult    <- rbindlist(imputeList)


# ------------------------------------------------------------------
# 5. Check the data
# ------------------------------------------------------------------
forecastResult[is.na(predSd1),]
forecastResult[is.na(predMean1),]


# ------------------------------------------------------------------
# 6. Delete the columns that simulation don't use.
# ------------------------------------------------------------------
forecastResult[, recordNum:=NULL]
forecastResult[, minDate:=NULL]
forecastResult[, maxDate:=NULL]
forecastResult[, dateGap:=NULL]
forecastResult[, missRatio:=NULL]
forecastResult[, salesFill:=NULL]


# ------------------------------------------------------------------
# 7. Save the data named "forecastInput.csv"
# ------------------------------------------------------------------
write.table(forecastResult, file="forecastInput.csv", sep=",", row.names=F)

