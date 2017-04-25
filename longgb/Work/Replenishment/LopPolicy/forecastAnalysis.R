library(stringr)
library(data.table)

salesData   <- fread("forecastResult.dat", sep="\t",integer64='numeric', na.strings=c("NULL","NA", "", "\\N"))

setnames(salesData, c("rdc", "skuid",  "cid2", "cid3","openpo", "inventory", "predSd", "predMean",
                                             "orderDate", "sales", "curDate"))
salesData[, orderDate:=NULL]
newSalesData   <- salesData
newSalesData[, curDate:=as.Date(curDate)]

write.table(newSalesData, file="newSales.csv",row.names=F)


sdNames <- paste("predSd", 1:28,sep="")

newSalesData[, predSd:=sub("\\[","", predSd)]; newSalesData[, predSd:=sub("\\]","", predSd)];
newSalesData[, predMean:=sub("\\[","", predMean)];newSalesData[, predMean:=sub("\\]","", predMean)]

newSalesData[, c("predSd1", "predSd2", "predSd3", "predSd4", "predSd5", "predSd6", "predSd7",
      "predSd8", "predSd9", "predSd10","predSd11", "predSd12", "predSd13", "predSd14",
     "predSd15", "predSd16", "predSd17", "predSd18", "predSd19", "predSd20", "predSd21",
    "predSd22", "predSd23", "predSd24", "predSd25", "predSd26", "predSd27", "predSd28"):=tstrsplit(predSd, ",")]
newSalesData[, c("predMean1", "predMean2", "predMean3", "predMean4", "predMean5", "predMean6", "predMean7",
      "predMean8", "predMean9", "predMean10","predMean11", "predMean12", "predMean13", "predMean14",
     "predMean15", "predMean16", "predMean17", "predMean18", "predMean19", "predMean20", "predMean21",
    "predMean22", "predMean23", "predMean24", "predMean25", "predMean26", "predMean27", "predMean28"):=tstrsplit(predMean, ",")]

newSalesData[, c("predMean", "predSd"):=NULL]

newSalesData[,rdcSkuid:=paste(rdc, skuid,sep='-')]
write.table(newSalesData, file="forecastInput.csv", row.names=F, sep=",",quote=F)

# newSalesData  <- fread("forecastInput.csv")
simulationKeyList   <- unique(forecastResult$rdcSkuid)
lapply(simulationKeyList, function(x){
    subForecastData     <- newSalesData[rdcSkuid==x,]
    ###calculate the Lop
})





######Filter 

