library(lubridate)
library(data.table)

vltData <- fread("vltData.csv",integer64='numeric',sep="\t",na.strings=c("NULL","NA", "", "\\N"))
#setnames(vltData, c("pur_bill_id","sku_id","item_third_cate_cd","supp_brevity_cd","int_org_num","store_id","create_tm","complete_dt","into_wh_tm","t_6","new_vlt","flag","isautopo"))
setnames(vltData, c("pur_bill_id","sku_id","item_third_cate_cd","supp_brevity_cd","int_org_num","store_id","create_tm","complete_dt","into_wh_tm","t_6","new_vlt","int_new_vlt","flag","isautopo"))

#####selection a third category for example
vltData <- vltData[item_third_cate_cd==7057,]

###remove one sku with multiple vendor 
vltData[,skuVendorNum:=length(unique(supp_brevity_cd)), by=c("sku_id", "int_org_num")]
vltData <- vltData[skuVendorNum==1  & (new_vlt >0.5 & new_vlt<=60),]

vltData[, t_6:=ymd_hms(t_6)];   vltData[, curDate:=as.Date(t_6)]
### training data date<as.Date("2016-10-01") Testing data        > as.Date("2016-10-01")
vltData[, mark:='train'];       vltData[curDate>=as.Date("2016-10-01"), mark:='test']

vltData[, intVlt:=round(new_vlt)]

selectNames <- c("sku_id", "item_third_cate_cd", "supp_brevity_cd", "int_org_num","intVlt")
vltTrain    <- subset(vltData, mark=='train',select=selectNames)
vltTrain[, metaFlag:=paste(item_third_cate_cd, supp_brevity_cd, int_org_num,sep="-")]

expandList  <- lapply(unique(vltTrain$metaFlag), function(x){
        print(x);
        subData     <- subset(vltTrain,metaFlag==x, select=c("sku_id", "intVlt"));
        subData$count   <- 1;
	skuList	    <- unique(subData$sku_id)	;
        intVltList  <- unique(subData$intVlt);
        expandResult    <- data.table(expand.grid(skuList, intVltList));
        setnames(expandResult, c("sku_id", "intVlt"));
        setkeyv(expandResult,  c("sku_id", "intVlt"));
        setkeyv(subData, c("sku_id", "intVlt"));
        result      <- subData[expandResult];
        result[is.na(count), count:=0];
        result$metaFlag <- x;
        result;
})

expandData  <- rbindlist(expandList)

#####The cid3 distribution
expandData[,cid3RecordCount:=sum(count), by=metaFlag];
expandData[,cid3VltCount   :=sum(count), by=c("metaFlag", "intVlt")];
expandData[,skuRecordCount:=sum(count), by=c("metaFlag", "sku_id")];
expandData[,skuVltCount   :=sum(count), by=c("metaFlag", "sku_id", "intVlt")];
expandData[,cid3DD:= cid3VltCount*1.0/cid3RecordCount];
expandData[,skuDD := skuVltCount *1.0/skuRecordCount];

recordThreshold <- 10;
expandData[, newDD:=cid3DD];
expandData[skuRecordCount>=recordThreshold, newDD:=skuDD];

write.table(expandData, file="vltDDInput.csv", sep=",", row.names=F)

