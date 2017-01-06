library(data.table)
path <- 'D:/white_list_0.7/'
history <- fread( paste(path,'history1.csv',sep = ''), sep='\t' ,encoding= 'UTF-8', header = T, colClasses = rep("character",5))
forecast <- fread( paste(path,'forecast1.csv',sep = ''), sep='\t' ,encoding= 'UTF-8', header = T, colClasses = rep("character",6))
wl <- fread( paste(path,'solution1_0.88.csv',sep = ''), sep=',' ,encoding= 'UTF-8', header = T,colClasses = rep("character",3))
names(wl) <- c('sku','x','inout')
length(unique(wl$sku))==length(unique(history$item_sku_id))
rwl <- fread( paste(path,'result.txt',sep = ''), sep='\t' ,encoding= 'UTF-8', header = T, colClasses = rep('character',4))
names(rwl) <- c('date','sku','M','C')
rwl <- rwl[,.(sku,M)]
pbwl <- fread( paste(path,'wl_all_in1.csv',sep = ''), sep=',' ,encoding= 'UTF-8', header = T, colClasses = rep('character',2))
pbwl <- pbwl$sku_id
wl2 <- fread( paste(path,'solution1_0.93.csv',sep = ''), sep=',' ,encoding= 'UTF-8', header = T,colClasses = rep("character",3))
names(wl2) <- c('sku','x','inout')
wlin2 <- wl2[inout>=1,sku]

hsku_all <- unique(history[,item_sku_id])#历史数据SKU 
fsku_all <- unique(forecast[,item_sku_id])#预测数据SKU
length(hsku_all)

wlin <- wl[inout>=1,sku]#白名单 
wlout <- wl[inout<1,sku]#不属于白名单
fouth <- setdiff(fsku_all,wl$sku)#预测比历史多出来的SKU

#新品
new_sku <- rbind(history[,.(item_sku_id,new_sku)],forecast[,.(item_sku_id,new_sku)])
new_sku <- new_sku[new_sku==T]
new_sku <- unique(new_sku$item_sku_id)#新品SKU
#new_sku_shelve <- unique(forecast[shelves_dt>=test_date[i],item_sku_id])
wlinnew <- intersect(wlin,new_sku)#白名单中的新品
wloutnew <- intersect(wlout,new_sku)#非白名单中的新品
fouthnew <- intersect(fouth,new_sku)#新SKU中的新品

#白名单满足率0.88
forecast$inout <- forecast[,item_sku_id] %in% wlin
forecast <- forecast[,skuinout:=sum(inout-1),by=parent_sale_ord_id]#预测订单在白名单中
#forecast <- forecast[,new :=sum(new_sku), by= parent_sale_ord_id]#预测订单包含新品
pord <- unique(forecast[,.(parent_sale_ord_id,skuinout)])
pord_in <- pord[skuinout==0,]
#pord_out <- parent_ord_in[skuinout<0,]

#白名单满足率0.93
forecast$inout2 <- forecast[,item_sku_id] %in% wlin2
forecast <- forecast[,skuinout2:=sum(inout2-1),by=parent_sale_ord_id]#预测订单在白名单中
#forecast <- forecast[,new :=sum(new_sku), by= parent_sale_ord_id]#预测订单包含新品
pord2 <- unique(forecast[,.(parent_sale_ord_id,skuinout2)])
pord2_in <- pord2[skuinout2==0,]
#pord_out <- parent_ord_in[skuinout<0,]

#纯band白名单满足率
forecast$pbinout <- forecast[,item_sku_id] %in% pbwl
forecast <- forecast[,skupbinout:=sum(pbinout-1),by=parent_sale_ord_id]#预测订单在白名单中
#forecast <- forecast[,new :=sum(new_sku), by= parent_sale_ord_id]#预测订单包含新品
pord_pb <- unique(forecast[,.(parent_sale_ord_id,skupbinout)])
pord_pb_in <- pord_pb[skupbinout==0,]
#pord_out <- parent_ord_in[skuinout<0,]

#原白名单满足率
rwlin <- rwl[ M=='1',sku]
forecast$rinout <- forecast[,item_sku_id] %in% rwlin
forecast <- forecast[,skurinout:=sum(rinout-1),by=parent_sale_ord_id]
pord_r <- unique(forecast[,.(parent_sale_ord_id,skurinout)])
pord_r_in <- pord_r[skurinout==0,]

#原白名单中剔除未在历史数据中出现的白名单
rwlin_outh <- intersect(rwlin,hsku_all)
forecast$routhinout <- forecast[,item_sku_id] %in% rwlin_outh
forecast <- forecast[,skurouthinout:=sum(routhinout-1),by=parent_sale_ord_id]
pord_routh <- unique(forecast[,.(parent_sale_ord_id,skurouthinout)])
pord_routh_in <- pord_routh[skurouthinout==0,]

#-------------------------------------------------------------------------------------------------------------------


#parent_ord_out <- parent_ord_in[skuinout<0,]
wlreal <- real_wl$sku
diff <- setdiff(wlin,wlreal)
length(diff)
sku_sale_diff <- setorder(sku_sale[item_sku_id %in% diff,],-sku_sale)
plot(sku_sale_diff$sku_sale,cex=0.2,xlab='在老白名单中但不在新白名单中的SKU',ylab='销量')

#在白名单中加入新出现且在老白名单中的SKU
fouth_realwl <- intersect(f_out_h,real_wl$sku)
wlin_fouth_realwl <- union(fouth_realwl,wlin) #名单
forecast$whrinout <- forecast[,item_sku_id] %in% wlin_fouth_realwl
forecast <- forecast[,whrskuinout:=sum(whrinout-1),by=parent_sale_ord_id]
parent_ord_whr <- unique(forecast[,.(parent_sale_ord_id,whrskuinout,new_sku)])
parent_ord_in_whr <- parent_ord_whr[whrskuinout==0,]
length(parent_ord_in_whr$parent_sale_ord_id)



####最大满足订单数量（订单中sku都包含在历史数据中）
forecast$existsku <- forecast[, item_sku_id] %in% as.character( wl0.86$sku)
forecast <- forecast[,ordexistsku:= sum(existsku-1),by =parent_sale_ord_id  ]
parent_ord_exist <- unique(forecast[,.(parent_sale_ord_id, ordexistsku, new_sku)])
parent_ord_exist_in <- parent_ord_exist[ordexistsku==0,]
parent_ord_exist_out <- parent_ord_exist[ordexistsku<0,]
length(parent_ord_exist_in$parent_sale_ord_id)
length(ord_all)#总订单数量


#test_date <- seq(as.Date('2016-09-02'),as.Date('2016-10-21'),by=7)
#i=1


####满足订单数量
forecast$inout <- forecast[,item_sku_id] %in% wlin
forecast <- forecast[,skuinout:=sum(inout-1),by=parent_sale_ord_id]
forecast <- forecast[,new :=sum(new_sku-1), by= parent_sale_ord_id]
parent_ord <- unique(forecast[,.(parent_sale_ord_id,skuinout,new_sku)])
parent_ord_in <- parent_ord_in[skuinout==0,]
parent_ord_out <- parent_ord_in[skuinout<0,]

length(parent_ord_in$parent_sale_ord_id)
head(parent_ord_in)

####最大满足订单数量（订单中sku都包含在历史数据中）
forecast$existsku <- forecast[, item_sku_id] %in% as.character( wl0.86$sku)
forecast <- forecast[,ordexistsku:= sum(existsku-1),by =parent_sale_ord_id  ]
parent_ord_exist <- unique(forecast[,.(parent_sale_ord_id, ordexistsku, new_sku)])
parent_ord_exist_in <- parent_ord_exist[ordexistsku==0,]
parent_ord_exist_out <- parent_ord_exist[ordexistsku<0,]
length(parent_ord_exist_in$parent_sale_ord_id)
length(ord_all)#总订单数量

######
wlin <- wl0.86[inout>=1,sku]
wlout <- wl0.86[inout<1,sku] 
f_out_h <- setdiff(fsku_all,wl0.86$sku)#fsku_all中不属于hsku_all的部分
wlinnew <- intersect(wlin,new_sku)
wloutnew <- intersect(wlout,new_sku)
f_out_h_new <- intersect(f_out_h,new_sku)

sku_sale <- unique(forecast[,sku_sale:=sum(sale_qtty),by=item_sku_id][,.(item_sku_id,sku_sale)])
sku_sale_wlin <- setorder(sku_sale[item_sku_id %in% wlin,],-sku_sale)
sku_sale_wlinnew <- setorder(sku_sale[item_sku_id %in% wlinnew,],-sku_sale)
plot(sku_sale_wlin$sku_sale,cex=0.2,xlab='白名单',ylab='销量')
sku_sale_wlout <- setorder(sku_sale[item_sku_id %in% wlout,],-sku_sale)
sku_sale_wloutnew <- setorder(sku_sale[item_sku_id %in% wloutnew,],-sku_sale)
plot(sku_sale_wlout$sku_sale,cex=0.2, xlab='非白名单',ylab='销量')
sku_sale_foh <- setorder(sku_sale[item_sku_id %in% f_out_h,],-sku_sale)
sku_sale_fohnew <- setorder(sku_sale[item_sku_id %in% f_out_h_new,],-sku_sale)
plot(sku_sale_foh$sku_sale,cex=0.2, xlab='新出现SKU',ylab='销量')
####未满足是由于新品导致的


path <- 'D:/'
real_wl1 <- fread( paste(path,'result.txt',sep = ''), sep='\t' ,encoding= 'UTF-8', header = T, colClasses = c('character','character','character','character'))
str(real_wl1)
names(real_wl1) <- c('date','sku','M','C')

real_wl <- real_wl1[ M=='1',]
forecast$rinout <- forecast[,item_sku_id] %in% real_wl$sku
forecast <- forecast[,rskuinout:=sum(rinout-1),by=parent_sale_ord_id]
parent_ord_r <- unique(forecast[,.(parent_sale_ord_id,rskuinout,new_sku)])
parent_ord_in_r <- parent_ord_r[rskuinout==0,]
length(parent_ord_in_r$parent_sale_ord_id)
#parent_ord_out <- parent_ord_in[skuinout<0,]
wlreal <- real_wl$sku
diff <- setdiff(wlin,wlreal)
length(diff)
sku_sale_diff <- setorder(sku_sale[item_sku_id %in% diff,],-sku_sale)
plot(sku_sale_diff$sku_sale,cex=0.2,xlab='在老白名单中但不在新白名单中的SKU',ylab='销量')

#在白名单中加入新出现且在老白名单中的SKU
fouth_realwl <- intersect(f_out_h,real_wl$sku)
wlin_fouth_realwl <- union(fouth_realwl,wlin) #名单
forecast$whrinout <- forecast[,item_sku_id] %in% wlin_fouth_realwl
forecast <- forecast[,whrskuinout:=sum(whrinout-1),by=parent_sale_ord_id]
parent_ord_whr <- unique(forecast[,.(parent_sale_ord_id,whrskuinout,new_sku)])
parent_ord_in_whr <- parent_ord_whr[whrskuinout==0,]
length(parent_ord_in_whr$parent_sale_ord_id)
