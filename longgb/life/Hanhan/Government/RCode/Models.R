# Title     : Build Models
# Objective : ---
# Created by: yuhan
# Created on: 2017/11/6


# ===============================================================
# =               Reflash the Code
# ===============================================================
rm(list=ls())

# install the package
install.packages('VGAM')
install.packages('spdep')
install.packages('ggplot2')
install.packages('mapproj')
install.packages('plyr')
install.packages('maptools')
install.packages('maps')
install.packages('RColorBrewer')
install.packages('classInt')
install.packages('psych')
install.packages('car')


# import the package
library(VGAM)         
library(maps)         
library(RColorBrewer) 
library(classInt)     
library(spdep)
library(ggplot2)
library(mapproj)
library(plyr)
library(maptools)
library(psych)
library(car)


# define function
getavg <- function(data, n){
    data_c <- c()
    for (i in 1:32){
        this_data <- 0
        if (i < 21){
            for (j in 0:(n-1)){
                this_data <- this_data + data[i+j*31]
            }
            this_data <- this_data / n
        }
        else if (i == 21){
            this_data <- 0
        }
        else if (i > 21){
            for (j in 0:(n-1)){
                this_data <- this_data + data[i-1+j*31]
            }
            this_data <- this_data / n
        }
        data_c <- c(data_c, this_data)
    }
    return(data_c)
}

plotTmpMap <- function(Mapdata, res, this_text, save_name, save_path){
    pdf(paste(save_path, save_name, sep='/'))
    res.palette <- colorRampPalette(c("#C44E52","#FFA455","#EEEED1", "#6AB27B","#4C72B0"), space = "rgb")
    pal <- res.palette(5)
    classes_fx <- classIntervals(res, n=5, style="fixed", fixedBreaks=c(-10,-5,-2,2,5,10), rtimes = 1)
    cols <- findColours(classes_fx,pal)
    par(mar=rep(0,4))
    plot(Mapdata,col=cols, main=this_text, pretty=T, border="grey")
    legend(x="bottom",cex=1,fill=attr(cols,"palette"),bty="n",legend=names(attr(cols, "table")),title=this_text,ncol=5)
    dev.off()
}


# set the path args
save_path <- 'C:/Users/longguangbin/Desktop/Data_Code/results'
data_path <- 'C:/Users/longguangbin/Desktop/Data_Code/data_arange'
map_path <- 'C:/Users/longguangbin/Desktop/Data_Code/data_map'
setwd(save_path)

# get the map data
CHN_adm1 <- readShapePoly(paste(map_path, 'CHN_adm1.shp', sep = '/'))
CHN_adm1_nb <- poly2nb(CHN_adm1, queen=T)
CHN_adm1_mat <- nb2listw(CHN_adm1_nb, style="W", zero.policy=TRUE)

# load my data
mydata <- read.csv(paste(data_path, 'arange_data.csv', sep='/'))
mydata_md <- read.csv(paste(data_path, 'arange_data_md.csv', sep='/'))
mydata_corr <- read.csv(paste(data_path, 'arange_data_corr.csv', sep='/'))


# -------------------------------------------------------
# -    Descriptive statistical analysis
# -------------------------------------------------------
sm <- summary(mydata_corr)
ct <- corr.test(mydata_corr)
print(ct, digits=4)
write.csv(sm, file="summary.csv")
write.csv(ct$r, file="corr_r.csv")
write.csv(ct$p, file="corr_p.csv")


# -------------------------------------------------------
# -    Build Models 
# -------------------------------------------------------
# ### Tobit Regression
# mod.tobit <- vglm(FiscalTransparency ~ MarketizationIndex + ProvincialFinancialStatisticsExpenditure
#              + LocalFiscalTaxRevenue
#              + UrbanPopulationDensity
#              + ManyPerCapitaUrbanRoadArea
#              + TotalInvestmentOfForeignInvestedEnterprises
#              + ManyBasicOilReserves
#              + ManyPermanentPopulation
#              + AverageWageOfStateOwnedUnit
#              + ProvincialFinancialStatisticsIncome
#              + GovernmentScaleRegionalGrossDomesticProduct
#              + ProvincialFinancialStatisticsIncomePre
#              + ManyDeathRate
#              + ManyBirthRate
#              + ManyCountyDivisionNumber
#              + LocalFiscalRevenue
#              + ManyBasicCoalReserves
#              + ManyPrefectureLevelDivisionNumber
#              + EducationLevelOfResidents
#              + ManyPrefectureLevelCity
#              + GovernmentScaleExpenditure
#              + ManyBasicReservesOfNaturalGas
#              , data=mydata, tobit, trace = TRUE)
# coef(mod.tobit, matrix = TRUE)
# summary(mod.tobit)


### OLS regression
mod.lm <- lm(FiscalTransparency ~ MarketizationIndex + ProvincialFinancialStatisticsExpenditure
             + LocalFiscalTaxRevenue
             + UrbanPopulationDensity
             + ManyPerCapitaUrbanRoadArea
             + TotalInvestmentOfForeignInvestedEnterprises
             + ManyPermanentPopulation
             + AverageWageOfStateOwnedUnit
             + ProvincialFinancialStatisticsIncome
             + ProvincialFinancialStatisticsIncomePre
             + ManyDeathRate
             + ManyBirthRate
             + LocalFiscalRevenue
             + EducationLevelOfResidents
             + ManyPrefectureLevelCity
             + GovernmentScaleExpenditurePre
             , data=mydata)
summary(mod.lm)
vif(mod.lm, digits = 3)


# 画散点图
pdf(paste(save_path, "output.pdf", sep='/'))
plot(mydata_corr[, 1:23])
dev.off()



res <- mod.lm$residuals
res_div <- getavg(res, 7)
plotTmpMap(CHN_adm1, res_div, "Residuals from OLS Model")
# Residual Autocorrelation
moran.test(res_div, listw=CHN_adm1_mat, zero.policy=T)


### SAR regression
mod.sar <- lagsarlm(FiscalTransparency ~ MarketizationIndex + ProvincialFinancialStatisticsExpenditure
                    + LocalFiscalTaxRevenue
                    + UrbanPopulationDensity
                    + ManyPerCapitaUrbanRoadArea
                    + TotalInvestmentOfForeignInvestedEnterprises
                    + ManyPermanentPopulation
                    + AverageWageOfStateOwnedUnit
                    + ProvincialFinancialStatisticsIncome
                    + ProvincialFinancialStatisticsIncomePre
                    + ManyDeathRate
                    + ManyBirthRate
                    + LocalFiscalRevenue
                    + EducationLevelOfResidents
                    + ManyPrefectureLevelCity
                    + GovernmentScaleExpenditurePre
                    , data=mydata_md, listw=CHN_adm1_mat, zero.policy=T, tol.solve=1e-12)
summary(mod.sar)
res <- mod.sar$residuals
# Residual Autocorrelation
moran.test(res, listw=CHN_adm1_mat, zero.policy=T)


### SEM regression
mod.sem <- errorsarlm(FiscalTransparency ~ MarketizationIndex + ProvincialFinancialStatisticsExpenditure
                      + LocalFiscalTaxRevenue
                      + UrbanPopulationDensity
                      + ManyPerCapitaUrbanRoadArea
                      + TotalInvestmentOfForeignInvestedEnterprises
                      + ManyPermanentPopulation
                      + AverageWageOfStateOwnedUnit
                      + ProvincialFinancialStatisticsIncome
                      + ProvincialFinancialStatisticsIncomePre
                      + ManyDeathRate
                      + ManyBirthRate
                      + LocalFiscalRevenue
                      + EducationLevelOfResidents
                      + ManyPrefectureLevelCity
                      + GovernmentScaleExpenditurePre
                      , data=mydata_md, listw=CHN_adm1_mat, zero.policy=T, tol.solve=1e-15)
summary(mod.sem)
res <- mod.sem$residuals
# Residual Autocorrelation
moran.test(res, listw=CHN_adm1_mat, zero.policy=T)



