# Title     : DataProcess.R
# Objective : ---
# Created by: longguangbin
# Created on: 2017/11/4


# install.packages('ape')

# 我已经受不了 R 语言编程了！！！
# R 你需要什么样的数据，我python处理好了在给你。你是大爷，行了吧！！！


# [ 我放弃治疗！！ ]
# read_path <- "D:\\Work\\Codes\\pythonstudy\\longgb\\life\\Hanhan\\Government\\Data"
# read_files <- function(read_path){
#     files_list <- list.files(read_path)
#     all_data <- hash()
#     for (file in files_list) {
#         this_file <- paste(read_path, file, sep = '\\')
#         sub_files <- list.files(this_file)
#         for (sub_file in sub_files){
#             file_split <- unlist(strsplit(sub_file, '[.]'))
#             if (length(file_split) == 2){
#                 this_read_file <- paste(this_file, sub_file, sep = '\\')
#                 print(this_read_file)
#                 data <- read.csv(this_read_file)
#                 
#                 .set(all_data, keys=file_split[1], values=data)
#                 }
#             }
#         print(this_file)
#         }
#     return(all_data)
#     }
# all_data <- read_files(read_path = read_path)


install.packages('ape')
library(ape)


help(ape)


tr <- rtree(30)
x <- rnorm(30)

help(rtree)

## weights w[i,j] = 1/d[i,j]:
w <- 1/cophenetic(tr)

help(cophenetic)

typeof(w)

## set the diagonal w[i,i] = 0 (instead of Inf...):
diag(w) <- 0

Moran.I(x, w)
Moran.I(x, w, alt = "l")
Moran.I(x, w, alt = "g")
Moran.I(x, w, scaled = TRUE) # usualy the same


# observed : the computed Moran’s I.
# expected : the expected value of I under the null hypothesis.
# sd : the standard deviation of I under the null hypothesis.
# p.value : the P-value of the test of the null hypothesis against the alternative hypothesis
#           specified in alternative
# 相关文档：Gittleman, J. L. and Kot, M. (1990) Adaptation: statistics and a null model for estimating phylogenetic effects. Systematic Zoology, 39, 227–241.







rm(list = ls())  # 清除变量


h <- hash()
h['a'] <- c(1,2,3)
choose(h, 'a')


install.packages('VGAM')
library(VGAM)


# Tobit regression
# Tobit distribution
mu <- 0.5; x <- seq(-2, 4, by = 0.01)
Lower <- -1; Upper <- 2.0
integrate(dtobit, lower = Lower, upper = Upper,
mean = mu, Lower = Lower, Upper = Upper)$value +
dtobit(Lower, mean = mu, Lower = Lower, Upper = Upper) +
dtobit(Upper, mean = mu, Lower = Lower, Upper = Upper) # Adds to unity
## Not run:
plot(x, ptobit(x, m = mu, Lower = Lower, Upper = Upper),
type = "l", ylim = 0:1, las = 1, col = "orange",
ylab = paste("ptobit(m = ", mu, ", sd = 1, Lower =", Lower,
", Upper =", Upper, ")"),
main = "Orange is cumulative distribution function; blue is density",
sub = "Purple lines are the 10,20,...,90 percentiles")
abline(h = 0)
lines(x, dtobit(x, m = mu, Lower = Lower, Upper = Upper), col = "blue")
probs <- seq(0.1, 0.9, by = 0.1)
Q <- qtobit(probs, m = mu, Lower = Lower, Upper = Upper)
lines(Q, ptobit(Q, m = mu, Lower = Lower, Upper = Upper),
col = "purple", lty = "dashed", type = "h")
lines(Q, dtobit(Q, m = mu, Lower = Lower, Upper = Upper),
col = "darkgreen", lty = "dashed", type = "h")
abline(h = probs, col = "purple", lty = "dashed")
max(abs(ptobit(Q, m = mu, Lower = Lower, Upper = Upper) - probs)) # Should be 0
endpts <- c(Lower, Upper) # Endpoints have a spike (not quite, actually)
lines(endpts, dtobit(endpts, m = mu, Lower = Lower, Upper = Upper),
col = "blue", lwd = 3, type = "h")
## End(Not run)


# Here, fit1 is a standard Tobit model and fit2 is a nonstandard Tobit model
tdata <- data.frame(x2 = seq(-1, 1, length = (nn <- 100)))
set.seed(1)
Lower <- 1; Upper <- 4 # For the nonstandard Tobit model
tdata <- transform(tdata,
Lower.vec = rnorm(nn, Lower, 0.5),
Upper.vec = rnorm(nn, Upper, 0.5))
meanfun1 <- function(x) 0 + 2*x
meanfun2 <- function(x) 2 + 2*x
meanfun3 <- function(x) 2 + 2*x
meanfun4 <- function(x) 3 + 2*x
tdata <- transform(tdata,
y1 = rtobit(nn, mean = meanfun1(x2)), # Standard Tobit model
y2 = rtobit(nn, mean = meanfun2(x2), Lower = Lower, Upper = Upper),
y3 = rtobit(nn, mean = meanfun3(x2), Lower = Lower.vec, Upper = Upper.vec),
y4 = rtobit(nn, mean = meanfun3(x2), Lower = Lower.vec, Upper = Upper.vec))
with(tdata, table(y1 == 0)) # How many censored values?
with(tdata, table(y2 == Lower | y2 == Upper)) # How many censored values?
with(tdata, table(attr(y2, "cenL")))
with(tdata, table(attr(y2, "cenU")))
fit1 <- vglm(y1 ~ x2, tobit, data = tdata, trace = TRUE)
coef(fit1, matrix = TRUE)
summary(fit1)
fit2 <- vglm(y2 ~ x2, tobit(Lower = Lower, Upper = Upper, type.f = "cens"),
data = tdata, trace = TRUE)
table(fit2@extra$censoredL)
table(fit2@extra$censoredU)
coef(fit2, matrix = TRUE)
fit3 <- vglm(y3 ~ x2, tobit(Lower = with(tdata, Lower.vec),
Upper = with(tdata, Upper.vec), type.f = "cens"),
data = tdata, trace = TRUE)
table(fit3@extra$censoredL)
table(fit3@extra$censoredU)
coef(fit3, matrix = TRUE)
# fit4 is fit3 but with type.fitted = "uncen".
fit4 <- vglm(cbind(y3, y4) ~ x2,
tobit(Lower = rep(with(tdata, Lower.vec), each = 2),
Upper = rep(with(tdata, Upper.vec), each = 2),
byrow.arg = TRUE),
data = tdata, crit = "coeff", trace = TRUE)
head(fit4@extra$censoredL) # A matrix
head(fit4@extra$censoredU) # A matrix
head(fit4@misc$Lower) # A matrix
head(fit4@misc$Upper) # A matrix
coef(fit4, matrix = TRUE)
## Not run: # Plot fit1--fit4
par(mfrow = c(2, 2))
plot(y1 ~ x2, tdata, las = 1, main = "Standard Tobit model",
col = as.numeric(attr(y1, "cenL")) + 3,
pch = as.numeric(attr(y1, "cenL")) + 1)
legend(x = "topleft", leg = c("censored", "uncensored"),
pch = c(2, 1), col = c("blue", "green"))
legend(-1.0, 2.5, c("Truth", "Estimate", "Naive"),
col = c("purple", "orange", "black"), lwd = 2, lty = c(1, 2, 2))
lines(meanfun1(x2) ~ x2, tdata, col = "purple", lwd = 2)
lines(fitted(fit1) ~ x2, tdata, col = "orange", lwd = 2, lty = 2)
lines(fitted(lm(y1 ~ x2, tdata)) ~ x2, tdata, col = "black",
lty = 2, lwd = 2) # This is simplest but wrong!
plot(y2 ~ x2, data = tdata, las = 1, main = "Tobit model",
col = as.numeric(attr(y2, "cenL")) + 3 +
as.numeric(attr(y2, "cenU")),
pch = as.numeric(attr(y2, "cenL")) + 1 +
as.numeric(attr(y2, "cenU")))
legend(x = "topleft", leg = c("censored", "uncensored"),
pch = c(2, 1), col = c("blue", "green"))
legend(-1.0, 3.5, c("Truth", "Estimate", "Naive"),
col = c("purple", "orange", "black"), lwd = 2, lty = c(1, 2, 2))
lines(meanfun2(x2) ~ x2, tdata, col = "purple", lwd = 2)
lines(fitted(fit2) ~ x2, tdata, col = "orange", lwd = 2, lty = 2)
lines(fitted(lm(y2 ~ x2, tdata)) ~ x2, tdata, col = "black",
lty = 2, lwd = 2) # This is simplest but wrong!
plot(y3 ~ x2, data = tdata, las = 1,
main = "Tobit model with nonconstant censor levels",
col = as.numeric(attr(y3, "cenL")) + 2 +
as.numeric(attr(y3, "cenU") * 2),
pch = as.numeric(attr(y3, "cenL")) + 1 +
as.numeric(attr(y3, "cenU") * 2))
legend(x = "topleft", leg = c("censored", "uncensored"),
pch = c(2, 1), col = c("blue", "green"))
legend(-1.0, 3.5, c("Truth", "Estimate", "Naive"),
col = c("purple", "orange", "black"), lwd = 2, lty = c(1, 2, 2))
lines(meanfun3(x2) ~ x2, tdata, col = "purple", lwd = 2)
lines(fitted(fit3) ~ x2, tdata, col = "orange", lwd = 2, lty = 2)
lines(fitted(lm(y3 ~ x2, tdata)) ~ x2, tdata, col = "black",
lty = 2, lwd = 2) # This is simplest but wrong!
plot(y3 ~ x2, data = tdata, las = 1,
main = "Tobit model with nonconstant censor levels",
col = as.numeric(attr(y3, "cenL")) + 2 +
as.numeric(attr(y3, "cenU") * 2),
pch = as.numeric(attr(y3, "cenL")) + 1 +
as.numeric(attr(y3, "cenU") * 2))
legend(x = "topleft", leg = c("censored", "uncensored"),
pch = c(2, 1), col = c("blue", "green"))
legend(-1.0, 3.5, c("Truth", "Estimate", "Naive"),
col = c("purple", "orange", "black"), lwd = 2, lty = c(1, 2, 2))
lines(meanfun3(x2) ~ x2, data = tdata, col = "purple", lwd = 2)
lines(fitted(fit4)[, 1] ~ x2, tdata, col = "orange", lwd = 2, lty = 2)
lines(fitted(lm(y3 ~ x2, tdata)) ~ x2, data = tdata, col = "black",
lty = 2, lwd = 2) # This is simplest but wrong!
## End(Not run)








install.packages('ggplot2')
install.packages('mapproj')



library(spdep)
library(ggplot2)
library(mapproj)
library(plyr)
library(maptools)



##### test Data
url <- url("http://www.people.fas.harvard.edu/~zhukov/Datasets.RData")
load(url)
ls()
rm(list=ls())


data <- election
names(data)

## Contiguity Neighbors
W_cont_el <- poly2nb(data, queen=T)
W_cont_el_mat <- nb2listw(W_cont_el, style="W", zero.policy=TRUE)


###### My test Data
CHN_adm1 <- readShapePoly("C:/Users/longguangbin/Desktop/CHN_adm/CHN_adm1.shp")
CHN_adm1_nb <- poly2nb(CHN_adm1, queen=T)
CHN_adm1_mat <- nb2listw(CHN_adm1_nb, style="W", zero.policy=TRUE)
# plot(chn_poly)
CHN_adm1_pd <- fortify(CHN_adm1)
x <- CHN_adm1@data          #读取行政信息
xs <- data.frame(x, id=seq(0:31)-1)          #总共345行
test_china_map_data  <- join(CHN_adm1_pd, xs, type="full")
# 901197
test_data <- join(test_china_map_data, mydata, type="full")


# ggplot(CHN_adm1, aes(x = long, y = lat, group = group)) +
#      geom_polygon(colour="grey40") +
#      # geom_polygon(colour="grey40") +
#      scale_fill_gradient(low="white",high="steelblue") +
#      coord_map() +
#      theme_grey()
#      # coord_map("polyconic") +
#      theme(              
#           panel.grid = element_blank(),
#           panel.background = element_blank(),
#           axis.text = element_blank(),
#           axis.ticks = element_blank(),
#           axis.title = element_blank()
#           )

mydata <- read.csv('C:/Users/longguangbin/Downloads/CodeData/chn_poly.csv')
test_data <- join(xs, mydata, type="full")

write.csv(CHN_adm1, file="chn_poly.csv")



mod.lm <- lm(Shape_Leng ~ Shape_Area, data=chn_poly)
mod.lm <- lm(Shape_Leng_t2 ~ Shape_Area_t2, data=chn_poly)
summary(mod.lm)


res <- mod.lm$residuals
res.palette <- colorRampPalette(c("red","orange","white", "lightgreen","green"), space = "rgb")
pal <- res.palette(5)
classes_fx <- classIntervals(res, n=5, style="fixed", fixedBreaks=c(-50,-25,-5,5,25,50), rtimes = 1)
cols <- findColours(classes_fx,pal)
par(mar=rep(0,4))
plot(chn_poly,col=cols, main="Residuals from OLS Model", pretty=T, border="grey")
legend(x="bottom",cex=1,fill=attr(cols,"palette"),bty="n",legend=names(attr(cols, "table")),title="Residuals from OLS Model",ncol=5)
dev.off()


# OLS regression
mod.lm <- lm(Bush_pct ~ pcincome + pctpoor, data=data)
summary(mod.lm)
res <- mod.lm$residuals
## Residual Autocorrelation
moran.test(res, listw=W_cont_el_mat, zero.policy=T)


# SAR regression
mod.sar <- lagsarlm(Bush_pct ~ pcincome + pctpoor, data = data, listw=W_cont_el_mat, zero.policy=T, tol.solve=1e-12)
summary(mod.sar)
res <- mod.sar$residuals
## Residual Autocorrelation
moran.test(res, listw=W_cont_el_mat, zero.policy=T)


# SEM regression
mod.sem <- errorsarlm(Bush_pct ~ pcincome, data = data, listw=W_cont_el_mat, zero.policy=T, tol.solve=1e-15)
summary(mod.sem)
res <- mod.sem$residuals
## Residual Autocorrelation
moran.test(res, listw=W_cont_el_mat, zero.policy=T)





# ===============================================================
# =               Reflash the Code
# ===============================================================
rm(list=ls())

# install the package
install.packages('spdep')
install.packages('ggplot2')
install.packages('mapproj')
install.packages('plyr')
install.packages('maptools')
install.packages("maps")
install.packages("RColorBrewer")
install.packages("classInt")


# import the package
library(maps)         
library(RColorBrewer) 
library(classInt)     
library(spdep)
library(ggplot2)
library(mapproj)
library(plyr)
library(maptools)


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

plotTmpMap <- function(Mapdata, res, this_text){
    res.palette <- colorRampPalette(c("#C44E52","#FFA455","#EEEED1", "#6AB27B","#4C72B0"), space = "rgb")
    pal <- res.palette(5)
    classes_fx <- classIntervals(res, n=5, style="fixed", fixedBreaks=c(-10,-5,-2,2,5,10), rtimes = 1)
    cols <- findColours(classes_fx,pal)
    par(mar=rep(0,4))
    plot(Mapdata,col=cols, main=this_text, pretty=T, border="grey")
    legend(x="bottom",cex=1,fill=attr(cols,"palette"),bty="n",legend=names(attr(cols, "table")),title=this_text,ncol=5)
}


# set the path args
root_path <- 'C:/Users/longguangbin/Desktop/Data_Code'
data_path <- paste(root_path, 'arange_data', sep = '/')

# get the map data
CHN_adm1 <- readShapePoly(paste(root_path, 'CHN_adm1.shp', sep = '/'))
CHN_adm1_nb <- poly2nb(CHN_adm1, queen=T)
CHN_adm1_mat <- nb2listw(CHN_adm1_nb, style="W", zero.policy=TRUE)

# load my data
mydata <- read.csv(paste(data_path, 'arange_data.csv', sep='/'))

# -------------------------------------------------------
# -    Build Models 
# -------------------------------------------------------
### OLS regression
mod.lm <- lm(FiscalTransparency ~ MarketizationIndex + ProvincialFinancialStatisticsExpenditure
             + LocalFiscalTaxRevenue
             + UrbanPopulationDensity
             + ManyPerCapitaUrbanRoadArea
             + TotalInvestmentOfForeignInvestedEnterprises
             + ManyBasicOilReserves
             + ManyPermanentPopulation
             + AverageWageOfStateOwnedUnit
             + ProvincialFinancialStatisticsIncome
             + GovernmentScaleRegionalGrossDomesticProduct
             + ProvincialFinancialStatisticsIncomePre
             + ManyDeathRate
             + ManyBirthRate
             + ManyCountyDivisionNumber
             + LocalFiscalRevenue
             + ManyBasicCoalReserves
             + ManyPrefectureLevelDivisionNumber
             + EducationLevelOfResidents
             + ManyPrefectureLevelCity
             + GovernmentScaleExpenditure
             + ManyBasicReservesOfNaturalGas
             , data=mydata)
summary(mod.lm)
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
                    + ManyBasicOilReserves
                    + ManyPermanentPopulation
                    + AverageWageOfStateOwnedUnit
                    + ProvincialFinancialStatisticsIncome
                    + GovernmentScaleRegionalGrossDomesticProduct
                    + ProvincialFinancialStatisticsIncomePre
                    + ManyDeathRate
                    + ManyBirthRate
                    + ManyCountyDivisionNumber
                    + LocalFiscalRevenue
                    + ManyBasicCoalReserves
                    + ManyPrefectureLevelDivisionNumber
                    + EducationLevelOfResidents
                    + ManyPrefectureLevelCity
                    + GovernmentScaleExpenditure
                    + ManyBasicReservesOfNaturalGas
                    , data=mydata, listw=W_cont_el_mat, zero.policy=T, tol.solve=1e-12)
summary(mod.sar)
res <- mod.sar$residuals
# Residual Autocorrelation
moran.test(res, listw=W_cont_el_mat, zero.policy=T)


### SEM regression
mod.sem <- errorsarlm(FiscalTransparency ~ MarketizationIndex + ProvincialFinancialStatisticsExpenditure
                      + LocalFiscalTaxRevenue
                      + UrbanPopulationDensity
                      + ManyPerCapitaUrbanRoadArea
                      + TotalInvestmentOfForeignInvestedEnterprises
                      + ManyBasicOilReserves
                      + ManyPermanentPopulation
                      + AverageWageOfStateOwnedUnit
                      + ProvincialFinancialStatisticsIncome
                      + GovernmentScaleRegionalGrossDomesticProduct
                      + ProvincialFinancialStatisticsIncomePre
                      + ManyDeathRate
                      + ManyBirthRate
                      + ManyCountyDivisionNumber
                      + LocalFiscalRevenue
                      + ManyBasicCoalReserves
                      + ManyPrefectureLevelDivisionNumber
                      + EducationLevelOfResidents
                      + ManyPrefectureLevelCity
                      + GovernmentScaleExpenditure
                      + ManyBasicReservesOfNaturalGas
                      , data=mydata, listw=W_cont_el_mat, zero.policy=T, tol.solve=1e-15)
summary(mod.sem)
res <- mod.sem$residuals
# Residual Autocorrelation
moran.test(res, listw=W_cont_el_mat, zero.policy=T)


