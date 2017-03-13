#### Learing: R base_01 ####
c(1,2,3,4) + c(3,4,5,6)
c(1,2,3,4) + c(1,2)
1:6
6:1
c(1,2,3,4) > c(1,1,2,3)
exp(1)
exp(x = 1)
exp(c(1,2,3,4))
x <- c(1,2,3,4)
log(x)
y <- log(x)
y[1]
y[1:3]
y[-4]          # 减去第四个元素
y > 1
y[y > 1]       # 切片
a <- array(data = x,dim = c(3,4))
a[2,]
a[2,2]
a[,2]
city <- c('bj','sh','cd','sh','bj')
age <- c(23,43,51,32,60)
sex <- c('F','M','F','F','M')
people <- data.frame(city,age,sex)    # 使用data.frame
people[2,3]    # iloc的定位
people[,2]     # 列定位
people$age
people$age>30
people[people$age>30,]    # true的定位
mylist <- list(age=age,city=city,sex=sex)    # 使用list，即为dict
mylist$age    
mylist[1]
mylist[[1]]
class(mylist)    # 查看类型
class(people)
attributes(people)     # 查看属性
str(people)      # str化数据


#### Learing: R base_02 ####
# graphics\lattice\ggplot2
# 散点图
plot(car$dist~car$speed)
# 直线图
plot(car$dist,type = 'l')
# 条状图
plot(car$dist,type = 'h')
# 直方图
hist(car$dist)

library(lattice)
num <- sample(1:3,size = 50,replace = T)
barchart(table(num))
qqmath(rnorm(100))
# 点图，根据Species分组，后在不同图中表示
stripplot(~ Sepal.Length | Species, data = iris,layout=c(1,3))
# 密度图，根据Species分组，后在相同图中表示
densityplot(~ Sepal.Length,groups = Species,data = iris,plot.points=FALSE)
# 箱线图
bwplot(Species ~ Sepal.Length,data = iris)
# 散点图
xyplot(Sepal.Width ~ Sepal.Length,groups = Sepcies,data = iris)
# 矩阵散点图
splom(iris[1:4])
# 直方图
histogram(~ Sepal.Length | Species,data = iris,layout=c(1,3))

library(plyr)
func3d <- function(x,y){
  sin(x^2/2 - y^2/4) * cos(2*x - exp(y))
}
vec1 <- vec2 <- seq(0,2,length=30)
# expend.grid : error
para <- expend.grid(x=vec1,y=vec2)
result6 <- mdply(.data = para,.fun = func3d)
# 三维图
wireframe(V1 ~ x*y,data = result6,scales=list(arrows=FALSE),drape=TRUE,colorkey=F)

library(ggplot2)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy))+geom_point()
print(p)
summary(p)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy),colour=factor(year))
p <- p + geom_point()
print(p)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy),colour=factor(year))
p <- p + stat_smooth()
print(p)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy)) + geom_point(aes(colour=factor(year))) + stat_smooth()
print(p)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy)) + geom_point(aes(colour=factor(year))) + stat_smooth() + scale_color_manual(values = c('blue2','red4'))
print(p)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy)) + geom_point(aes(colour=factor(year))) + stat_smooth() + scale_color_manual(values = c('blue2','red4')) + facet_wrap(~ year,ncol = 1)
print(p)
p <- ggplot(data = mpg,mapping = aes(x=cty,y=hwy)) + geom_point(aes(colour=class,size=displ),alpha=0.5,position = "jitter") + stat_smooth() + scale_size_continuous(range = c(4,10)) + facet_wrap(~ year,ncol = 1) + opts(title='汽车型号与油耗') + labs(y='每加仑告高速公路行驶距离',x='每加仑城市公路行驶距离',size='排量',colour='车型')
# docs.ggplot2.org/current


#### Learing: R base_03 ####
# 读取一个string
x <- readline() 
# 读取一些real
x <- scan()
# 建立连接，cat输出文件
output <- file('output.txt')
cat(1:100,sep = '\t',file = output)
close(output)
# 查看工作目录
getwd()
# 文本输入
input <- file('output.txt')
input_data <- scan(file = input)
close(input)
# 建立连接，cat输出文件
output <- file('output.txt')
writeLines(as.character(1:12),con = output)
input <- readLines(output)
close(output)
close(input)

head(iris)
write.table(iris,file = 'iris.csv',sep = ',')
data <- read.table(file = 'iris.csv',sep = ',')
data <- read.table('clipboard')

library(RODBC)
# 建立连接
channel <- odbcConnect('mysql',uid = user,pwd =password)
channel <- odbcConnect('MySQLSERVER2008',uid = user,pwd =password)
# 查看有哪些表
sqlTables(channel)
# 将customers的数据取出
data <- sqlFetch(channel,'customers')
# 执行sql语句
sqlQuery(channel,'select * from orders')

# 读取excel文件
exceldata <- odbcConnectExcel('c:/iris.xls',readOnly = FALSE)
sqlTables(exceldata)
data <- sqlFetch(exceldata,'sheet1')
data$new <- with(data,Sepal_Length/Sepal_Width)
sqlSave(exceldata,data,tablename = 'sheet3')
odbcClose(exceldata)

# WEB数据获取
install.packages("XML")
library(XML)
url <- 'http://www.google.com/adplanner/static/top1000'
tables <- readHTMLTable(url, stringsAsFactor=FALSE,header = F)
data <- tables[[2]]
res <- with(data,aggregate(V3,list(V3),FUN=length))
res[order(res$x, decreasing = T),][1:10,]


#### Learing: R base_04 ####
install.packages("reshape2")
data(tips,package = 'reshape2')
library(plyr)
head(tips)
aggregate(x=tips$tip,by=list(tips$sex),FUN=mean)
ddply(.data=tips,.variables='sex',.fun=function(x){
  mean(x$tip)
})
ratio_fun <- function(x){
  sum(x$tip)/sum(x$total_bill)
}
ddply(tips,.(sex),ratio_fun)
ddply(.data=tips,        # 拆分计算的对象
       .variables='sex',  # 按照什么变量来拆分
       .fun=ratio_fun)    # 计算的函数
# a:array, l:list, d:dataframe 组合的，第一个表示输入，第二个表示输出，如： aaply-adply-alply，a_ply表示不需要输出。
data <- as.matrix(iris[,-5])
# result4 <- adply(.data=data,
#                  .margins=2,
#                  .fun=function(x){
#                    each(max,min,median,sd)(x)
#                  })
result4 <- adply(.data=data,
                 .margins=2,
                 .fun=function(x){
                   max <- max(x)
                   min <- min(x)
                   median <- median(x)
                   sd <- round(sd(x),2)
                   return(c(max,min,median,sd))
                 })

head(iris)
summary(iris)
model <- function(x){
  lm(Sepal.Length~Sepal.Width, data = x)
}
models <- dlply(.data = iris,.variables = 'Species',.fun = model)
results <- ldply(.data = models,.fun = coef)

# 辅助函数
x <- rnorm(10)
each(max,min,median,sd)(x)
colwise(mean, is.numeric)(iris)
# join ~ merge
# mutate ~ transform
# summarise ~ transform
# arrange ~ order
# rename ~ name
# mapvalues ~ relevel
# count ~ length

load('data/Knicks.rda')
daply(data, .(season), function(x) sum(x$win=='W')/length(x$win))
daply(data, .(season,visiting), function(x) sum(x$win=='W')/length(x$win))
daply(data, .(season), function(x) each(mean,sd)(x$points))
ddply(data, .(season), function(x) colwise(mean,c('points','opp'))(x))
output1 <- ddply(data, .(opponent), function(x) sum(x$win=='W')/length(x$win))
output2 <- ddply(data, .(opponent), function(x) mean(x$points-x$opp))
opponents <- join(output1, output2, by='opponent')
names(opponents)[2:3] <- c('winratio','pointsdiff')

install.packages('ggplot2')
library(ggplot2)
p <- ggplot(opponents, aes(x=pointsdiff,y=winratio)) +
  geom_point(color='red4',size=4) +
  geom_hline(y=0.5,colour='grey20',size=0.5,linetype=2) +
  geom_vline(x=0,colour='grey20',size=0.5,linetype=2) +
  geom_text(data=opponents[opponents$winratio>0.6],
            aes(x=pointsdiff,y=winratio,label=opponent),
            hjust=0.7, vjust=1.4, angle=-30) +
  theme_bw()
print(p)


#### Learing: R base_05 ####
set.seed(1)
x <- seq(1,5,length.out=100)
noise <- rnorm(n=100,mean=0,sd=1)
beta0 <- 1
beta1 <- 2
y <- beta0 + beta1 * x + noise
plot(y ~ x)
model <- lm(y~x)
plot(y~x)
abline(model)
summary(model)
model.matrix(model)
ybar <- mean(y)
yPred <- model$fitted.values
Rsquared <- sum((yPred-ybar)^2)/sum(y-ybar)^2
sqrt(sum(model$residuals^2)/98)
names(model)
model$coef  # 只要能分辨就可以
yConf <- predict(model,interval = 'confidence')
yPred <- predict(model,interval = 'prediction')
plot(y~x,col='gray',pch=16)
yConf <- as.data.frame(yConf)
yPred <- as.data.frame(yPred)
lines(yConf$lwr~x,col='black',lty=3)
lines(yConf$upr~x,col='black',lty=3)
lines(yPred$lwr~x,col='black',lty=2)
lines(yPred$upr~x,col='black',lty=2)
lines(yPred$fit~x,col='black',lty=1)

set.seed(1)
x <- factor(rep(c(0,1)),each=30)
y <- c(rnorm(30,0,1),rnorm(30,1,1))
plot(y~x)
model <- lm(y~x)
model.matrix(model)

set.seed(1)
k <- seq(1,5,length.out = 100)
noise <- rnorm(n=100,mean=0,sd=1)
beta0 <- 1
beta1 <- 2
y <- beta0 + beta1 + x^2 + noise
model <- lm(y~x)
summary(model)
plot(y ~ x)
abline(model)
plot(model$residuals~x)

model2 <- lm(y~x+I(x^2))
summary(model2)

model3 <- update(model2, y~.-x)
summary(model3)
AIC(model, model2, model3)


