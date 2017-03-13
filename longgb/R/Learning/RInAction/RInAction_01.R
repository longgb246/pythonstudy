#### Charpter 02 数据结构 ####

# （1）向量：要是有一个元素是 string，则别的元素都是 string
a <- c(1,2,23,"aa"); a[3]

# （2）矩阵：默认按照列填充
y <- matrix(1:20,nrow = 5,ncol = 4)
y <- matrix(1:20,nrow = 5,ncol = 4,byrow = TRUE)
rnames <- c("R1","R2")
cnames <- c("C1","C2")   # dimnames(列名，行名)
mymatrix <- matrix(1:4, nrow = 2, ncol = 2, dimnames = list(rnames, cnames)); mymatrix
mymatrix[,2]
mymatrix[2,]

# （3）数组
dim1 <- c("A1","A2")
dim2 <- c("B1","B2","B3")
dim3 <- c("C1","C2","C3","C4")
z <- array(1:24,c(2,3,4),dimnames = list(dim1,dim2,dim3)) 

# （4）数据框
patientID <- c(1,2,3,4)
age <- c(25,34,28,52)
diabetes <- c("TYPE1","TYPE2","TYPE1","TYPE1")
status <- c("Poor","Improved","Excelent","Poor")
patientdata <- data.frame(patientID,age,diabetes,status);patientdata
patientdata[,1:2]
patientdata[,c("diabetes","status")]
patientdata$patientID
# 常用函数
class(patientdata)
names(patientdata)
length(patientdata)
length(patientdata$patientID)
dim(patientdata)
aa <- cbind(patientdata$patientID,patientdata$age)
aa <- rbind(patientdata$patientID,patientdata$age)
tail(patientdata)
# 列联表
table(patientdata$diabetes,patientdata$status)
head(mtcars)
mtcars
with(mtcars,{
  stat <- summary(mpg,disp,wt)
  stat2 <<- summary(mpg,disp,wt)   # 使用<<-可以在 with 外保存赋值
  plot(mpg~disp)
  stat
})

# （5）因子（暂时过）

# （6）列表
q <- "My First List"
h <- c(25,26,18,39)
j <- matrix(1:10, nrow = 5)
k <- c("one","two","three")
mylist <- list(title=g,age=h,j,k)
mylist[[2]]
mylist[["age"]]

mydata <- data.frame(age=numeric(0),gender=numeric(0),weight=numeric(0))
mydata <- edit(mydata)
fix(mydata)

# 文件 IO
# 读取csv，txt文件
file_path <- "F:\\forread\\data.csv"
mydataframe <- read.table(file_path, sep = ",", header = TRUE)
head(mydataframe)
testa <- read.csv(file_path)
head(testa)
# # 读取excel，32位有效
# install.packages("RODBC")
# library(RODBC)
# file_path2 <- "F:\\forread\\combine.xlsx"
# channel <- odbcConnectExcel(file_path2)
# mydataframe <- sqlFetch(channel, "sheet1")
# 读取excel，64位有效
install.packages("XLConnect")
library(XLConnect)
channel <- loadWorkbook(file_path2)
mydataframe <- readWorksheet(channel, 'sheet1')
head(mydataframe)
# # 连接mysql，连接成功，查询未成功
# library(RODBC)
# channel_mysql <- odbcConnect('mysql',uid = "root",pwd = "lgb1937456")
# sqlTables(channel_mysql)
# data <- sqlFetch(channel_mysql,'')
# sqlQuery(channel_mysql,'use mysql;select * from user;')


#### Charpter 03 图形初阶 ####
# 保存图片
pdf("C:\\Users\\longgb246\\Desktop\\test.pdf")
with(mtcars,{
  plot(wt,mpg)
  abline(lm(mpg~wt))
  title("Regression of MPG on Weight")
})
dev.off()
# 图片选项
dose <- c(20,30,40,45,60)
drugA <- c(16,20,27,40,60)  # plot 属于 graphics
plot(dose,drugA,type = "b") # "b"表示画点和线
plot(drugA~dose,type="b")
# 线、点选项
opar <- par(no.readonly = TRUE)  # 复制当前图形参数
par(lty=2,pch=17)  # 虚线lty=2，点实心三角形pch=17
plot(dose,drugA,type = "b")
par(opar)  # 还原之前设置
# 参数：lty、pch、cex（缩放大小）、lwd（线宽）
n <- 10
mycolors <- rainbow(n)  # 生成连续10个彩虹色
pie(rep(1,n),labels = mycolors,col = mycolors)
mygray <- gray(0:n/n)  # 生成连续的灰度色
pie(rep(1,n),labels = mygray,col = mygray)

# <P75>


#### Charpter 04 基本数据管理 ####
# （1）数据框
manager <- 1:5
date <- c("10/24/08","10/28/08","10/1/08","10/12/08","5/1/09")
country <- c("US","US","UK","UK","UK")
gender <- c("M","F","F","M","F")
age <- c(32,45,25,39,99)
q1 <- c(5,3,3,3,2)
q2 <- c(4,5,5,3,2)
q3 <- c(5,2,5,4,1)
q4 <- c(5,5,5,NA,2)
q5 <- c(5,5,2,NA,1)
leadership <- data.frame(manager,date,country,gender,age,q1,q2,q3,q4,q5,stringsAsFactors = FALSE)
# 赋值新变量
mydata <- data.frame(x1=c(2,2,6,4),x2=c(3,4,2,8))
mydata$sumx <- mydata$x1 + mydata$x2  
# 切片索引赋值
leadership$age[leadership$age==99] <- NA  
leadership$agecat[leadership$age>75] <- "Elder"
leadership$agecat[leadership$age>=55 & leadership$age<=75] <- "Middle Aged"
leadership$agecat[leadership$age<55] <- "Young"
# 列名重命名
a <- names(leadership)
a[2] <- "NoAct_testDate"
names(leadership)[2] <- "testDate"  
names(leadership)[2] <- "date"  
names(leadership)[6:10] <- c("item1","item2","item3","item4","item5")
install.packages("reshape")
library(reshape)
leadership <- rename(leadership,
                     c(manager="managerID",date="testDate"))
# 缺失值
y <- c(1,2,3,NA)
is.na(y)
is.na(leadership[,6:10])
newdata <- na.omit(leadership)
# merge
total <- merge(dataframeA,dataframeB,by=c("ID","Country"))
total <- cbind(A,B)
total <- rbind(A,B)
# 剔除变量
newdata <- leadership[order(leadership$age),]  # order 排序
order(leadership$gender,-leadership$age)  # sort 按照2个顺序排序,age 按照降序排
newdata <- leadership[order(leadership$gender,-leadership$age),]
myvars <- paste("q",1:5,sep = "")  # paste 函数
myvars <- names(leadership) %in% c("q1","q2")
!myvars
newdata <- leadership[,!myvars]
newdata <- leadership[,-c(8,9)]
leadership$q3 <- leadership$q4 <- NULL
# 选入变量
which(leadership$gender=="M" & leadership$age>30)  # 选出TRUE的位置
newdata <- leadership[which(leadership$gender=="M" & leadership$age>30),]
newdata <- leadership[leadership$gender=="M" & leadership$age>30,]
# attach
attach(leadership)
newdata <- leadership[which(gender=="M" & age>30),]
detach(leadership)
# 变量子集筛选
newdata <- subset(leadership, age>=35 | age<24,select = c(q1,q2,q3,q4))
newdata <- subset(leadership, gender=="M" & age>25,select = gender:q4)
# 随机抽样
nrow(leadership)  # leadership 的列数
mysample <- leadership[sample(1:nrow(leadership), 3,replace = FALSE),]  # 无放回抽样
# 使用sql操作数据框
install.packages("sqldf")
library(sqldf)
newdf <- sqldf("select * from mtcars where carb=1 order by mpg", row.names=TRUE)

# （2）日期值
mydate <- as.Date(c("2007-06-22","2004-02-13"))
strdate <- as.Date(c("01/05/1965","08/16/1975"),"%m/%d/%Y")
today <- Sys.Date() # 当前日期
format(today, format="%B %d %Y")
format(today, format="%A")
startdate <- as.Date("2004-02-13")
enddate <- as.Date("2011-01-22")
days <- enddate - startdate
dob <- as.Date("1993-02-09")
difftime(today, dob, units = "weeks")
difftime(today, dob, units = "days")
strDates <- as.integer(as.character(days))
# 数据框的日期改变
leadership$date <- as.Date(leadership$date, "%m/%d/%y")
startdate <- as.Date("2009-01-01")
enddate <- as.Date("2009-10-31")
newdata <- leadership[which(leadership$date>=startdate & leadership$date<=enddate),]


#### Charpter 05 高级数据管理 ####
# （1）数学函数
x <- 1:10
abs(x)
sqrt(x)
ceiling(x)
floor(x)
trunc(x)  # 向0方向截取整数
round(x,digits = n)  # 四舍五入，指定小数精确
cos(x);sin(x);tan(x)
acos(x);asin(x);atan(x)
cosh(x);sinh(x);tanh(x)
acosh(x);asinh(x);atanh(x)
log(x,base = n)  # 取以n为底的对数
log(x)  # 自然对数
exp(x)

# （2）统计函数
mean(x)
median(x)
sd(x)  # 标准差
var(x)
mad(x)  # 绝对中位差
quantile(x,probs = c(0.3,0.84))  # 求分位数
range(x)  # 范围，值域
sum(x)
min(x)
max(x)
scale(x,center = TRUE,scale = TRUE)  # center中心化，标准化（center = TRUE,scale = TRUE）
scale(x)  # 默认下，是0-1标准化
scale(x)*SD + M

# （3）概率函数
# d：密度函数；p：分布函数；q：分位数函数；r：生成随机数函数
beta  
unif  # 均匀分布
binom  # 二项分布
nbinom  # 负二项分布
multinom  # 多项分布
norm  # 正态分布
lnorm  # 对数正态分布
pois  # 泊松分布
weibull  # Weibull分布
wilcox  # Wilcoxon秩和分布
signrank  # Wilcoxon符号秩分布
cauchy  # 柯西分布
chisq  # 卡方分布
exp  # 指数分布
t  # T分布
f  # F分布
gamma  # Gamma分布
geom  # 几何分布
hyper  # 超几何分布
logis  # Logistic分布
# 例子
x <- pretty(c(-3,3),30)  # 把(-3,3)分成30段
y <- dnorm(x)
plot(x,y,type = "l",xlab = "NormalDeviate",ylab = "Density",yaxs="i")
plot(x,y,type = "l",xlab = "NormalDeviate",ylab = "Density")
pnorm(1.96)
qnorm(0.9,mean = 500,sd = 100)
rnorm(50,mean = 50,sd = 10)
set.seed(1234)  # 用于结果重现，分享
runif(5)
set.seed(1234)
runif(5)
# 生成多元正态分布
library(MASS)
options(digits = 3)  # 设置小数
set.seed(1234)
mean <- c(230.7,146.7,3.6)
sigma <- matrix(c(15360.8,6721.2,-47.1,6721.2,4700.9,-16.5,-47.1,-16.5,0.3),nrow = 3,ncol = 3)
mydata <- mvrnorm(500, mean, sigma)
mydata <- as.data.frame(mydata)
names(mydata) <- c("y","x1","x2")
head(mydata)

# （4）字符处理函数
nchar("ad")  # 字符长度
x <- "abcdef"
substr(x,2,4)
substr(x,2,4) <- "1111111" # 只替换2-4的3个，多的不替换进去
grep("A",c("asd","sdA","fds"))  # 找出第几个里面有前面的正则
sub("\\s",".","Hello World!")  # 替换，使用"."来替换\\s"（空格），在最后的字符串中
strsplit("a,bc",",")  # 按照","，把"a,bc"拆分
paste("x",1:3,sep = "_")
paste(c("x","y"),c("bb","aa"),sep = "")
toupper("xs")
tolower("DAD")

# （5）实用函数
seq(1,10,2)  # seq(from, to, by)
x <- pretty(c(1,3),3)  # 生成3+1个间距 
rep("x",3)  # 重复3次
cstra <- "aaa"

# （6）函数用于矩阵
c <- matrix(runif(20),nrow = 5)
log(c)
apply(c, 1, mean)  # 1:对行
apply(c, 2, mean)  # 2:对列
# 数据处理例子
options(digits = 2)
Student <- c("John Davis","Angela Williams","Bullwinkle Moose","David Jones","Janice Markhammer","Chervl Cushina","Reuven Ytzrhak","Greg Knox","Joel England","Mary Rayburn")
Math <- c(502,600,412,358,495,512,410,625,573,522)
Science <- c(95,99,80,82,75,85,80,95,89,86)
English <- c(25,22,18,15,20,28,15,30,27,18)
roster <- data.frame(Student,Math,Science,English,stringsAsFactors = FALSE)
z <- scale(roster[,2:4])
score <- apply(z, 1, mean)
# roster$score <- score
roster <- cbind(roster, score)
y <- quantile(score, c(0.8,0.6,0.4,0.2))
roster$grade[score>=y[1]] <- "A"
roster$grade[score<y[1]&score>=y[2]] <- "B"
roster$grade[score<y[2]&score>=y[3]] <- "C"
roster$grade[score<y[3]&score>=y[4]] <- "D"
roster$grade[score<y[4]] <- "F"
name <- strsplit(roster$Student," ")
lastname <- sapply(name, "[", 2)
firstname <- sapply(name, "[", 1)
roster <- cbind(firstname, lastname, roster[,-1])
roster[order(lastname, firstname),]

# （7）控制流
# for
for (i in 1:10) {
  print("Hello")
}
# while
i = 10
while (i>0) {
  print("Hello")
  i <- i-1
}
# if - else if - else
if (2<1) {
  print("Yes")
}else if(3<2){
  print("A_No")
}else{
  print("A_Yes")
}
# switch
feelings <- c("sad", "afraid")
for (i in feelings) {
  print(
    switch (i,
      happy = "I am glad you are happy",
      afraid = "There is nothing to fear",
      sad = "Cheer up",
      angry = "Calm down now"
    )
  )
}
# function
mystats <- function(x, parametric=TRUE, print=FALSE){
  if(parametric){
    center <- mean(x); spread <- sd(x)
  }else{
    center <- median(x); spread <- mad(x)
  }
  if(print & parametric){
    cat("Mean=",center, "\n","SD=", spread,"\n")
  }else if(print & parametric){
    cat("Median=",center, "\n","MAD=", spread,"\n")
  }
  result <- list(center=center, spread=spread)
  return(result)
}
x <- rnorm(500)
y <- mystats(x)
y <- mystats(x,print = TRUE)
mydate <- function(type="long") {
  switch (type,
    long = format(Sys.time(),"%A %B %d %Y"),
    short = format(Sys.time(), "%m-%d-%y"),
    cat(type, "is not a recognized type\n")
  )
}
mydate("long")
mydate("short")

# （7）数据框
cars <- mtcars[1:5,1:4]
t(cars)  # 转置
aggdata <- aggregate(mtcars,by = list(mtcars$cyl,mtcars$gear),FUN = mean,na.rm=TRUE)  # 整合函数
# 重塑数据 - 整合数据【极其强大】
ID <- c(1,1,2,2)
Time <- c(1,2,1,2)
x1 <- c(5,3,6,2)
x2 <- c(6,5,1,4)
mydata <- data.frame(ID, Time, x1, x2)
library(reshape)
md <- melt(mydata,id=c("ID","Time"))
cast(md, ID+Time~variable)
cast(md, ID+variable~Time)
cast(md, ID~variable+Time)
cast(md, ID~variable, mean)
cast(md, Time~variable, mean)
cast(md, ID~Time, mean)


#### Charpter 06 基本图形 ####
install.packages("vcd")
library(vcd)
# （1）条形图 <barplot>
counts <- table(Arthritis$Improved)
barplot(counts, main = "Simple Bar Plot", xlab = "Improvement", ylab = "Frequency")
barplot(counts, main = "Horizontal Bar Plot", xlab = "Frequency", ylab = "Improvement", horiz = TRUE)
# 堆砌条形图 <barplot>
counts <- table(Arthritis$Improved, Arthritis$Treatment)
barplot(counts, main = "Stacked Bar Plot", xlab = "Treatmet", ylab = "Frequency", col = c("red", "yellow", "green"), legend = rownames(counts), beside = TRUE)
barplot(counts, main = "Stacked Bar Plot", xlab = "Treatmet", ylab = "Frequency", col = c("red", "yellow", "green"), legend = rownames(counts))
barplot(counts, main = "Stacked Bar Plot", xlab = "Treatmet", ylab = "Frequency", col = rainbow(3))
states <- data.frame(state.region, state.x77)
means <- aggregate(states$Illiteracy, by=list(state.region), FUN=mean)
sort(means$x)
order(means$x)
means <- means[order(means$x),]
barplot(means$x, names.arg = means$Group.1)
barplot(means$x, names = means$Group.1)
title("Mean Illiteracy Rate")
# 微调 <par(las=2)>
par(mar=c(5,8,4,2))
par(las=2)  # 使得纵轴的文字横着
counts <- table(Arthritis$Improved)
barplot(counts, main = "Treatment Outcome", horiz = TRUE,names.arg = names(counts), cex.names = 0.8)

# （2）棘状图 <spine>
counts <- table(Arthritis$Treatment, Arthritis$Improved)
spine(counts, main = "Spinogram Example")

# （3）饼图 <pie\pie3D>
par(mfrow=c(2,2))  # 布局一个2*2的布局
slices <- c(10,12.4,16,8)
lbls <- c("US","UK","Australia","Germany","France")
pie(slices, labels = lbls, main = "Simple Pie Chart")
pct <- round(slices/sum(slices)*100)
lbls2 <- paste(lbls," ",pct,"%",sep = "")
pie(slices, labels = lbls2, col = rainbow(4), main = "Pie Chart")
install.packages("plotrix")
library(plotrix)
pie3D(slices, explode = 0.1, main = "Pie3d Chart", labels = lbls)
pie(slices, main = "Pie3d Chart", labels = lbls2)
# 扇形图 <fan.plot>
slices <- c(10,12.4,16,8)
lbls <- c("US","UK","Australia","Germany","France")
fan.plot(slices, labels = lbls, main = "Fan Plot")

# （4）直方图 <hist\lines>
hist(mtcars$mpg)
hist(mtcars$mpg,breaks = 12,col = "red",xlab = "Miles Per Gallon", main = "Colored histogram with 12 bins")
hist(mtcars$mpg,freq = FALSE,breaks = 12,col = "red",xlab = "Miles Per Gallon", main = "Colored histogram with 12 bins")
lines(density(mtcars$mpg),col="blue",lwd=2)  # 原图上面添加
x <- mtcars$mpg
h <- hist(mtcars$mpg,breaks = 12,col = "red",xlab = "Miles Per Gallon", main = "Colored histogram with 12 bins")
xfit <- seq(min(x),max(x),length=40)
yfit <- dnorm(xfit, mean = mean(x),sd = sd(x))
yfit <- yfit*diff(h$mids[1:2]*length(x))
lines(xfit, yfit, col="blue", lwd=2)
box()  # 图放入方框中

# （5）核密度图 <density\plot\polygon\rug>
d <- density(mtcars$mpg)
plot(d)
plot(d,main = "Kernel Density of Miles Per Gallon")
polygon(d,col = "red",border = "blue")  # 原图上画多边形
rug(mtcars$mpg, col = "brown")
install.packages("sm")
# 转换因子
library(sm)
par(las=2)
cyl.f <- factor(mtcars$cyl, levels = c(4,6,8), labels = c("4 cylinder","6 cylinder","8 cylinder"))
levels(cyl.f)
sm.density.compare(mtcars$mpg, mtcars$cyl, xlab="Miles Per Gallon")
title("Mpg Distribution by Car Cylinder")
colfill <- c(2:(1+length(levels(cyl.f))))
# 单击图上确定在哪个地方添加图例
legend(locator(1), levels(cyl.f), fill = colfill)

# （6）箱型线 <boxplot>
boxplot(mtcars$mpg, main="Box Plot", ylab="Miles per Gallon")
# 跨组比较
boxplot(mpg~cyl,data = mtcars, main="Car Mileage Data", xlab="Number of Cylinder", ylab="Miles Per Gallon")
# 凹槽箱型线
boxplot(mpg~cyl,data = mtcars,notch=TRUE,varwidth=TRUE, main="Car Mileage Data", xlab="Number of Cylinder", ylab="Miles Per Gallon", col="red")
# 两个交叉因子箱型线
cyl.f <- factor(mtcars$cyl, levels = c(4,6,8), labels = c("4","6","8"))
am.f <- factor(mtcars$am, levels = c(0,1), labels = c("auto","standard"))
test <- data.frame(cyl.f, am.f, mtcars$mpg)
names(test)
boxplot(mtcars.mpg~am.f*cyl.f,data = test, varwidth=TRUE, col=c("gold","darkgreen"), main="MPG Distribution", xlab="Auto Type")

# （7）小提琴图 <vioplot>
install.packages("vioplot")
library(vioplot)
x1 <- mtcars$mpg[mtcars$cyl==4]
x2 <- mtcars$mpg[mtcars$cyl==6]
x3 <- mtcars$mpg[mtcars$cyl==8]
vioplot(x1,x2,x3,names=c("4 cyl","6 cyl","8 cyl"),col="gold")
title("Violin Plots of Miles Per Gallon")

# （8）点图 <dotchart>
test <- data.frame(mtcars$mpg,mtcars$disp)
names(test) <- c("mpg","disp")
test <- test[order(test$disp),]
plot(mpg~disp,data = test, type="l")
dotchart(mtcars$mpg,labels = row.names(mtcars), cex = 0.7, main = "Gas", xlab = "Miles")
# 分组、排序、着色
x <- mtcars[order(mtcars$mpg),] 
x$cyl <- factor(x$cyl)
x$color[x$cyl==4] <- "red"
x$color[x$cyl==6] <- "blue"
x$color[x$cyl==8] <- "darkgreen"
dotchart(x$mpg,labels = row.names(x),cex = 0.7,groups = x$cyl,gcolor = "black",color = x$color,pch = 19,main = "Gas",xlab = "Miles")


#### Charpter 07 基本统计分析 ####
# （1）描述性统计分析
# summary
summary(mtcars[,c("mpg","hp","wt")])
name.select <- c("mpg","hp","wt")
test <- mtcars[,name.select]
test[!is.na(test)]  # 剔除缺失值
mystats2 <- function(x, na.omit=FALSE){
  if(na.omit){
    x <- x[!is.na(x)]
  }
  m <- mean(x)
  n <- length(x)
  s <- sd(x)
  skew <- sum((x-m)^3/s^3)/n
  kurt <- sum((x-m)^4/s^4)/n-3
  return(c(n=n,mean=m,stdev=s,skew=skew,kurtosis=kurt))
}
sapply(test, mystats2)
# describe
install.packages("Hmisc")
library(Hmisc)
describe(test)
# stat.desc
install.packages("pastecs")
library(pastecs)
stat.desc(test)
# psych::describe
install.packages("psych")
library(psych)
describe(test)
Hmisc::describe(test)
psych::describe(test)
# aggregate 分组求统计量
aggregate(test, by=list(am=mtcars$am), mean)
# doBy
install.packages("doBy")
library(doBy)
summaryBy(mpg+hp+wt~am,data = mtcars, FUN=mystats3)
summaryBy(mpg+hp+wt~am,data = mtcars)
# reshape
library(reshape)
mystats4 <- function(x){
  c(n=length(x),mean=mean(x),sd=sd(x))
}
dfm <- melt(mtcars, measure.vars = c("mpg","hp","wt"),id.vars = c("am","cyl"))
head(dfm)
res.a <- cast(dfm,am+cyl+variable~.,mystats4)

# （2）列联分析
library(vcd)
head(Arthritis)
# <table><prop.table>
mytable <- with(Arthritis,table(Improved))
prop.table(mytable)
table()
head(mtcars)
# 二维列联表
mytable <- table(mtcars$cyl,mtcars$vs)
mytable <- xtabs(~Treatment+Improved,data=Arthritis)
mytable <- xtabs(~cyl+vs,data=mtcars)
margin.table(mytable,1)  # 边际累积值
prop.table(mytable)  # 百分比
prop.table(mytable,1)  # 边际百分比
prop.table(mytable,2)  # 边际百分比
addmargins(mytable)  # 添加边际和
addmargins(prop.table(mytable))  # 添加百分比边际和
addmargins(prop.table(mytable),2)  # 添加百分比边际和
test <- c(1,2,3,4,NA)
# 不忽略NA值
table(test, useNA = "ifany")
install.packages("gmodels")
library(gmodels)
CrossTable(Arthritis$Treatment,Arthritis$Improved)












