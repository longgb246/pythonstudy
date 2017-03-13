#### Charpter 01 从qplot入门 ####
library(ggplot2)
set.seed(1410)
dsmall <- diamonds[sample(nrow(diamonds),100),]  # 取出数据
qplot(carat, price, data = diamonds)      # 快速画图，散点图 
qplot(log(carat), log(price), data = diamonds)   # 使用log函数
qplot(dsmall$carat, dsmall$price, color = dsmall$color)  # 使用color指定分类，画分类图
qplot(dsmall$carat, dsmall$price, shape = dsmall$cut, color=dsmall$color)   # 注意：最好使用data，不然label会被改变。使用shape，当shape和color相同的时候，效果叠加。
qplot(carat, price, data = diamonds, alpha = I(1/10))    # 这里必须使用I(1/10)，而不能使用0.1来表示。
# geom = "point"     # 散点图 
# geom = "smooth"    # 平滑图
# geom = "boxplot"   # 胡须图
# geom = "path"      # 连线图
# geom = "line"      # 连线图
# geom = "histogram" # 直方图
# geom = "freqploy"  # 多边图
# geom = "density"   # 密度图
# geom = "bar"       # 条形图
qplot(carat, price, data = dsmall, geom = c("point","smooth"))   # 使用2个图
qplot(carat, data = diamonds, geom="histogram", binwidth=0.1, xlim = c(0,3), color=color, fill=color)
qplot(carat, data = diamonds, geom="density", color=color)

# p37

