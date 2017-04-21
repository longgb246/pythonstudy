library(data.table)
help("data.table")


# =======================================================
# =                       �ĵ�������                    =
# =======================================================

# 1 ������Ϣ ####
df1 = data.frame(x=rep(c("b", "a", "c"), each=3), y=c(1,3,6), v=1:9)
dt1 = data.table(x=rep(c("b", "a", "c"), each=3), y=c(1,3,6), v=1:9)
identical(dim(df1), dim(dt1))     # TRUE
identical(df1$a, dt1$a)           # TRUE
is.list(df1)                      # TRUE
is.list(dt1)                      # TRUE
is.data.frame(dt1)                # TRUE
tables()

  
# 2 ��������Ƭ ####  
dt1[2,]
dt1[3:2,]
dt1[order(y),]                    # ����x���������﷨�߼�:1���ȶ�x�������е�����2������������ԭdata.table����
dt1[y>2 & v>5,][order(y, v),]     # ��ô���а���2��˳���������python����
dt1[!(2:4)]
dt1[,list(v)]                     # ʹ��list()��ʹ�÷��ؽ����Ȼ��data.table
dt1[,sum:=sum(v)]                 # ʹ�ü��㺯��
dt1[,sum:=NULL]                   # ɾ��ĳһ��
dt1[,list(a=sum(v), b=v^2)]       # �������У���������Ϊa,b


# 3 �ۺϺ��� ####
dt1[, sum(v), by=x]               # ����x���оۺϣ�����ÿ��x��sum
dt1[, sum(v), by=list(x, sm)]     # ���groupby


# 4 ���� ####
dt2 = data.table(x=c("c", "b"), v=8:7, foo=c(4,2))
dt1[dt2, on="x"]                  # right join��������ߵ��ֶ��Ǳ���
dt2[dt1, on="x"]                  # left join���յ�ֵΪNA
dt1[dt2, on="x", nomatch=0]       # inner join
dt1[!dt2, on="x"]                 # �
dt1[dt2, on=c("x", "v"), nomatch=0]          # ����ֶεĽ���


# 5 ���� ####
dt1[nrow(dt1),]                   # ���һ�� 
dt1[, ncol(dt1)]                  # ��ô����iloc���Ƶ�������


# 6 �߼� ####
dt3 = data.table(x=rep(c("b","a","c"),each=3), v=c(1,1,1,2,2,1,1,2,2), y=c(1,3,6), a=1:9, b=9:1)
dt3[, sum(v), by=.(y%%2)]         # ����y%%2�ļ��������з���ͳ��
dt3[, list(sm=sum(v)), by=.(bool = y%%2)]   # �ı�������
dt3[, list(MySum=sum(v), MyMin=min(v), MyMax=max(v)), by=list(x, y%%2)]     # ���м���
dt3[, .(seq=min(a):max(b)), by=x]
dt3[, {tmp <- mean(y);tmp <- tmp-1;.(a=a-tmp, b=b-tmp)}, by=x]              # ʹ��{�����м����} 












# =======================================================
# =                      R In Action                    =
# =======================================================

vars <- c("mpg", "hp", "wt")
mtcars[vars]
mm <- data.table(mtcars[vars])
mm$index <- row.names(mtcars)
names(mm)             # ȡ����
row.names(mtcars)     # ȡindex��
summary(subset(mm, select = vars))      # ʹ��subset()����ѡȡ

# 143.
# Ӧ��ֱ��ѧϰԭ����Ȼ����pythonֱ��ʵ�֣������Ǻ����ӡ�










