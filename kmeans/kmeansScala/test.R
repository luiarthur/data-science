library(rscala)
scala(classpath='target/scala-2.12/kmeansscala_2.12-0.1.0.jar')

K = 8
P = 10
N = 50000
cen = matrix(rnorm(P * K, sd=10), K, P)
idx = sample(1:K, N, replace=T)
X = matrix(cen[idx,], N, ) + rnorm(N*P)

X = as.matrix(iris[,-5])


s %@% 'import kmeansScala.Kmeans.{kmeans,timer}'
s %@% 'val out = timer{kmeans(X=R.getD2("X"),K=@{K})}'
lab = s %~% 'out._2'
cenScala = s %~% 'out._1'

system.time(kmR <- kmeans(X,K))
labR <- kmR$clus
table(labR)
table(lab)

plot(0, xlim=c(1,ncol(cenScala)), ylim=range(cenScala), type='n')
for (r in 1:NROW(cenScala)) lines(cenScala[r,], lwd=2, col='blue')
for (r in 1:NROW(cenScala)) lines(kmR$centers[r,], col='red', lwd=2)
