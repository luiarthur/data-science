source("kmeans.R")

X = as.matrix(iris[,1:4])
idx = sample(1:NROW(X))
out1 = my.kmeans(X[idx,], 3)
out2 = kmeans(X[idx,], 3)

### Compare Centers ###
out1$centers
out2$centers

### Choosing number of clusters ###
sim = lapply(as.list(2:10), function(k) my.kmeans(X,k))
costs = sapply(sim, function(s) s$cost)
plot(costs, type='b')
