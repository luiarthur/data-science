fit = function(y, X, k=3) {
  pX = prcomp(X, center=FALSE, scale=FALSE)
  mod = lm(y ~ pX$x[,1:k] - 1)
  list(pX=pX, mod=mod, k=k)
}

pred = function(X, fit_ls) {
  k = fit_ls$k
  V = fit_ls$pX$rotation # eigenvectors
  gam = fit_ls$mod$coef
  b = V[,1:k] %*% gam
  X %*% b
}

dat = scale(mtcars)
y = dat[,1]
X = dat[,-1]
pX = prcomp(X)
plot(cumsum(pX$sd^2 / sum(pX$sd^2)), ylim=c(0,1)); abline(h=.9, lty=2)
pairs(cbind(y, pX$x[,1:3]))
N = nrow(X)

models = lapply(as.list(1:N), function(i) fit(y[-i], X[-i,]))

y_hat = sapply(1:N, function(i) pred(X[i,], models[[i]]))
plot(y, y_hat); abline(0,1)

