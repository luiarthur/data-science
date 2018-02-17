### INCOMPLETE ###

kernel = function(x, knots) {
  apply(knots, 1, function(knot) sum((knot - x)^2/2))
}

N = 1000
X = matrix(rnorm(N*2), N, 2)
y = X[,1] + X[,1]^2 - sin(X[,1]) + X[,2]^2/3 + rnorm(N, sd=.0001)


plot(X[,2], y)

pairs(cbind(y, X))

num_knots = N #10
knots = X #matrix(runif(2*num_knots), num_knots, 2)
kernel(X[1,], knots)
kernel(X, knots)
f = t(apply(X, 1, kernel, knots))

pairs(cbind(y, f[,1:3]))
