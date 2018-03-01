# Exponential Moving Average

ema = function(moving_avg, new_obs, weight_on_hist=.9) {
  weight_on_hist * moving_avg + (1-weight_on_hist) * new_obs
}

### Test ###
N = 1000
y = double(N)
x = double(N)
mx = double(N)
my = double(N)
alpha = .99
for (i in 2:N) {
  y[i] = y[i-1] + rnorm(1, sd=1)
  x[i] = x[i-1] + rnorm(1, sd=1)
  mx[i] = ema(mx[i-1], x[i], w=alpha)
  my[i] = ema(my[i-1], y[i], w=alpha)
}

### Plots ###
plot(y, type='l')
lines(my, col='red', lwd=3)

plot(x, type='l')
lines(mx, col='red', lwd=3)

plot(x,y, type='l', col='grey')
points(x[1], y[1], pch=20, cex=3)
points(x[N], y[N], pch=4, cex=3, lwd=3)
lines(mx,my, col='red', lwd=3)

