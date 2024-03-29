import StatsBase.countmap
using NPZ, RCall
include("../src/MyNN.jl")

function formatY(y, numClasses::Int64)
  const N = length(y)
  Y = zeros(Int32, N, numClasses)

  for i in 1:N
    c = (y[i] == 0 ? numClasses : y[i])
    Y[i, c] = 1
  end

  return Y
end

function formatX(X)
  const N = size(X,1)
  const P = prod(size(X, 2, 3))
  newX = [ones(N) zeros(N, P)]
  for i in 1:N
    newX[i, 2:end] = vec(X[i,:,:])
  end
  return newX
end


function computeError(X, y_truth, Theta)::Float64
  yy = copy(y_truth)
  yy[yy .== 0] = 10
  (dummy, pred) = MyNN.predict(X, Theta)
  return mean(yy .!= pred)
end

function confusion(X, y_truth, Theta)
  yy = copy(y_truth)
  yy[yy .== 0] = 10
  (dummy, pred) = MyNN.predict(X, Theta)

  classes = unique(yy)
  numClasses = length(classes)
  C = zeros(Int64, numClasses, numClasses)

  for i in 1:length(yy)
    C[pred[i], yy[i]] += 1
  end

  return C ### columns are truth, rows are pred
end

X_train = npzread("../dat/mnist_train_X.npy")
y_train = npzread("../dat/mnist_train_y.npy")
const numClasses = 10

### FORMAT DATA
X_pre = formatX(X_train)
X = copy(X_pre)
X = (X_pre[:,2:end] .- mean(X_pre[:,2:end], 1)) / maximum(X_pre[:, 2:end])
Y = formatY(y_train, numClasses)
#[y_train[idx] Y]

### TEST SET ###
X_test = npzread("../dat/mnist_test_X.npy");
y_test = npzread("../dat/mnist_test_y.npy");
Xnew = formatX(X_test)
Xnew[:,2:end] = (Xnew[:,2:end] .- mean(X_pre[:,2:end], 1)) / maximum(X_pre[:, 2:end])
Ynew = formatY(y_test, numClasses)
### TEST SET ###


#include("../src/MyNN.jl")
#### Find Optimal lambda
idx = randperm(length(y_train))
num_reps = 10 
lambda = linspace(0, 5, num_reps)
testError = zeros(num_reps)

for l in 1:length(lambda)
  println(l)
  @time out = MyNN.fit(X[idx[1:1000],:], Y[idx[1:1000],:], [35], 1.0,
                       maxIters=10000, eps=1E-4, lambda=lambda[l]);
  testError[l] = computeError(X[idx[1001:end],:], y_train[idx[1001:end]],
                              out.Theta)
end
lam_best = lambda[indmin(testError)]
n_train = size(X,1)

### NOTE ###
### Using the following configurations, this one-layer neural network produces
### a test-error of 5.9% (or 94% predictive accuracy).
#
#alpha=1.0
#lam_best=2.22
#@time out = MyNN.fit(X, Y, [35], alpha, lambda=lam_best, 
#                     maxIters=10000, eps=1E-4, printIter=true);
###

@time out = MyNN.fit(X[idx[1:n_train],:], Y[idx[1:n_train],:], [35], 1.0,
                     maxIters=10000, eps=1E-4, lambda=lam_best, printIter=true);
#@time out = MyNN.fit(X[idx,:], Y[idx,:], [35], 2.0, maxIters=10000, eps=1E-4, lambda=10.0);

### Frequency of Labels
sort(countmap(y_train))
sort(countmap(y_test))

### Training Error
train_error = computeError(X[idx[1:n_train],:], y_train[idx[1:n_train]], out.Theta)
confusion(X[idx[1:n_train],:], y_train[idx[1:n_train]], out.Theta)
### Testing Error
test_error = computeError(Xnew, y_test, out.Theta)
confusion(Xnew, y_test, out.Theta)


### PRINT RESULTS
println("Training Error: ", round(train_error,3))
println("Test Error:     ", round(test_error,3))

#### RF ###
#@rput X  Y  Xnew  y_test  y_train
#R"""
#library(randomForest)
#rfmod = randomForest(X[1:1000,], as.factor(y_train[1:1000]), ntree=500, maxnodes=4)
#### Training error
#mean(rfmod$pred != y_train)
#### Testint error
#mean(as.numeric(predict(rfmod, Xnew)) != y_test)
#"""
