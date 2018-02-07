using NPZ, RCall
include("../src/MyNN.jl")

function formatY(y, numClasses::Int64)
  const N = length(y)
  Y = zeros(Int32, N, numClasses)

  for i in 1:N
    c = (y[i] == 0 ? 10 : y[i])
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

X_train = npzread("../dat/mnist_train_X.npy")
y_train = npzread("../dat/mnist_train_y.npy")
const numClasses = 10

### FORMAT DATA
X_pre = formatX(X_train)
X = copy(X_pre)
X[:,2:end] = (X_pre[:,2:end] .- mean(X_pre[:,2:end], 1)) / maximum(X_pre[:, 2:end])
Y = formatY(y_train, numClasses)
#[y_train[idx] Y]

### PCR ###
@rput X
R"""
pX = prcomp(X)
pX_prop = cumsum(pX$sd^2) / sum(pX$sd^2)
thresh = .90 # TODO: vary this
idx = min(which(pX_prop > thresh))
X_pcr = cbind(1, pX$x[,1:idx])
"""
@rget X_pcr
### PCR ###

#include("../src/MyNN.jl")
idx = randperm(length(y_train))[1:end]
@time out = MyNN.fit(X_pcr[idx,:], Y[idx,:], [35], 2.0,
                     maxIters=10000, eps=1E-4, lambda=2.0, printIter=true);
#@time out = MyNN.fit(X[idx,:], Y[idx,:], [35], 2.0, maxIters=10000, eps=1E-4, lambda=10.0);

### Training Error
(pred,predYtrain) = MyNN.predict(X_pcr[idx,:], out.Theta)
y_train[y_train .== 0] = 10
#[y_train predYtrain]

X_test = npzread("../dat/mnist_test_X.npy");
y_test = npzread("../dat/mnist_test_y.npy");
Xnew = formatX(X_test)
Xnew[:,2:end] = (Xnew[:,2:end] .- mean(X_pre[:,2:end], 1)) / maximum(X_pre[:, 2:end])
Ynew = formatY(y_test, numClasses)

### PCR ###
@rput Xnew;
R"""
V = cbind(1,Xnew %*% pX$rotation[,1:idx])
"""
@rget V;
### PCR ###

(pred,predYtest) = MyNN.predict(V, out.Theta)
y_test[y_test .== 0] = 10
#[y_test  predYtest]


### PRINT RESULTS
println("Training Error: ", mean(y_train[idx] .!= predYtrain))
println("Test Error: ", mean(y_test .!= predYtest))

