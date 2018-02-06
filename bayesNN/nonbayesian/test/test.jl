using NPZ
include("../src/MyNN.jl")

function formatY(y, numClasses::Int64)
  const N = length(y)
  Y = zeros(Int32, N, numClasses)

  for i in 1:N
    Y[i, y[i] == 0 ? 10 : y[i]] = 1
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
idx = randperm(length(y_train))
X = formatX(X_train)[idx,:]
X[:,2:end] = (X[:,2:end] .- mean(X[:,2:end], 1)) / maximum(X[:, 2:end])
Y = formatY(y_train, numClasses)[idx,:]

include("../src/MyNN.jl")
out = MyNN.fit(X[1:1000,:], Y[1:1000,:], [25], 1E-4, eps=1E-3,
               maxIters=200, eps=1E-3, lambda=0.0);

X_test = npzread("../dat/mnist_test_X.npy");
y_test = npzread("../dat/mnist_test_y.npy");
Xnew = formatX(X_test)
Xnew[:,2:end] = (Xnew[:,2:end] .- mean(X[:,2:end], 1)) / maximum(X[:, 2:end])
Ynew = formatY(y_test, numClasses)


#pred = MyNN.predict(Xnew, out.Theta)
(pred,predY) = MyNN.predict(Xnew, out.Theta)
[y_test predY]
