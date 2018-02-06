using NPZ
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
X = formatX(X_train)
X[:,2:end] = (X[:,2:end] .- mean(X[:,2:end], 1)) / maximum(X[:, 2:end])
Y = formatY(y_train, numClasses)
#[y_train[idx] Y]


#include("../src/MyNN.jl")
idx = randperm(length(y_train))#[1:10000]
@time out = MyNN.fit(X[idx,:], Y[idx,:], [35], 2.0, maxIters=10000, eps=1E-4, lambda=2.0);

### Training Error
(pred,predYtrain) = MyNN.predict(X, out.Theta)
y_train[y_train .== 0] = 10
[y_train predYtrain]
println("Training Error: ", mean(y_train .!= predYtrain))

X_test = npzread("../dat/mnist_test_X.npy");
y_test = npzread("../dat/mnist_test_y.npy");
Xnew = formatX(X_test)
Xnew[:,2:end] = (Xnew[:,2:end] .- mean(X[:,2:end], 1)) / maximum(X[:, 2:end])
Ynew = formatY(y_test, numClasses)


(pred,predYtest) = MyNN.predict(Xnew, out.Theta)
y_test[y_test .== 0] = 10
[y_test  predYtest]
println("Test Error: ", mean(y_test .!= predYtest))

