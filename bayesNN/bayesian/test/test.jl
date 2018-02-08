import StatsBase.countmap
using NPZ, RCall
R"library(rcommon)"
include("../src/BayesNN.jl")
include("../../nonbayesian/src/MyNN.jl")

X_train = BayesNN.format_X(npzread("../../nonbayesian/dat/mnist_train_X.npy"));
y_train = BayesNN.format_y(npzread("../../nonbayesian/dat/mnist_train_y.npy"));
X_test  = BayesNN.format_X(npzread("../../nonbayesian/dat/mnist_test_X.npy"));
y_test  = BayesNN.format_y(npzread("../../nonbayesian/dat/mnist_test_y.npy"));
Y_train = convert(Matrix{Int32}, BayesNN.y_vec2mat(y_train))

(X_train, xbar, xmax) = BayesNN.scale(X_train)
X_test = BayesNN.scale(X_test, xbar, xmax)

idx = randperm(length(y_train))

### Fit the Model ########################################################
n_train = 10000
X1 = [ones(n_train) X_train[idx[1:n_train],:]]
@time warmup = MyNN.fit(X1, Y_train[idx[1:n_train],:], [end][35], 1.0,
                        printIter=true, maxIters=10000, eps=1E-4, lambda=2.0)
#n_train = 100
#@time out = BayesNN.fit(y_train[idx[1:n_train]], X_train[idx[1:n_train],:], 35,
#                        B=50, burn=5000, prior=BayesNN.Prior(10, 2, 10, .01),
#                        printEvery=5, thin=10);
#
#n_train = 1000
#@time out = BayesNN.fit(y_train[idx[1:n_train]], X_train[idx[1:n_train],:], 35,
#                        B=50, burn=1000, prior=BayesNN.Prior(10, 2, 10, .01),
#                        printEvery=5, thin=10, init=out[end]);

n_train = 10000
@time out = BayesNN.fit(y_train[idx[1:n_train]], X_train[idx[1:n_train],:], 35,
                        B=50, burn=0, prior=BayesNN.Prior(10, 2, 10, .01),
                        printEvery=5, thin=10,
                        init=BayesNN.State(warmup.Theta, 1/sqrt(2.0), -Inf));

n_train = 10000
@time out = BayesNN.fit(y_train[idx[1:n_train]], X_train[idx[1:n_train],:], 35,
                        B=50, burn=100, prior=BayesNN.Prior(10, 2, 10, .01),
                        printEvery=5, thin=10, init=out[end]);
### Fit the Model ########################################################


### View Posterior Samples ###
lambda = [o.lambda for o in out]
ll = [o.loglike for o in out]
R"plotPost($lambda, main='lambda')";
R"plot($ll, type='b', xlab='MCMC iteration', ylab='loglike')";

### Predictions ###
@time (pred_y, pred_Y) = BayesNN.predict(X_test, out)
mean(pred_y .== y_test)
