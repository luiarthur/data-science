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
@time out = BayesNN.fit(y_train[idx[1:n_train]], X_train[idx[1:n_train],:], 35,
                        B=50, burn=50000, prior=BayesNN.Prior(10, 2, 10, .01),
                        printEvery=5, thin=100);
### Fit the Model ########################################################


### View Posterior Samples ###
lambda = [o.lambda for o in out]
lfc = [o.log_fc for o in out]
R"plotPost($lambda, main='lambda')";
R"plot($lfc, type='b', xlab='MCMC iteration', ylab='loglike')";

### Predictions ###
@time (pred_y, pred_Y) = BayesNN.predict(X_test, out)
mean(pred_y .== y_test)
BayesNN.confusion(pred_y, y_test, 10)
pred_certainty = mean(pred_Y' .== pred_y, 2)

### Look at confusion matrix of uncertain predictions ###
idx = find(x -> x < .5, pred_certainty)
[ y_test[idx] pred_y[idx] pred_Y[:,idx]' ]
mean( y_test[idx] .== pred_y[idx] )
BayesNN.confusion(pred_y[idx], y_test[idx], 10)

### Look at confusion matrix of certain predictions ###
idx_confident = find(x -> x > .5, pred_certainty);
[ y_test[idx_confident] pred_y[idx_confident] pred_Y[:,idx_confident]' ]
mean( y_test[idx_confident] .== pred_y[idx_confident] )
BayesNN.confusion(pred_y[idx_confident], y_test[idx_confident], 10)
