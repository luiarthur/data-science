include("../src/BayesNN.jl")

N = 10000
y = zeros(Int64, N)
X = randn(N, 100)


### TODO: Use real Data
@time out = BayesNN.fit(y, X, 35, B=100, burn=200, printEvery=10);
