# Simulate a random walk with drift
# Use bayesian methods to recover posterior distribution 
# of model parameters.
# This model implicitly clusters.
#
# Model:
# y_t | mu_t, sig2 ~ N(y_{t-1} | mu_t, sig2)
#            sig2 ~ IG(a, b)
#       mu_t | G ~ G
#              G ~ DP(alpha, Normal(0, 1))
