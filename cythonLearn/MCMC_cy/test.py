import random
import math
import copy
import numpy as np
from scipy import stats


import mcmc
from gibbs import gibbs
from TuningParam import TuningParam


J = 3
muTrue = [random.normalvariate(0, 10) for j in range(J)]
sig2True = 0.5

nj = 300
y = [np.random.normal(m, math.sqrt(sig2True), nj).tolist() for m in muTrue]

n = nj * J

muPriorMean = 0.0
muPriorSd = 0.0

sig2Prior_a = 3.0
sig2Prior_b = 2.0

tunerMu = [copy.deepcopy(TuningParam(1.0)) for j in range(J)]
tunerSig2 = TuningParam(1.0)

def logpdf_invgamma(x: float, a: float, b: float):
    return a * math.log(b) - math.lgamma(a) - (a - 1) * math.log(x) - b / x

def update(s):
    def update_muj(j: int):
        def lfc(muj: float):
            ll = 0.0
            for j in range(J):
                for i in range(nj):
                    ll += stats.norm.logpdf(y[j][i], muj, math.sqrt(s.sig2))

            lp = stats.norm.logpdf(muj, muPriorMean, muPriorSd)
            return ll + lp

        s.mu[j] = mcmc.metropolisAdaptive(s.mu[j], lfc, tunerMu[j])

    def update_mu():
        for j in range(J):
            update_muj(j)

    def update_sig2():
        def ll(sig2: float):
            out = 0.0
            for j in range(J):
                out += stats.norm.logpdf(y[j], s.mu[j], math.sqrt(sig2))

            return out

        def lp(sig2: float):
            return logpdf_invgamma(sig2, sig2Prior_a, sig2Prior_b)

        s.sig2 = mcmc.metLogAdaptive(s.sig2, ll, lp, tunerSig2)

    update_mu()
    update_sig2()


class State:
    def __init__(self, mu: list, sig2: float):
        self.mu = mu
        self.sig2 = sig2


init = State([0.0] * J, 1.0)
out = gibbs(init, update, nmcmc=1000, burn=1000, printFreq=1)
