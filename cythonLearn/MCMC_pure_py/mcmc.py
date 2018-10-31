import math
import random
from TuningParam import TuningParam

# Example:
# creating type name for function that takes in int and str as args
# and returns float.
# ctypedef float (*f_type)(int, str)

def metropolisBase(curr: float, logFullCond, proposalSd):
    cand = random.normalvariate(curr, proposalSd)
    logU = math.log(random.random())
    logP = logFullCond(cand) - logFullCond(curr)
    accept = logP > logU
    draw = cand if accept else curr
    return (draw, accept)


def metropolis(curr: float, logFullCond, proposalSd):
    return metropolisBase(curr, logFullCond, proposalSd)[0]


# def metropolisMv(curr, logFullCond, proposalCov):
#     pass


def metropolisAdaptive(curr: float, logFullCond, tuner: TuningParam):
    draw, accept = metropolisBase(curr, logFullCond, tuner.value)
    tuner.update(accept)
    return draw


def logit(p: float, a: float, b: float):
    return math.log(p - a) - math.log(b - p)


def sigmoid(x: float, a: float, b: float):
    out = 0.0

    if a == 0 and b == 1:
        out = 1 / (1 + math.exp(-x))
    else:
        ex = math.exp(x)
        out = (b * ex + a) / (1 + ex)

    return out


def sech(x: float):
    return 1 / math.cosh(x)

def logpdf_logistic(x: float):
    return -math.log(4) + 2 * math.log(sech(x / 2.0))


def logpdfLogX(logX: float, logpdfX):
    return logpdfX(math.exp(logX)) + logX


def logpdfLogitX(logitX: float, logpdfX: float, a: float, b: float):
    x = sigmoid(logitX, a, b)
    logJacobian = logpdf_logistic(logitX) + math.log(b - a)
    return logpdfX(x) + logJacobian


def metLogitAdaptive(curr: float, ll, lp, tuner: TuningParam, a: float, b:float):
    def lfc_logitX(logit_x: float):
        x = sigmoid(logit_x, a, b)
        lp_logitX = logpdfLogitX(logit_x, lp, a, b)
        return ll(x) + lp_logitX

    logit_x = metropolisAdaptive(logit(curr, a, b), lfc_logitX, tuner)

    return sigmoid(logit_x, a, b)


def metLogAdaptive(curr: float, ll, lp, tuner):
    def lfc_logX(log_x: float):
        x = math.exp(log_x)
        return ll(x) + logpdfLogX(log_x, lp)

    log_x = metropolisAdaptive(math.log(curr), lfc_logX, tuner)

    return math.exp(log_x)
