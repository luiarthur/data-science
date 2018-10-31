cimport libc.math as math
# cimport numpy as np
import numpy as np
cimport TuningParam
# from libcpp cimport bool

# cimport numpy as np
# from TuningParam import TuningParam

# Example:
# creating type name for function that takes in float as arg
# and returns float.
ctypedef float (*fl_fl_t)(float)

cdef float metropolis(float curr, fl_fl_t logFullCond, float proposalSd):
    cdef float cand = np.random.normal(curr, proposalSd)
    cdef float logU = math.log(np.random.rand())
    cdef float logP = logFullCond(cand) - logFullCond(curr)
    return cand if logP > logU else curr


# def metropolisMv(curr, logFullCond, proposalCov):
#     pass


# cdef metropolisAdaptive(float curr, fl_fl_t logFullCond, TP.TuningParam tuner):
#     draw = metropolis(curr, logFullCond, tuner.value)
#     accept = (draw == curr)
#     TP.update(tuner, accept)
#     return draw


cdef float logit(float p, float a, float b):
    return math.log(p - a) - math.log(b - p)


cdef float sigmoid(float x, float a, float b):
    cdef float out = 0.0

    if a == 0 and b == 1:
        out = 1 / (1 + math.exp(-x))
    else:
        ex = math.exp(x)
        out = (b * ex + a) / (1 + ex)

    return out


cdef float sech(float x):
     return 1 / math.cosh(x)

cdef float logpdf_logistic(float x):
    return -math.log(4) + 2 * math.log(sech(x / 2.0))


cdef logpdfLogX(float logX, fl_fl_t logpdfX):
    return logpdfX(math.exp(logX)) + logX


cdef float logpdfLogitX(float logitX, fl_fl_t logpdfX, float a, float b):
    cdef float x = sigmoid(logitX, a, b)
    cdef float logJacobian = logpdf_logistic(logitX) + math.log(b - a)
    return logpdfX(x) + logJacobian


# cdef double metLogitAdaptive(float curr, fl_fl_t ll, fl_fl_t lp, TuningParam
#                               tuner, float a, float b):
#     cdef lfc_logitX(float logit_x):
#         cdef float x = sigmoid(logit_x, a, b)
#         cdef float lp_logitX = logpdfLogitX(logit_x, lp, a, b)
#         return ll(x) + lp_logitX
# 
#     cdef float logit_x = metropolisAdaptive(logit(curr, a, b), lfc_logitX, tuner)
#
#    return sigmoid(logit_x, a, b)


# # def metLogAdaptive(curr: float, ll, lp, tuner):
# #     def lfc_logX(log_x: float):
# #         x = math.exp(log_x)
# #         return ll(x) + logpdfLogX(log_x, lp)
# # 
# #     log_x = metropolisAdaptive(math.log(curr), lfc_logX, tuner)
# # 
# #     return math.exp(log_x)
