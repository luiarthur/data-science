# https://notes-on-cython.readthedocs.io/en/latest/std_dev.html

cimport numpy as np
cimport cython
import copy

def gibbs(init, update, monitors, nmcmc: int, nburn: int, printFreq: int=0):
    state = copy.deepcopy(init)
    out = [ copy.deepcopy(state) for i in nmcmc ]

    def printMsg(i: int):
        if printFreq > 0 and (i + 1) % printFreq == 0:
            print("{} / {}".format(i + 1, nmcmc + nburn))

    for i in range(nburn):
        update(state)
        printMsg(i)

    for i in range(nmcmc):
        update(state)
        printMsg(i + nburn)
        out[i] = copy.deepcopy(state)

    return out


cdef class State:
    cdef public np.double_t[:] alpha
    cdef public np.int_t[:] c
    cdef public np.double_t[:] mu
    cdef public double sig2

    def __init__(self, alpha, c, mu, sig2):
        self.alpha = alpha
        self.c = c
        self.mu = mu
        self.sig2 = sig2

cdef void update(s: State):
    #TODO
    pass

def cmodel(np.ndarray[np.float64_t, ndim=1] y not None, int J, State init, constants, int nmcmc, int nburn):
    cdef int N = constants["N"]
    cdef int n = 0

    return gibbs(init, update, [], nmcmc, nburn, printFreq=0)
