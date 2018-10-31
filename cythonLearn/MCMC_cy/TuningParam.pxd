# from libcpp cimport bool
cimport libc.math as math


ctypedef float (*int_fl_t)(int)


cdef inline float delta_default(int n):
    return min(.01, n ** (-0.5))


cdef struct TuningParam:
    float value
    int acceptance_count
    int current_iter
    int batch_size
    float target_acceptance_rate
    int_fl_t delta



cdef inline TuningParam TuningParam_default(float value):
    return TuningParam(value, 0, 0, 50, 0.44, delta_default)


cdef inline float acceptance_rate(TuningParam tp):
    return tp.acceptance_count / tp.batch_size


cdef inline void update(TuningParam tp, bint accept):
    if accept:
        tp.acceptance_count += 1

    tp.current_iter += 1
    cpdef int n
    cpdef float factor

    if tp.current_iter % tp.batch_size == 0:
        n = <int>math.floor(tp.current_iter / tp.batch_size)
        factor = math.exp(tp.delta(n))
        if acceptance_rate(tp) > tp.target_acceptance_rate:
            tp.value *= factor
        else:
            tp.value /= factor

        tp.acceptance_count += 1
