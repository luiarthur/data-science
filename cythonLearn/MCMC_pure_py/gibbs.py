import copy


def get_substate(state, monitor):
    out = dict()
    for k in monitor:
        out[k] = copy.deepcopy(state.__getattribute__(k))
    return out


def gibbs(init, update_fn, nmcmc: int, burn: int, monitors=None,
          printFreq: int=0):

    # Create state object
    state = copy.deepcopy(init)

    # Create monitor
    if monitors is None:
        monitors = [list(state.__dict__.keys())]

    # Preallocate memory for output
    out = [[get_substate(state, mtr) for mtr in monitors]
           for i in range(nmcmc)]

    # Print Message
    def printMsg(mcmc_iter: int):
        if printFreq is not 0 and mcmc_iter % printFreq == 0:
            print("{} / {}".format(mcmc_iter, nmcmc))

    # Burn
    for i in range(burn):
        printMsg(i)
        update_fn(state)

    # Gibbs
    for i in range(nmcmc):
        printMsg(i + burn)
        update_fn(state)

        # Grab params to put in output
        out[i] = [get_substate(state, mtr) for mtr in monitors]

    return out

# Implement the individual update functions for each parameter
# in update_fn using Cython.
