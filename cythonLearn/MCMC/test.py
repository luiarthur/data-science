import MCMC
import numpy as np

print("Hello!")

J = 3
s = MCMC.State(np.array(range(J))*1.0, np.array(range(J)), np.zeros(3), 0.0)
print(s)
print(type(s))
print(s.alpha)
for a in s.alpha:
    print(a)


def fieldnames(x):
    return list(filter(lambda xi: xi[0:2] != '__', x.__dir__()))


fieldnames(s)

MCMC.cmodel(np.random.randn(100), 10, s, {'N': 100}, nmcmc=1000, nburn=1000)
