import numpy as np

X = np.random.randn(5,3)
np.pad(X, ((1,1), (2,2)), 'constant', constant_values=(1,1))

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(9.0).reshape((3, 3))
np.multiply(x1, x2)
