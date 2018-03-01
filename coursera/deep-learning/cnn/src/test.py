import mnist
import scipy.misc
import numpy as np
from mnist_model import *
import matplotlib.pyplot as plt

### Get Training Data ###
train_images = mnist.train_images()
train_labels = mnist.train_labels()

### Get Test Data ###
test_images = mnist.test_images()
test_labels = mnist.test_labels()

### Data Dimensions ###
num_features = train_images.shape[1] * train_images.shape[2]
num_classes = len(set(test_labels))

### Format Data ###
X_train = train_images.reshape((train_labels.size, num_features))
Y_train = one_hot(train_labels, 10)
dat_train = split_train_val(X_train, Y_train, val_prop=.3)

X_test = test_images.reshape((test_labels.size, num_features))
Y_test = one_hot(test_labels, 10)

### Plot Image ###
#plt.imshow(train_images[0,:, :])
#plt.show()

### Fit Model ###
#model = mnist_model(X_train, Y_train, X_test, Y_test, 35, num_epochs=20)

num_models = 5
lam = np.linspace(0, 2, num_models)
model = [None] * num_models

data = split_train_val(X_train, Y_train, val_prop=.3)
for i in range(num_models):
    print("lambda: ", lam[i])
    model[i] = mnist_model(data['X_train'], data['Y_train'],
                           data['X_val'], data['Y_val'],
                           hidden_layer_size=35, num_epochs=50,
                           mini_batch_size=500, lam=lam[i])

### Plot Cost ###
costs = [mod['costs'] for mod in model]
plot_cost(costs[0], lam[0])

### TODO ###
# - implement regularization
# - cross validation to pick lambda
# see: http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
