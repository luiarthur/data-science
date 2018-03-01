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

### Format Data ###
X_train = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
Y_train = one_hot(train_labels, 10)
X_test = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
Y_test = one_hot(test_labels, 10)

### Plot Image ###
#plt.imshow(train_images[0,:, :])
#plt.show()

### Fit Model ###
train_accuracy, test_accuracy, params, costs = mnist_model(X_train, Y_train, X_test, Y_test, 35, 
                                                           num_epochs=20)

### Plot Cost ###
plot_cost(costs, .001)

### TODO ###
# - implement regularization
# - cross validation to pick lambda
# see: http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
