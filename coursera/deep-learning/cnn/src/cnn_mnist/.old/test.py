import mnist
import scipy.misc
import numpy as np
from cnn_mnist import *
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
Y_train = one_hot(train_labels, num_classes)
dat_train = split_train_val(X_train, Y_train, val_prop=.3)

X_test = test_images.reshape((test_labels.size, num_features))
Y_test = one_hot(test_labels, num_classes)

### Plot Image ###
#plt.imshow(train_images[0,:, :])
#plt.show()

### Fit Model ###
#model = mnist_model(X_train, Y_train, X_test, Y_test, 35, num_epochs=20)

num_models = 5
lam = np.linspace(0, 3, num_models)
model = [None] * num_models

data = split_train_val(X_train, Y_train, val_prop=.3)

#for i in range(num_models):
i=2
print("lambda: ", lam[i])
model[i] = fit_cnn(data['X_train'], data['Y_train'],
                   data['X_val'], data['Y_val'],
                   learning_rate=.001, lam=lam[i],
                   num_epochs=30, mini_batch_size=500)



