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
for i in range(num_models):
    print("lambda: ", lam[i])
    model[i] = mnist_model(data['X_train'], data['Y_train'],
                           data['X_val'], data['Y_val'],
                           learning_rate=.001, lam=lam[i],
                           hidden_layer_size=25, num_epochs=30,
                           mini_batch_size=500)


### Plot Cost ###
costs = [mod['costs'] for mod in model]
test_errors = [1 - mod['test_accuracy'] for mod in model]
indmin_testerror = np.argmin(test_errors)
lam_best = lam[indmin_testerror]

plot_cost(costs[indmin_testerror], lam_best)

### View the trained parameters
#model[0]['parameters']
#model[0]['params']
#tf.trainable_variables()


### TODO ###
# - Predict on new data
final_model = mnist_model(X_train, Y_train,
                          X_test, Y_test,
                          learning_rate=.001, lam=lam_best,
                          hidden_layer_size=25, num_epochs=50,
                          mini_batch_size=500)


y_hat = predict(X_test, final_model['parameters'])
np.mean(y_hat == test_labels)


### Stuff ###
#sess = tf.Session()
#saver = tf.train.import_meta_graph('out/model.ckpt.meta')
#saver.restore(sess,tf.train.latest_checkpoint('out/'))
#
#keys = final_model['params'].keys()
#params = {}
#for k in keys:
#    params[k] = sess.run(k+":0")
#
