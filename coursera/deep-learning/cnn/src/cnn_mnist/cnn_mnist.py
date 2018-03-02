### My Mnist Model ###
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops # used to refresh session

def one_hot(y, num_labels):
    N = y.size
    Y = np.zeros((N, num_labels))
    Y[np.arange(N), y] = 1
    return Y

def initialize_weights(shape, name, initializer=None):
    if initializer is None:
        return tf.get_variable(name, shape)
    else:
        return tf.get_variable(name, shape, initializer=initializer)

def forward_propagation(X, parameters):
    # Retrieve the parameters from the dictionary "parameters" 
    b = parameters['b']
    W1 = parameters['W1']
    W2 = parameters['W2']
        
    # CONV2D: stride of 1, padding 'SAME'
    s=1
    Z1 = tf.nn.conv2d(X, W1, strides = [1,s,s,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    s,f = 8,8
    P1 = tf.nn.max_pool(A1, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    s=1
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,s,s,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f,s = 4,4
    P2 = tf.nn.max_pool(A2, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')
    # FLATTEN
    F = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(F, num_outputs=6, activation_fn=None) + b

    return Z3

def predict(X, parameters, reset_graph=True):
    if reset_graph: ops.reset_default_graph() 

    X = tf.constant(X, dtype=tf.float32)
    keys = parameters.keys()
    params = {}

    for k in keys:           
        p = parameters[k]
        params[k] = tf.Variable(p)

    Z_last = forward_prop(X, params)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred_y = sess.run(tf.argmax(Z_last, 1))

    return pred_y

 
def split_train_val(X, Y, val_prop=.3):
    n = X.shape[0]
    n_val = int(n * val_prop)
    idx = np.random.permutation(n)
    idx_val = idx[:n_val]
    idx_train = idx[n_val:]
    data = {'X_train': X[idx_train,:],
            'Y_train': Y[idx_train,:],
            'X_val': X[idx_val,:],
            'Y_val': Y[idx_val,:]}
    return data

def fit_cnn(X_train, Y_train, learning_rate=.001, lam=0,
            mini_batch_size, num_mini_batches=100, print_cost=True,
            print_accuracy=True):

    # Clear tf graphs after they are used
    ops.reset_default_graph() 

    return None
