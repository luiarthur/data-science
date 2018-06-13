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

    Z_last = forward_propagation(X, params)

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

### FIXME!!!
def fit_cnn(X_train, Y_train, X_val, Y_val, 
            learning_rate=.001, num_epochs=100, lam=0,
            mini_batch_size=100, print_cost=True, print_accuracy=True,
            outdir=None):
    nH = [4, 8]

    # Clear tf graphs after they are used
    ops.reset_default_graph() 

    # Save shapes to variables for convenience
    num_pixels = X_train.shape[1]
    num_classes = Y_train.shape[1]
    N = X_train.shape[0]

    # Note that tf.Variables will be updated tf.placeholders are fixed after
    # they are fed into the session.  They are like (lazy) constants, that can
    # be defined early on but initialize (and fixed) later.

    # Create placeholders for X,Y
    X = tf.placeholder(tf.float32, [None, num_pixels])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    #X = tf.placeholder(tf.float32)   # Alternatively, you can leave off the
    #Y = tf.placeholder(tf.float32)   # dimensions completely. But it's less clear.

    # Initialize weights
    W1 = initialize_weights([num_pixels, nH[0]], 'W1')
    W2 = initialize_weights([nH[1], num_classes], 'W2')
    b = initialize_weights([num_classes], 'b2')
    params = {"W1": W1, "W2": W2, "b": b}

    # Forward prop
    Z3 = forward_propagation(X, params)

    # Cost function
    # References for regularization:
    # https://greydanus.github.io/2016/09/05/regularization/
    # http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    regularizer = lam * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) / mini_batch_size
    cost = cost + regularizer

    # Storage for costs in ephcs
    costs = []

    # Backprop. Define the optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    ### The next six lines is pretty standard boilerplate.
    # Initialize all the variables (defined previously using `tf.Variable`) globally
    init = tf.global_variables_initializer()

    # An object to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        ### End of the boilerplate.

        for epoch in range(num_epochs):
            epoch_cost = 0.0
            num_minibatches = int(N / mini_batch_size)

            for minibatch in range(num_minibatches):
                # This is not the best way to obtain mini-batch 
                # Ideally, minibatches should be sampled
                # from original data without replacement.
                # This method samples with replacement.
                # Sampling without replacement can theoretically
                # lead to faster convergence.
                idx = np.random.randint(0, N, mini_batch_size)
                mini_X = X_train[idx, :]
                mini_Y = Y_train[idx, :]

                # sess.run([a, b]) will evaluate, then output (aka fetch) 
                # (a,b). Since the result of the optimizer is not important, it
                # is not stored. Hence `_`. The evaluation of the cost function
                # is useful for monitoring convergence. Hence, it is stored
                # in `mini_cost`. We had previously created placeholders for 
                # the data (X,Y). We feed actual data into the model
                # by supplying a dictionary to feed_dict as {X: my_X, Y: my_Y},
                # where my_X and my_Y are the actual data (like a np matrix).
                # Also, when `optimizer` is run, only one update is done. 
                # That is only one (and not multiple) step of the gradient
                # descent is done. Hence, this is SGD with minibatches.
                _ , mini_cost = sess.run(fetches=[optimizer, cost],
                                         feed_dict={X:mini_X, Y:mini_Y})
                epoch_cost += mini_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                costs.append(epoch_cost)


        # Calculate the correct predictions
        predicted_Y = tf.argmax(Z2, 1)
        correct_prediction = tf.equal(predicted_Y, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_val, Y: Y_val})

        if print_accuracy:
            print(accuracy)
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
        
        ### Write the model
        if outdir is not None:
            saver.save(sess, outdir)
            tf.summary.FileWriter(outdir + '.fw')
        
        parameters = {'W1': sess.run(W1), 
                      'W2': sess.run(W2),
                      'b' : sess.run(b)}
                
        return {'train_accuracy': train_accuracy, 
                'test_accuracy': test_accuracy, 
                'parameters': parameters, 'costs': costs, 'params': params}

