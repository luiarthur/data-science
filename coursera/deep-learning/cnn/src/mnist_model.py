### My Mnist Model ###
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(y, num_labels):
    N = y.size
    Y = np.zeros((N, num_labels))
    Y[np.arange(N), y] = 1
    return Y

def initialize_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forward_prop(X, params):
    """
    Note that we don't return the softmax Y's, but Z2 in the forward prop.
    Reason being that the cost function does the softmax also.
    Also note that the cost function requires labels and logits (not predicted labesl).
    """
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    Z1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    Z2 = tf.matmul(Z1, W2) + b2
    return Z2
    
#def compute_cost(Z2, Y):
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z2, labels=Y))
#    return cost

def split_train_val(X, Y, val_prop=.3):
    n = X.shape[0]
    n_val = int(n * val_prop)
    idx = np.random.permutation(n)
    idx_val = idx[:n_val]
    idx_train = idx[n_val:]
    return {'X_train': X[idx_train,:],
            'Y_train': Y[idx_train,:],
            'X_val': X[idx_val,:],
            'Y_val': Y[idx_val,:]}


def mnist_model(X_train, Y_train, X_test, Y_test, hidden_layer_size,
                learning_rate=.001, num_epochs=100, lam=0,
                mini_batch_size=100, print_cost=True):

    # TODO: Am I doing something wrong with the regualrizing?

    num_pixels = X_train.shape[1]
    num_classes = Y_train.shape[1]
    N = X_train.shape[0]

    # Create placeholders for X,Y
    X = tf.placeholder(tf.float32, [None, num_pixels])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    # Initialize weights
    W1 = initialize_weights((num_pixels, hidden_layer_size))
    W2 = initialize_weights((hidden_layer_size, num_classes))
    b1 = tf.Variable(tf.zeros(hidden_layer_size))
    b2 = tf.Variable(tf.zeros(num_classes))
    params = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

    # Forward prop
    Z2 = forward_prop(X, params)

    # Cost function
    #cost = compute_cost(Z2,Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z2, labels=Y))
    #regularizer = tf.reduce_mean(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    #cost = cost + lam * regularizer
    # FIXME:
    # https://greydanus.github.io/2016/09/05/regularization/
    # http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/
    regularizer = lam * tf.nn.l2_loss(W1) + lam * tf.nn.l2_loss(W2)
    cost = cost + tf.reduce_mean(regularizer)

    # Storage for costs in ephcs
    costs = []

    # Backprop. Define the optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    ### The next six lines is pretty standard boilerplate.
    # Initialize all the variables (defined previously using `tf.Variable`) globally
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        ### End of the boilerplate.

        for epoch in range(num_epochs):
            epoch_cost = 0.0
            num_minibatches = int(N / mini_batch_size)

            for minibatch in range(num_minibatches):
                idx = np.random.randint(0, N, mini_batch_size)
                mini_X = X_train[idx, :]
                mini_Y = Y_train[idx, :]

                _ , mini_cost = sess.run([optimizer, cost], feed_dict={X:mini_X, Y:mini_Y})
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
        if print_cost: print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        if print_cost: print("Train Accuracy:", train_accuracy)
        if print_cost: print("Test Accuracy:", test_accuracy)
                
        return {'train_accuracy': train_accuracy, 
                'test_accuracy': test_accuracy,
                'params': params, 'costs': costs}


def plot_cost(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (thinned)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


