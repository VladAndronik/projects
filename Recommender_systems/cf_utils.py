# Some util functions used in notebooks
import tensorflow as tf
import numpy as np
RANDOM_STATE = 17

def create_placeholders(n_cols, n_factors):
    Y = tf.placeholder(tf.float32, name='Y', shape=(None, n_cols))
    user_emb = tf.placeholder(tf.float32, name='user_emb', shape=(2, n_factors))
    return Y, user_emb

def initialize_parameters(u_emb, i_emb):
    
    p_u = tf.Variable(u_emb, name='p_u', validate_shape=False, dtype=tf.float32)  # variable for user embedding
    q_i = tf.Variable(i_emb, name='q_i', dtype=tf.float32) # variable for item embedding
    
    parameters = {
        'p_u':p_u,
        'q_i':q_i
    }
    return parameters

def initialize_parameters_with_random():
    p_u = tf.get_variable(shape=(10000, 20), dtype=tf.float64, name='user', 
                          initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_STATE))
    q_i = tf.get_variable(shape=(20, 4516), dtype=tf.float64, name='item', 
                          initializer=tf.contrib.layers.xavier_initializer(seed=RANDOM_STATE))
    parameters = {
        'p_u':p_u,
        'q_i':q_i
    }
    
    return parameters

def forward_pass(params):
    P, Q = params['p_u'],params['q_i']
    Y_pred = tf.matmul(P, Q)
    return Y_pred


def compute_cost(Y_activation, Y):
    predictions = tf.transpose(Y_activation)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels, predictions))
    return cost

def random_mini_batches(X, Y, mini_batch_size = 10, seed = 17):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []

    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :].astype(np.float32)
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :].astype(np.float32)
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :].astype(np.float32)
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :].astype(np.float32)
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches
