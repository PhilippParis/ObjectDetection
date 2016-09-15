"""
based on the tensorflow tutorial:
https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/mnist.py
"""

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# ----------------- HELPERS ----------------- #

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# ============================================================= #

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ============================================================= #

def weight_var(shape):
    weights = tf.get_variable('weights', shape,
                           initializer = tf.truncated_normal_initializer(stddev=0.1))
    variable_summaries(weights, 'weights')
    return weights

# ============================================================= #

def bias_var(shape):
    biases = tf.get_variable('biases', shape,
                           initializer = tf.constant_initializer(0.1))
    variable_summaries(biases, 'biases')
    return biases

# ============================================================= #

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    name = tf.get_variable_scope().name + '/' + name
    
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)


# ----------------- NETWORK ----------------- #

def create_network(network_input, keep_prob, image_size):
    """
    Builds the convolutional network model (2x conv, 2x fully connected with dropout)
    
    Returns:
        logits
    """        
    input_reshaped = tf.reshape(network_input, [-1, image_size, image_size, 1])
    
    # - Layer 1 - #
    with tf.variable_scope('conv1') as scope:
        weights = weight_var([3, 3, 1, 32])
        biases = bias_var([32])
        
        conv = tf.nn.relu(conv2d(input_reshaped, weights) + biases)
        output_layer_1 = max_pool_2x2(conv)
        
        
    # - Layer 2 - #
    with tf.variable_scope('conv2') as scope:
        weights = weight_var([3, 3, 32, 64])
        biases = bias_var([64])
        
        conv = tf.nn.relu(conv2d(output_layer_1, weights) + biases)
        output_layer_2 = max_pool_2x2(conv)
        
        
    # - Layer 3 - #
    with tf.variable_scope('local3') as scope:
        size = image_size / 4
        weights = weight_var([size * size * 64, 1024])
        biases = bias_var([1024])
        
        output_layer_2_flat = tf.reshape(output_layer_2, [-1, size * size * 64])
        output_layer_3 = tf.nn.relu(tf.matmul(output_layer_2_flat, weights) + biases)
        
        # - Drop out -#
        output_layer_3_drop = tf.nn.dropout(output_layer_3, keep_prob)
        
        
    #- layer 4 -#
    with tf.variable_scope('local4') as scope:
        weights = weight_var([1024, 2])
        biases = bias_var([2])
        
        output = tf.nn.softmax(tf.matmul(output_layer_3_drop, weights) + biases)
        variable_summaries(output, 'network-output')
        return output


# ----------------- TRAINING ----------------- #

def train(network_output, desired_output):
    with tf.name_scope('xent'):
        cross_entropy = -tf.reduce_mean(desired_output * tf.log(network_output))
        tf.scalar_summary('cross entropy', cross_entropy)
        
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
