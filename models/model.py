"""
based on the tensorflow tutorial:
https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/mnist.py
"""

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# ----------------- HELPERS ----------------- #

def conv2d(x, ksize, strides):
    weights = weight_var(ksize, 0.0)
    biases = bias_var(ksize[3])
    
    conv = tf.nn.conv2d(x, weights, strides=strides, padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv, biases))

# ============================================================= #

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# ============================================================= #

def weight_var(shape, wd):
    weights = tf.get_variable('weights', shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
    variable_summaries(weights, 'weights')
    
    if wd != 0.0:
        weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
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

def create(network_input, keep_prob):
    """
    Builds the convolutional network model (2x conv, 2x fully connected with dropout)
    
    Returns:
        logits
    """        
    input_reshaped = tf.reshape(network_input, [-1, FLAGS.input_size, FLAGS.input_size, 1])
    
    # - Layer 1 - #
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(input_reshaped, [4, 4, 1, 20], [1, 1, 1, 1])
        norm1 = tf.nn.lrn(conv1, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        # output size img_size x img_size x 20
        
    # - Layer 2 - #
    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(norm1, [4, 4, 20, 20], [1, 1, 1, 1])
        norm2 = tf.nn.lrn(conv2, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        # output size img_size x img_size x 20
        
    # - Layer 3 - #
    with tf.variable_scope('local1') as scope:
        weights = weight_var([FLAGS.input_size * FLAGS.input_size * 20, 1024], 0.0005)
        biases = bias_var([1024])
        
        conv2_flat = tf.reshape(norm2, [-1, FLAGS.input_size * FLAGS.input_size * 20])
        local3 = tf.nn.relu(tf.matmul(conv2_flat, weights) + biases)
        drop3 = tf.nn.dropout(local3, keep_prob)
        return drop3
    
    
# ----------------- SAVE / RESTORE  ----------------- #

def weights_saver():
    with tf.variable_scope("conv1", reuse=True):
        conv1_weights = tf.get_variable("weights")
        conv1_biases = tf.get_variable("biases")
    with tf.variable_scope("conv2", reuse=True):
        conv2_weights = tf.get_variable("weights")
        conv2_biases = tf.get_variable("biases")
    with tf.variable_scope("local1", reuse=True):
        local1_weights = tf.get_variable("weights")
        local1_biases = tf.get_variable("biases")
    
    return tf.train.Saver([conv1_weights, conv2_weights, local1_weights,
                           conv1_biases, conv2_biases, local1_biases])
