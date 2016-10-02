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
    input_reshaped = tf.reshape(network_input, [-1, FLAGS.image_size, FLAGS.image_size, 1])
    
    # - Layer 1 - #
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(input_reshaped, [7, 7, 1, 48], [1, 2, 2, 1])
        norm1 = tf.nn.lrn(conv1, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        pool1 = max_pool(norm1)
        # output size 32 x 32 x 48
        
    # - Layer 2 - #
    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(pool1, [5, 5, 48, 128], [1, 1, 1, 1])
        norm2 = tf.nn.lrn(conv2, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm2')
        pool2 = max_pool(norm2)
        # output size 16 x 16 x 128
        
    # - Layer 3 - #
    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d(pool2, [3, 3, 128, 192], [1, 1, 1, 1])
        # output size 16 x 16 x 192
        
    # - Layer 4 - #
    with tf.variable_scope('conv4') as scope:
        conv4 = conv2d(conv3, [3, 3, 192, 192], [1, 1, 1, 1])
        # output size 16 x 16 x 192
        
    # - Layer 5 - #
    with tf.variable_scope('conv5') as scope:
        conv5 = conv2d(conv4, [3, 3, 192, 128], [1, 1, 1, 1])
        pool5 = max_pool(conv5)
        # output size 8 x 8 x 128
        
    # - Layer 6 - #
    with tf.variable_scope('local1') as scope:
        weights = weight_var([8 * 8 * 128, 2048], 0.004)
        biases = bias_var([2048])
        
        pool5_flat = tf.reshape(pool5, [-1, 8 * 8 * 128])
        local6 = tf.nn.relu(tf.matmul(pool5_flat, weights) + biases)
        drop6 = tf.nn.dropout(local6, keep_prob)
     
    #- layer 7 -#
    with tf.variable_scope('local2') as scope:
        weights = weight_var([2048, 2048], 0.004)
        biases = bias_var([2048])
        
        local7 = tf.nn.relu(tf.matmul(drop6, weights) + biases)
        drop7 = tf.nn.dropout(local7, keep_prob)
        return drop7
    
    
# ----------------- SAVE / RESTORE  ----------------- #

def weights_saver():
    with tf.variable_scope("conv1", reuse=True):
        conv1_weights = tf.get_variable("weights")
        conv1_biases = tf.get_variable("biases")
    with tf.variable_scope("conv2", reuse=True):
        conv2_weights = tf.get_variable("weights")
        conv2_biases = tf.get_variable("biases")
    with tf.variable_scope("conv3", reuse=True):
        conv3_weights = tf.get_variable("weights")
        conv3_biases = tf.get_variable("biases")
    with tf.variable_scope("conv4", reuse=True):
        conv4_weights = tf.get_variable("weights")
        conv4_biases = tf.get_variable("biases")
    with tf.variable_scope("conv5", reuse=True):
        conv5_weights = tf.get_variable("weights")
        conv5_biases = tf.get_variable("biases")
    with tf.variable_scope("local1", reuse=True):
        local1_weights = tf.get_variable("weights")
        local1_biases = tf.get_variable("biases")
    with tf.variable_scope("local2", reuse=True):
        local2_weights = tf.get_variable("weights")
        local2_biases = tf.get_variable("biases")
    
    return tf.train.Saver([conv1_weights, conv2_weights, conv3_weights, conv4_weights, conv5_weights, local1_weights, local2_weights,
                           conv1_biases, conv2_biases, conv3_biases, conv4_biases, conv5_biases, local1_biases, local2_biases])
    
