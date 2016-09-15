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
    weights = tf.get_variable('weights', shape, initializer = tf.truncated_normal_initializer(stddev=0.001))
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

def model(network_input, keep_prob):
    """
    Builds the convolutional network model (2x conv, 2x fully connected with dropout)
    
    Returns:
        logits
    """        
    input_reshaped = tf.reshape(network_input, [-1, FLAGS.image_size, FLAGS.image_size, 1])
    
    # - Layer 1 - #
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(input_reshaped, [5, 5, 1, 64], [1, 2, 2, 1])
        norm1 = tf.nn.lrn(conv1, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        pool1 = max_pool(norm1)
        
    # - Layer 2 - #
    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(pool1, [3, 3, 64, 128], [1, 2, 2, 1])
        norm2 = tf.nn.lrn(conv2, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        pool2 = max_pool(norm2)
        
    # - Layer 3 - #
    with tf.variable_scope('local1') as scope:
        weights = weight_var([2048, 1024], 0.004)
        biases = bias_var([1024])
        
        pool2_flat = tf.reshape(pool2, [-1, 2048])
        local1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)
        drop1 = tf.nn.dropout(local1, keep_prob)
        
    #- layer 4 -#
    with tf.variable_scope('local2') as scope:
        weights = weight_var([1024, 256], 0.004)
        biases = bias_var([256])
        
        local2 = tf.nn.relu(tf.matmul(drop1, weights) + biases)
        drop2 = tf.nn.dropout(local2, keep_prob)
    
    #- layer 5 -#
    with tf.variable_scope('softmax') as scope:
        weights = weight_var([256, 2], 0.0)
        biases = bias_var([2])
        
        softmax_linear = tf.add(tf.matmul(drop2, weights), biases)
        variable_summaries(softmax_linear, 'network-output')
        return softmax_linear


# ----------------- TRAINING ----------------- #
def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.
    Args:
        logits: Logits from model().
        labels: dataset labels. 1-D tensor of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.scalar_summary('cross entropy', cross_entropy_mean)
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    """
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps processed.
    Returns:
        train_op: op for training.
    """
    
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads = opt.compute_gradients(total_loss)
        
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    return train_op
