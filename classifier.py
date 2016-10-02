"""
based on the tensorflow tutorial:
https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/mnist.py
"""

import model
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

NUM_CLASSES = 2

# ----------------- MODEL ----------------- #

def create(network_input, keep_prob):
    logits = model.create(network_input, keep_prob)
        
    #- layer 8 -#
    with tf.variable_scope('classifier') as scope:
        weights = model.weight_var([2048, NUM_CLASSES], 0.0)
        biases = model.bias_var([NUM_CLASSES])
        
        # do not use tf.nn.softmax here -> loss uses tf.nn.sparse_softmax_cross_entropy_with_logits
        classifier = tf.add(tf.matmul(logits, weights), biases)
        model.variable_summaries(classifier, 'network-output')
        return classifier

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

# --------------------------------------------#

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

# ----------------- SAVE / RESTORE  ----------------- #

def weights_saver():
    return model.weights_saver()
