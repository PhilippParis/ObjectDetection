
import model
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

NUM_CLASSES = 2

# ----------------- NETWORK ----------------- #

def create(network_input, keep_prob):
    """
    Builds the convolutional network model (2x conv, 1x fully connected, softmax output)
    
    Returns:
        logits
    """        
    input_reshaped = tf.reshape(network_input, [-1, FLAGS.image_size, FLAGS.image_size, 1])
    
    # - Layer 1 - #
    with tf.variable_scope('conv1') as scope:
        conv1 = model.conv2d(input_reshaped, [4, 4, 1, 20], [1, 1, 1, 1])
        norm1 = tf.nn.lrn(conv1, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        # output size img_size x img_size x 20
        
    # - Layer 2 - #
    with tf.variable_scope('conv2') as scope:
        conv2 = model.conv2d(norm1, [4, 4, 20, 20], [1, 1, 1, 1])
        norm2 = tf.nn.lrn(conv2, 3, bias=2.0, alpha=0.0001, beta=0.75, name='norm1')
        # output size img_size x img_size x 20
        
    # - Layer 3 - #
    with tf.variable_scope('local1') as scope:
        weights = model.weight_var([FLAGS.image_size * FLAGS.image_size * 20, 500], 0.004)
        biases = model.bias_var([500])
        
        conv2_flat = tf.reshape(norm2, [-1, FLAGS.image_size * FLAGS.image_size * 20])
        local3 = tf.nn.relu(tf.matmul(conv2_flat, weights) + biases)
        drop3 = tf.nn.dropout(local3, keep_prob)
     
    #- layer 4 -#
    with tf.variable_scope('classifier') as scope:
        weights = model.weight_var([500, NUM_CLASSES], 0.0)
        biases = model.bias_var([NUM_CLASSES])
        
        # do not use tf.nn.softmax here -> loss uses tf.nn.sparse_softmax_cross_entropy_with_logits
        classifier = tf.add(tf.matmul(drop3, weights), biases)
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
