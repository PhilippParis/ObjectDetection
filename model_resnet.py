import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers
flags = tf.app.flags
FLAGS = flags.FLAGS

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

def conv2d(x, ksize, stride, padding, activation, is_training, use_bias):
    weights = weight_var(ksize)
    conv = tf.nn.conv2d(x, weights, stride, padding=padding)
    
    if use_bias:
        bias = bias_var(ksize[3])
        conv = tf.nn.bias_add(conv, bias)

    conv = batch_norm(conv, is_training)
    
    if activation != None:
        return activation(conv)
    else:
        return conv
    
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

# ============================================================= #

def max_pool(x, kernel, stride, padding):
    return tf.nn.max_pool(x, kernel, stride, padding=padding)

# ============================================================= #

def avg_pool(x, kernel, stride, padding):
    return tf.nn.avg_pool(x, kernel, stride, padding=padding)

# ============================================================= #

def fc_softmax(x, n_out):
    # - reshape layer to conventional 1D layer - #
    shape = x.get_shape().as_list()
    n_in = shape[1] * shape[2] * shape[3]
    x = tf.reshape(x, [-1, n_in])
    
    # hidden layer with 1024 nodes
    with tf.variable_scope('hidden-layer') as scope:
        weights = weight_var([n_in, 1024])
        biases = bias_var([1024])
        hidden_layer = tf.nn.relu(tf.matmul(x, weights) + biases)
                       
    # output layer with softmax     
    with tf.variable_scope('output-layer') as scope:
        weights = weight_var([1024, n_out])
        biases = bias_var([n_out])
        return tf.nn.softmax(tf.matmul(hidden_layer, weights) + biases)
    
# ============================================================= #
    
def batch_norm(x, is_training):
    return tf.cond(is_training, 
                   lambda: layers.batch_norm(x, scale=True, is_training=True), 
                   lambda: layers.batch_norm(x, scale=True, is_training=False))

# ============================================================= #

def bottleneck_block(x, num_filters, activation, is_training):
    num_filters_in = x.get_shape()[3]
    
    # - adjust short cut connections - #
    short_cut = tf.identity(x)
    if (num_filters_in != num_filters[2]):
        with tf.variable_scope('short_cut') as scope:
            short_cut = tf.nn.conv2d(x, weight_var([1, 1, num_filters_in, num_filters[2]]), [1, 1, 1, 1], 'SAME')
            
    
    # - convolution layers - #
    with tf.variable_scope('reduce') as scope:
        x = conv2d(x, [1, 1, num_filters_in, num_filters[0]], [1, 1, 1, 1], 'VALID', activation, is_training, FLAGS.use_bias)
    with tf.variable_scope('apply') as scope:
        x = conv2d(x, [3, 3, num_filters[0], num_filters[1]], [1, 1, 1, 1], 'SAME', activation, is_training, FLAGS.use_bias)
    with tf.variable_scope('restore') as scope:
        x = conv2d(x, [1, 1, num_filters[1], num_filters[2]], [1, 1, 1, 1], 'VALID', None, is_training, FLAGS.use_bias)
    
    # - apply short cut connection - #
    return activation(short_cut + x)
    
# ============================================================= #

def residual_block(x, n_out, act, is_training, scope):
    with tf.variable_scope(scope):
        n_in = x.get_shape()[3]
        
        with tf.variable_scope('short_cut') as scope:
            if (n_in != n_out):
                short_cut = tf.nn.conv2d(x, weight_var([3, 3, n_in, n_out]), [1, 2, 2, 1], 'SAME')
            else:
                short_cut = tf.identity(x)
            
        with tf.variable_scope('res_1') as scope:
            if (n_in != n_out):
                y = conv2d(x, [3, 3, n_in, n_out], [1, 2, 2, 1], 'SAME', act, is_training, FLAGS.use_bias)
            else:
                y = conv2d(x, [3, 3, n_out, n_out], [1, 1, 1, 1], 'SAME', act, is_training, FLAGS.use_bias)
        
            
        # - convolution layers - #
        with tf.variable_scope('res_2') as scope:
            y = conv2d(y, [3, 3, n_out, n_out], [1, 1, 1, 1], 'SAME', None, is_training, FLAGS.use_bias)
            
        y = y + short_cut
    return act(y)

# ============================================================= #

def create_model(x, n_out, is_training, activation = tf.nn.relu):
    """
    Builds a residual network model with bottleneck layers (based on <http://arxiv.org/abs/1512.03385>).
    The model consists out of 22 layers (2 x 3 bottleneck blocks (each 3 layers) plus additional layers)
    
    Args:
        x: input, 4D Tensor ([batch size, height, width, channels]) or 2D square Tensor
        n_out: number of output units
        is_training: indicator if the model will be used for training or testing
        activation: activation function
    
    Returns:
        residual network output
    """   
    
    # ----- reshape input ----- #
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 2:
        ndim = int(np.sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])
    
    # - Convolution - Batch Norm - reLU - #
    with tf.variable_scope('conv1') as scope:
        y = conv2d(x, [3, 3, 1, 32], [1, 2, 2, 1], 'SAME', activation, is_training, FLAGS.use_bias)

    # - max pool - #
    with tf.variable_scope('max_pool1') as scope:
        y = max_pool(y, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME') 
    
    # - bottleneck blocks - #
    '''
    with tf.variable_scope('block1_1') as scope:
        y = bottleneck_block(y, [32, 32, 128], activation, is_training)
    with tf.variable_scope('block1_2') as scope:
        y = bottleneck_block(y, [32, 32, 128], activation, is_training)
    with tf.variable_scope('block1_3') as scope:
        y = bottleneck_block(y, [32, 32, 128], activation, is_training)
    with tf.variable_scope('block2_1') as scope:
        y = bottleneck_block(y, [64, 64, 256], activation, is_training)
    with tf.variable_scope('block2_2') as scope:
        y = bottleneck_block(y, [64, 64, 256], activation, is_training)
    with tf.variable_scope('block2_3') as scope:
        y = bottleneck_block(y, [64, 64, 256], activation, is_training)
    '''
    
    # - residual blocks - #
    y = residual_block(y, 32, activation, is_training, 'block1_1')
    y = residual_block(y, 32, activation, is_training, 'block1_2') 
    y = residual_block(y, 32, activation, is_training, 'block1_3') 
        
    y = residual_block(y, 64, activation, is_training, 'block2_1')
    y = residual_block(y, 64, activation, is_training, 'block2_2')    
    y = residual_block(y, 64, activation, is_training, 'block2_3')
    
    y = residual_block(y, 128, activation, is_training, 'block3_1')
    y = residual_block(y, 128, activation, is_training, 'block3_2')    
    y = residual_block(y, 128, activation, is_training, 'block3_3')
    
    # - average pool - #
    with tf.variable_scope('avg_pool') as scope:
        y = avg_pool(y, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
    
    # - fully connected layer - #
    with tf.variable_scope('fc') as scope:
        y = fc_softmax(y, n_out)
    
    return y
    
# ============================================================= #

def train(y, y_):
    '''
    Args:
        y: actual output
        y_: desired output
    Returns:
         returns an optimizer for the res-net
    '''
    cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
    tf.scalar_summary('cross entropy', cross_entropy)
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
