import tensorflow as tf
import numpy as np


# ============================================================= #

def weight_var(shape):
    weights = tf.get_variable('weights', shape,
                           initializer = tf.truncated_normal_initializer(stddev=0.1))
    variable_summaries(weights, 'weights')
    return weights

# ============================================================= #

def bias_var(shape):
    biases = tf.get_variable('biases', shape,
                           initializer = tf.constant_initializer(0.1), trainable=False)
    variable_summaries(biases, 'biases')
    return biases

# ============================================================= #

def conv2d(x, ksize, stride, padding, activation, is_training):
    weights = weight_var(ksize)
    biases = bias_var(ksize[3])
        
    x = tf.nn.conv2d(x, weights, stride, padding=padding)
    x = batch_norm(x, is_training)
    if activation != None:
        return activation(x + biases)
    else:
        return x + biases
    
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
                       
    weights = weight_var([n_in, n_out])
    biases = bias_var([n_out])
    return tf.nn.softmax(tf.matmul(x, weights) + biases)
    
# ============================================================= #
    
def batch_norm(x, is_training):
    params_shape = x.get_shape()[-1:]
    
    beta = tf.Variable(tf.constant(0.0, shape=params_shape), name='bn_beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=params_shape), name='bn_gamma', trainable=True)
    
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
    def mean_var():
        return ema.average(batch_mean), ema.average(batch_var)
    
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean, var = tf.cond(is_training, mean_var_with_update, mean_var)
    
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

# ============================================================= #

def bottleneck_block(x, num_filters, activation, is_training):
    num_filters_in = x.get_shape()[3]
    
    # - adjust short cut connections - #
    short_cut = x
    if (num_filters_in != num_filters[2]):
        with tf.variable_scope('short_cut') as scope:
            short_cut = conv2d(x, [1, 1, num_filters_in, num_filters[2]], [1, 1, 1, 1], 'VALID', None, is_training)
    
    # - convolution layers - #
    with tf.variable_scope('reduce') as scope:
        x = conv2d(x, [1, 1, num_filters_in, num_filters[0]], [1, 1, 1, 1], 'VALID', activation, is_training)
    with tf.variable_scope('apply') as scope:
        x = conv2d(x, [3, 3, num_filters[0], num_filters[1]], [1, 1, 1, 1], 'SAME', activation, is_training)
    with tf.variable_scope('restore') as scope:
        x = conv2d(x, [1, 1, num_filters[1], num_filters[2]], [1, 1, 1, 1], 'VALID', None, is_training)
    
    # - apply short cut connection - #
    return activation(short_cut + x)
    
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
        x = conv2d(x, [7, 7, 1, 64], [1, 2, 2, 1], 'SAME', activation, is_training)

    # - max pool - #
    with tf.variable_scope('max_pool1') as scope:
        x = max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME') 
    
    # - bottleneck blocks - #
    with tf.variable_scope('block1_1') as scope:
        x = bottleneck_block(x, [64, 64, 256], activation, is_training)
    with tf.variable_scope('block1_2') as scope:
        x = bottleneck_block(x, [64, 64, 256], activation, is_training)
    with tf.variable_scope('block1_3') as scope:
        x = bottleneck_block(x, [64, 64, 256], activation, is_training)
    
    # - bottleneck blocks - #
    with tf.variable_scope('block2_1') as scope:
        x = bottleneck_block(x, [128, 128, 512], activation, is_training)
    with tf.variable_scope('block2_2') as scope:
        x = bottleneck_block(x, [128, 128, 512], activation, is_training)
    with tf.variable_scope('block2_3') as scope:
        x = bottleneck_block(x, [128, 128, 512], activation, is_training)

    # - average pool - #
    with tf.variable_scope('avg_pool') as scope:
        x = avg_pool(x, [1, 7, 7, 1], [1, 1, 1, 1], 'VALID')
    
    # - fully connected layer - #
    with tf.variable_scope('fc') as scope:
        x = fc_softmax(x, n_out)
    
    return x
    
# ============================================================= #

def train(y, y_):
    '''
    Args:
        y: actual output
        y_: desired output
    Returns:
         returns an optimizer for the res-net
    '''
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    return tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
