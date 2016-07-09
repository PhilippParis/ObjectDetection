import tensorflow as tf

def conv_layer(x, kernel, stride, padding, activation):
    x = tf.nn.conv2d(x, kernel, stride, padding=padding)
    x = batch_norm(x, is_training)
    if activation != None:
        x = activation(x)
    return x

# ============================================================= #

def max_pool(x, kernel, stride, padding):
    return tf.nn.max_pool(x, kernel, stride, padding=padding)

# ============================================================= #

def avg_pool(x, kernel, stride, padding):
    return tf.nn.avg_pool(x, kernel, stride, padding=padding)

# ============================================================= #

def fc_soft_max(x, n_out):
    n_in = x.get_shape()[1]
    weights = tf.get_variable('weights', [n_in, n_out], initializer = tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable('biases', [n_out], initializer = tf.constant_initializer(0.1))
    return tf.soft_max(tf.matmul(x, weights) + biases)
    
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
        with tf.control_dependecies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean, var = tf.cond(is_training, mean_var_with_update, mean_var)
    
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-3)

# ============================================================= #

def bottleneck_block(x, num_filters, activation):
    short_cut = conv2d(x, [1, 1, 1, num_filters[2]], [1, 1, 1, 1] 'SAME', None)
    
    x = conv2d(x, [1, 1, 1, num_filters[0]], [1, 1, 1, 1], 'SAME', activation)
    x = conv2d(x, [3, 3, 1, num_filters[1]], [1, 1, 1, 1], 'SAME', activation)
    x = conv2d(x, [1, 1, 1, num_filters[2]], [1, 1, 1, 1], 'SAME', None)
    
    return activation(short_cut + x)
    

def create_model(x, n_out, is_training, activation = tf.nn.relu):
    # ----- reshape input ----- #
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])
        
    # - Convolution - Batch Norm - reLU - #
    x = conv2d(x, [7, 7, 1, 64], [1, 2, 2, 1], 'SAME', activation);

    # - max pool - #
    x = max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME') 
    
    # - bottleneck blocks - #
    x = bottleneck_block(x, [64, 64, 256], activation)
    x = bottleneck_block(x, [64, 64, 256], activation)
    x = bottleneck_block(x, [64, 64, 256], activation)
    
    # - bottleneck blocks - #
    x = bottleneck_block(x, [128, 128, 512], activation)
    x = bottleneck_block(x, [128, 128, 512], activation)
    x = bottleneck_block(x, [128, 128, 512], activation)
    
    # - bottleneck blocks - #
    x = bottleneck_block(x, [256, 256, 1024], activation)
    x = bottleneck_block(x, [256, 256, 1024], activation)
    x = bottleneck_block(x, [256, 256, 1024], activation)
    
    # - average pool - #
    x = avg_pool(x, [1, 7, 7, 1], [1, 1, 1, 1], 'SAME')
    
    # - fully connected layer - #
    x = fc_soft_max(x, n_out)
    
    return x
    
    
