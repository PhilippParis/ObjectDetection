import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# ----------------- HELPERS ----------------- #

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def lrn(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

def weight_var(shape, std):
    return tf.get_variable('weights', shape,
                           initializer = tf.truncated_normal_initializer(stddev=std))

def bias_var(shape, init):
    return tf.get_variable('biases', shape,
                           initializer = tf.constant_initializer(init))

def weight_var_with_decay(shape, std, wd):
    """
    creates an initialized variable with weight decay
    
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a trucated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight 
            decay is not added for this var.
    Returns:
        Variable
    
    """
    
    var = weight_var(shape, std)
    
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        
    return var
    

# ----------------- NETWORK ----------------- # 

def create_network(network_input):
    """
    Builds the convolutional network model
    
    Args:
        network_input: 4D tensor [batch_size, IMAGE_SIZE * IMAGE_SIZE]
    Returns:
        Logits
    """
    
    input_reshaped = tf.reshape(network_input, [-1, FLAGS.image_size, FLAGS.image_size, 1])
    
    # - Layer 1 - #
    
    # convolution
    with tf.variable_scope('conv1') as scope:
        
        # kernel tensor [height, width, in_channels, out_channels] with weight decay
        kernel = weight_var_with_decay(shape=[5, 5, 1, 64], std=1e-4, wd=0.0)
        
        # convolute the input with the kernel using 1 as stride
        conv = conv2d(input_reshaped, kernel)
        
        # bias values 
        biases = bias_var([64], 0.0)
        bias = tf.nn.bias_add(conv, biases)
        
        # apply relu activation function
        conv1 = tf.nn.relu(bias, name=scope.name)
        
        # maxpooling
        pool1 = max_pool_3x3(conv1)
    
        # local response normalization
        norm1 = lrn(pool1)
    
    
    # - Layer 2 - #
    
    # convolution
    with tf.variable_scope('conv2') as scope:
        kernel = weight_var_with_decay(shape=[5, 5, 64, 64], std=1e-4, wd=0.0)
        conv = conv2d(norm1, kernel)
        biases = bias_var([64], 0.1)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        
        # local response normalization
        norm2 = lrn(conv2)
    
        # maxpooling
        pool2 = max_pool_3x3(norm2)
    
    
    # - Layer 3 - #
    
    # local
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        batch_size = tf.shape(pool2)[0]
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, tf.pack([batch_size, dim]))
        
        weights = weight_var_with_decay(shape=[dim, 384], std=0.04, wd=0.004)
        biases = bias_var([384], 0.1)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        
        
    # - Layer 4 - #
    
    # local
    with tf.variable_scope('local4') as scope:
        weights = weight_var_with_decay(shape=[384, 192], std=0.04, wd=0.004)
        biases = bias_var([192], 0.1)
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        
    
    # - Layer 5 - #
    
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_var_with_decay([192, 2], std=(1/192.0), wd=0.0)
        biases = bias_var([2], 0.0)
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        
    return softmax_linear
        

# ----------------- TRAINING ----------------- #

def train(network_output, desired_output):
    """
    with tf.name_scope('xent'):
        cross_entropy = -tf.reduce_sum(desired_output * tf.log(network_output))
        _ = tf.scalar_summary('cross entropy', cross_entropy)
        
    return tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
    """
    
    global_step = tf.Variable(0, trainable=False)
    return tf.train.exponential_decay(0.1,
                                  global_step,
                                  350 * 50,
                                  0.1,
                                  staircase=True)
    
    
    
        
        
