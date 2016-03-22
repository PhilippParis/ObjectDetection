import tensorflow as tf

# ----------------- HELPERS ----------------- #

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_var(shape):
    return tf.get_variable('weights', shape,
                           initializer = tf.truncated_normal_initializer(stddev=0.1))

def bias_var(shape):
    return tf.get_variable('biases', shape,
                           initializer = tf.constant_initializer(0.1))


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
        weights = weight_var([5, 5, 1, 32])
        biases = bias_var([32])
        
        #_ = tf.histogram_summary("weights1", weights)
        conv = tf.nn.relu(conv2d(input_reshaped, weights) + biases)
        output_layer_1 = max_pool_2x2(conv)
        
        
    # - Layer 2 - #
    with tf.variable_scope('conv2') as scope:
        weights = weight_var([5, 5, 32, 64])
        biases = bias_var([64])
        
        #_ = tf.histogram_summary("weights2", weights)
        conv = tf.nn.relu(conv2d(output_layer_1, weights) + biases)
        output_layer_2 = max_pool_2x2(conv)
        
        
    # - Layer 3 - #
    with tf.variable_scope('local3') as scope:
        size = image_size / 4
        weights = weight_var([size * size * 64, 1024])
        biases = bias_var([1024])
        
        #_ = tf.histogram_summary("weights3", weights)
        output_layer_2_flat = tf.reshape(output_layer_2, [-1, size * size * 64])
        output_layer_3 = tf.nn.relu(tf.matmul(output_layer_2_flat, weights) + biases)
        
        # - Drop out -#
        output_layer_3_drop = tf.nn.dropout(output_layer_3, keep_prob)
        
        
    #- layer 4 -#
    with tf.variable_scope('local4') as scope:
        weights = weight_var([1024, 2])
        biases = bias_var([2])
        
        #tf.histogram_summary("weights4", weights)
        output = tf.nn.softmax(tf.matmul(output_layer_3_drop, weights) + biases)
        #_ = tf.histogram_summary("network-output", output)
        return output
