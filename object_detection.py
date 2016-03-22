import network as nn
import tensorflow as tf
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 28, 'width and height of the input images')
flags.DEFINE_integer('batch_size', 50, 'training batch size')
flags.DEFINE_integer('max_steps', 100, 'number of steps to run trainer')

def main(_):
    # Import data
    data = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    data.add('../space_crater_dataset/data/1_24.csv',
             '../space_crater_dataset/images/tile1_24.pgm')
    data.add('../space_crater_dataset/data/1_25.csv',
             '../space_crater_dataset/images/tile1_25.pgm')
    data.add('../space_crater_dataset/data/2_24.csv',
             '../space_crater_dataset/images/tile2_24.pgm')
    data.add('../space_crater_dataset/data/2_25.csv',
             '../space_crater_dataset/images/tile2_25.pgm')
    data.add('../space_crater_dataset/data/3_24.csv',
             '../space_crater_dataset/images/tile3_24.pgm')
    data.add('../space_crater_dataset/data/3_25.csv',
             '../space_crater_dataset/images/tile3_25.pgm')
    
    data.finalize()
    
    
    # start session
    sess = tf.InteractiveSession()


    # create model
    network_input   = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    desired_output  = tf.placeholder("float", shape=[None, 2])
    keep_prob       = tf.placeholder("float")
    #network_output  = nn.create_network(network_input, keep_prob, FLAGS.image_size)
    network_output  = nn.create_network(network_input)


    # training
    with tf.name_scope('xent'):
        cross_entropy = -tf.reduce_sum(desired_output * tf.log(network_output))
        _ = tf.scalar_summary('cross entropy', cross_entropy)
        
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(desired_output, 1), tf.argmax(network_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.scalar_summary('accuracy', accuracy)


    # merge summaries and write them to /tmp/crater_logs
    merged_summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/crater_logs", sess.graph_def)
    tf.initialize_all_variables().run()

    # train the model
    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = data.next_batch(FLAGS.batch_size)
    
        if i % 100 == 0:
            # record summary data and accuracy
            feed = {network_input:batch_xs, desired_output:batch_ys, keep_prob:1.0}
            summary_str, acc = sess.run([merged_summary, accuracy], feed_dict = feed)
            writer.add_summary(summary_str, i)
            print 'Accuracy at step %s: %s' % (i, acc)
        else:
            # train batch
            feed = {network_input:batch_xs, desired_output: batch_ys, keep_prob:0.5}
            sess.run(train_step, feed_dict = feed)

    """
    feed = {network_input:data.images, desired_output: data.labels, keep_prob:0.5}
    acc = sess.run(accuracy, feed_dict = feed)
    print 'Overall Accuracy: %s' % (acc)
    """

if __name__ == '__main__':
    tf.app.run()
