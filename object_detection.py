import network_simple as nn
import tensorflow as tf
import input_data
import utils
import numpy
import cv2
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 28, 'width and height of the input images')
flags.DEFINE_integer('batch_size', 50, 'training batch size')
flags.DEFINE_integer('max_steps', 1000, 'number of steps to run trainer')

# start session
sess = tf.InteractiveSession()

def import_data():
    """
    Returns a Data object which contains training data
    """
    
    # Import data
    data = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    #data.add('../images/Land.csv',
    #         '../images/Land.jpg')
    
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
    
    print '(datasets, positive, negative)'
    print data.info()
    print ''
    
    return data


def overall_accuracy(data, x, y_, keep_prob, acc_op):
    """
    returns the accuracy of the model over all data in 'data'
    Args:
        data: data to test
        x: model input placeholder
        y_: desired output placeholder
        keep_prob: keep probability placeholder
        acc_op: accuracy operation
    Returns:
        accuracy of the model (0 < acc < 1)
    """
    global sess
    feed = {x:data.images, y_: data.labels, keep_prob:0.5}
    return sess.run(acc_op, feed_dict = feed)


def train_model(model, data, x, y_, keep_prob):
    global sess
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(desired_output, 1), tf.argmax(model, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.scalar_summary('accuracy', accuracy)
    
    # merge summaries and write them to /tmp/crater_logs
    merged_summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/crater_logs", sess.graph_def)
    
    # training
    with tf.name_scope('train'):
        train_step = nn.train(model, y_)
    
    tf.initialize_all_variables().run()
    
    # print overall accuracy
    print 'Overall Accuracy: %s' % (overall_accuracy(data, x, y_, keep_prob, accuracy))
    
    # train the model
    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = data.next_batch(FLAGS.batch_size)
    
        if i % 100 == 0:
            # record summary data and accuracy
            feed = {x:batch_xs, y_:batch_ys, keep_prob:1.0}
            summary_str, acc = sess.run([merged_summary, accuracy], feed_dict = feed)
            writer.add_summary(summary_str, i)
            print 'Accuracy at step %s: %s' % (i, acc)
        else:
            # train batch
            feed = {x:batch_xs, y_:batch_ys, keep_prob:0.5}
            sess.run(train_step, feed_dict = feed)
            
    # print overall accuracy
    print 'Overall Accuracy: %s' % (overall_accuracy(data, x, y_, keep_prob, accuracy))
    
    

def detect_objects(model, x, keep_prob):
    global sess
    src = utils.getImage('../space_crater_dataset/images/tile3_25.pgm')
    for windows, coords in utils.slidingWindow(src, 7, (FLAGS.image_size, FLAGS.image_size)):
        
        feed = {x:windows, keep_prob:1.0}
        y = sess.run(model, feed_dict = feed)

        for i in range(0, len(y)):
            if y[i][0] < 0.25 and y[i][1] > 0.75:
                cv2.circle(src,(coords[i][0], coords[i][1]), 2, (0,0,255), 0)
    return src        
    
def main(_):
    # import data
    data = import_data()

    # ---------- create model ----------------#
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("float", shape=[None, 2])
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    
    # use for 'network_simple' model
    model  = nn.create_network(x, keep_prob, FLAGS.image_size)
    # use for 'network' model
    #model  = nn.create_network(network_input)

    tf.initialize_all_variables().run()

    # ---------- train model -----------------#
    start = time.time()
    train_model(model, data, x, y_, keep_prob)
    print 'training time: %d' % (time.time() - start)
    
    # ---------- object detection ------------#
    start = time.time()
    detected = detect_objects(model, x, keep_prob)
    print 'detection time: %d' % (time.time() - start)
    
    utils.showImage('title', detected)
    

if __name__ == '__main__':
    tf.app.run()
