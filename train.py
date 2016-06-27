import utils
import input_data
import numpy
import cv2
import time
import tensorflow as tf
import network_simple as nn
import csv
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 28, 'width and height of the input images')
flags.DEFINE_integer('batch_size', 50, 'training batch size')
flags.DEFINE_integer('max_steps', 10000, 'number of steps to run trainer')

flags.DEFINE_string('checkpoint_path','checkpoints/simple_nn', 'path to checkpoint')
flags.DEFINE_string('log_dir','/tmp/object_detection_logs', 'path to log directory')

sess = tf.InteractiveSession()

def import_data():
    """
    Returns a Data object which contains training data
    """
    # Import data
    data = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    
    data.add('../images/data/1.csv', '../images/1.tif')
    data.add('../images/data/2.csv', '../images/2.tif')
    data.add('../images/data/3.csv', '../images/3.tif')
    data.add('../images/data/4.csv', '../images/4.tif')
    """
    data.add('../images/data/Winter.csv',
             '../images/Winter.tif')
    """
    data.finalize()     
    
    print '(datasets, positive, negative)'
    print data.info()
    print ''
    
    return data


# ============================================================= #


def train_model(model, data, x, y_, keep_prob):
    """
    trains the model
    Args:
        model: model to train
        data: training datasets
        x: input data placeholder
        y_: desired output placeholder
        keep_prob: keep probability placeholder    
    """
    global sess
    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(model, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.scalar_summary('accuracy', accuracy)
    
    # merge summaries and write them to /tmp/crater_logs
    merged_summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # deep model
    #loss = nn.loss(model, y_)
    
    # training
    with tf.name_scope('train'):
        # for simple nn
        train_step = nn.train(model, y_)
        #train_step = nn.train(data.num_examples, global_step, loss)
    
    tf.initialize_all_variables().run()
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_path) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))  
    
    # ------------- train --------------------#
    for i in xrange(FLAGS.max_steps):
        batch_xs, batch_ys = data.next_batch(FLAGS.batch_size)
        
        # train batch
        feed = {x:batch_xs, y_:batch_ys, keep_prob:0.5}
        sess.run([global_step.assign_add(1),train_step], feed_dict = feed)
        #_, loss_value = sess.run([train_step, loss], feed_dict = feed)
        step = tf.train.global_step(sess, global_step)
        
        #assert not numpy.isnan(loss_value)
    
        if step % 100 == 0:
            # record summary data and accuracy
            feed = {x:batch_xs, y_:batch_ys, keep_prob:1.0}
            summary_str, acc = sess.run([merged_summary, accuracy], feed_dict = feed)
            writer.add_summary(summary_str, step)
            print 'Accuracy at step %s: %s' % (step, acc)
            
        if step % 1000 == 0 or (i + 1) == FLAGS.max_steps:
            saver.save(sess, FLAGS.checkpoint_path + '/model.ckpt', global_step = step)
            
    
    
# ============================================================= #    


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
    #model  = nn.create_network(x)
    
    # ---------- train model -----------------#
    
    start = time.time()
    train_model(model, data, x, y_, keep_prob)
    print 'training time: %d' % (time.time() - start)
    
if __name__ == '__main__':
    tf.app.run()
