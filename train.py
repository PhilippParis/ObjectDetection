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

flags.DEFINE_string('checkpoint_path','checkpoints/simple_nn_new', 'path to checkpoint')
flags.DEFINE_string('log_dir','/tmp/object_detection_logs', 'path to log directory')

sess = tf.InteractiveSession()

def import_data():
    """
    Returns training and evaluation data sets
    """
    train_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    train_set.add('../data/training/data/train_1.csv', '../data/training/train_1.tif')
    train_set.finalize()   
    
    eval_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    eval_set.add('../data/evaluation/data/eval_1.csv', '../data/evaluation/eval_1.tif')
    eval_set.finalize()     
    
    print '(datasets, positive, negative)'
    print train_set.info()
    print eval_set.info()
    
    return train_set, eval_set

# ============================================================= #


def train_model(model, train_set, eval_set, x, y_, keep_prob):
    """
    trains the model
    Args:
        model: model to train
        train_set: training dataset
        eval_set: evaluation dataset
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
    
    # training
    with tf.name_scope('train'):
        train_step = nn.train(model, y_)
    
    tf.initialize_all_variables().run()
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_path) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))  
    
    # ------------- train --------------------#
    for i in xrange(FLAGS.max_steps):
        train_batch_xs, train_batch_ys = train_set.next_batch(FLAGS.batch_size)
        
        # train batch
        feed = {x:train_batch_xs, y_:train_batch_ys, keep_prob:0.5}
        sess.run([global_step.assign_add(1), train_step], feed_dict = feed)
        step = tf.train.global_step(sess, global_step)
        
        if step % 100 == 0:
            feed = {x:eval_set.images, y_:eval_set.labels, keep_prob:1.0}
            summary_str, acc = sess.run([merged_summary, accuracy], feed_dict = feed)
            writer.add_summary(summary_str, step)
            print 'Accuracy at step %s: %s' % (step, acc)
            
        if step % 1000 == 0 or (i + 1) == FLAGS.max_steps:
            saver.save(sess, FLAGS.checkpoint_path + '/model.ckpt', global_step = step)
            
    
# ============================================================= #    


def main(_):
    # ---------- import data ----------------#
    train_set, eval_set = import_data()

    # ---------- create model ----------------#
    
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("float", shape=[None, 2])
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    
    # use for 'network_simple' model
    model  = nn.create_network(x, keep_prob, FLAGS.image_size)
    
    # ---------- train model -----------------#
    start = time.time()
    train_model(model, train_set, eval_set, x, y_, keep_prob)
    print 'training time: %d' % (time.time() - start)
    
if __name__ == '__main__':
    tf.app.run()
