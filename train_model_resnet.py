"""
trains the residual network

usage:
1. edit import_data() to import your training/evaluation data
2. python2 train_model_resnet.py    -image_size="width/height of examples" 
                                    -batch_size="batch_size" 
                                    -max_steps="training steps" 
                                    -checkpoint_path="directory to store tensorflow checkpoints"
                                    -log_dir="tensorboard logdir"
"""

import utils
import input_data
import numpy
import res_net
import cv2
import time
import tensorflow as tf
import csv
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 64, 'width and height of the input images')
flags.DEFINE_integer('batch_size', 50, 'training batch size')
flags.DEFINE_integer('max_steps', 200, 'number of steps to run trainer')

flags.DEFINE_string('checkpoint_path','../output/checkpoints/resnet', 'path to checkpoint')
flags.DEFINE_string('log_dir','../output/log/resnet', 'path to log directory')

sess = tf.InteractiveSession()

def import_data():
    """
    Returns training and evaluation data sets
    """
    train_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    train_set.add_from_single_image("../data/train/10414_positives.png", FLAGS.image_size, 
                                    FLAGS.image_size, [0,1], 10414, 100)
    train_set.add_from_single_image("../data/train/20038_negatives.png", FLAGS.image_size, 
                                    FLAGS.image_size, [1,0], 20038, 100)
    
    eval_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    eval_set.add_from_single_image("../data/eval/1510_positives.png", FLAGS.image_size, 
                                    FLAGS.image_size, [0,1], 1510, 100)
    
    eval_set.add_from_single_image("../data/eval/4054_negatives.png", FLAGS.image_size, 
                                    FLAGS.image_size, [1,0], 4054, 100)
    train_set.finalize()
    eval_set.finalize()
    
    print '(datasets, positive, negative)'
    print train_set.info()
    print eval_set.info()
    
    return train_set, eval_set

# ============================================================= #


def train_model(model, train_set, eval_set, x, y_, is_training):
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
        tf.scalar_summary('accuracy', accuracy)
    
    # merge summaries and write them to /tmp/crater_logs
    merged_summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # training
    with tf.name_scope('train'):
        train_step = res_net.train(model, y_)
    
    tf.initialize_all_variables().run()
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_path) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))  
    
    # ------------- train --------------------#
    for i in xrange(FLAGS.max_steps):
        # train mini batches
        batch_xs, batch_ys = train_set.next_batch(FLAGS.batch_size)
        feed = {x:batch_xs, y_:batch_ys, is_training:True}
        sess.run([train_step], feed_dict = feed)
        
        # increment global step count
        sess.run(global_step.assign_add(1))
        step = tf.train.global_step(sess, global_step)
        
        # validate 
        if step % 10 == 0:
            feed = {x:eval_set.images, y_:eval_set.labels, is_training:False}
            summary_str, acc = sess.run([merged_summary, accuracy], feed_dict = feed)
            writer.add_summary(summary_str, step)
            print 'Accuracy at step %s: %s' % (step, acc)
            
        # save model
        if step % 1000 == 0 or (i + 1) == FLAGS.max_steps:
            saver.save(sess, FLAGS.checkpoint_path + '/model.ckpt', global_step = step)
            
    
# ============================================================= #    


def main(_):
    # ---------- import data ----------------#
    train_set, eval_set = import_data()

    # ---------- create model ----------------#
    
    # model input placeholder
    x           = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder(tf.float32, shape=[None, 2])
    # is_training indicator placeholder
    is_training = tf.placeholder(tf.bool, name='is_training') 
    
    model = res_net.create_model(x, 2, is_training)
    
    
    # ---------- train model -----------------#
    start = time.time()
    train_model(model, train_set, eval_set, x, y_, is_training)
    print 'training time: %d' % (time.time() - start)
    
if __name__ == '__main__':
    tf.app.run()
