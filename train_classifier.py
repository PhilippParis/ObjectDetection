"""
trains the classifier network
"""

import numpy
import cv2
import time
import tensorflow as tf
import csv
import os

from utils import utils
from utils import input_data
from models import classifier as classifier

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_size', 24, 'width and height of the input images')
flags.DEFINE_integer('batch_size', 256, 'training batch size')
flags.DEFINE_integer('max_steps', 1000, 'number of steps to run trainer')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.75, 'Keep probability for training dropout.')

flags.DEFINE_string('checkpoint_path','../output/checkpoints/classifier', 'path to checkpoint')
flags.DEFINE_string('log_dir','../output/log/classifier', 'path to log directory')
flags.DEFINE_string('output_file','../output/results/classifier/train.csv', 'path to log directory')

sess = 0

def import_data():
    """
    Returns training and evaluation data sets
    """
    train_set = input_data.Data((FLAGS.input_size, FLAGS.input_size), (1,1))
    eval_set = input_data.Data((FLAGS.input_size, FLAGS.input_size), (1,1))

    train_set.add_examples("../data/train/8300_positives.png", 8300, 100, 1)
    train_set.add_examples("../data/train/20038_negatives.png", 20038, 100, 0)
    
    eval_set.add_examples("../data/train/eval_1510_positives.png", 1510, 100, 1)
    eval_set.add_examples("../data/train/eval_3710_negatives.png", 3710, 100, 0)
    
    train_set.finalize()
    eval_set.finalize()
    
    utils.print_to_file(FLAGS.output_file, 'training: ' + str(train_set.count))
    utils.print_to_file(FLAGS.output_file, 'evaluation: ' + str(eval_set.count))

    return train_set, eval_set

# ============================================================= #

def evaluation(step, data_set, top_k_op, x, y_, keep_prob):
    """
    evaluates current training progress. 
    
    Args: 
        step: current training step
        data_set: evaluation data set
        top_k_op: evaluation operation
        x: input data placeholder
        y_: desired output placeholder
        keep_prob: keep probability placeholder (dropout)
    
    Returns:
        percentage of examples of data_set correctly classified
    """
    
    true_count = 0
    num_examples = 0
    
    for batch_xs, batch_ys, count in data_set.batches(FLAGS.batch_size):
        feed = {x:batch_xs, y_:batch_ys, keep_prob:1.0}
        predictions = sess.run(top_k_op, feed_dict = feed)
        true_count += numpy.sum(predictions)
        num_examples += count
            
    precision = float(true_count) / float(num_examples)
    return precision

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
    # global steps
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # evaluation ops
    with tf.name_scope('test'):
        top_k_op = tf.nn.in_top_k(tf.nn.softmax(model), y_, 1)
        
    # training ops
    with tf.name_scope('train'):
        loss = classifier.loss(model, y_)
        train_step = classifier.train(loss, global_step)
    
    # summary ops
    merged_summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
    
    # init vars
    tf.initialize_all_variables().run(session=sess)
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_path) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))  
    
    
    utils.print_to_file(FLAGS.output_file,'step, test_precision, train_precision')
    
    # ------------- train --------------------#
    for i in xrange(FLAGS.max_steps + 1):
        # train mini batches
        batch_xs, batch_ys = train_set.next_batch(FLAGS.batch_size)    
        feed = {x:batch_xs, y_:batch_ys, keep_prob:FLAGS.dropout}
        
        _, loss_value = sess.run([train_step, loss], feed_dict = feed)
        assert not numpy.isnan(loss_value), 'Model diverged with loss = NaN'
        
        # increment global step count
        step = tf.train.global_step(sess, global_step)
        
        # write summary 
        if step % 500 == 0:
            summary_str = sess.run(merged_summary, feed_dict = feed)
            writer.add_summary(summary_str, step)
            writer.flush()
            
        # evaluation
        if step % 100 == 0:
            test_precision = evaluation(step, eval_set, top_k_op, x, y_, keep_prob)
            train_precision = evaluation(step, train_set, top_k_op, x, y_, keep_prob)
            utils.print_to_file(FLAGS.output_file,str(step) + ',' + str(test_precision) + ',' + str(train_precision))
            
        # save model
        if step % 3000 == 0 or i == FLAGS.max_steps:
            saver.save(sess, FLAGS.checkpoint_path + '/model.ckpt', global_step = step)
            
    
# ============================================================= #    

def main(_):
    global sess
    sess = tf.Session()
    
    # ---------- import data ----------------#
    train_set, eval_set = import_data()

    # ---------- create model ----------------#
    
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.input_size * FLAGS.input_size])
    # desired output placeholder
    y_          = tf.placeholder("int32")
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    
    # use for 'network_simple' model
    model = classifier.create(x, keep_prob)
    
    utils.print_to_file(FLAGS.output_file,'batch size, learning rate, drop out, image size')
    utils.print_to_file(FLAGS.output_file, str(FLAGS.batch_size) + ',' + str(FLAGS.learning_rate) + ',' + str(FLAGS.dropout) + ',' + str(FLAGS.input_size))
    
    # ---------- train model -----------------#
    train_model(model, train_set, eval_set, x, y_, keep_prob)
    
if __name__ == '__main__':
    tf.app.run()
