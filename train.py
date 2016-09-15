"""
trains the 4layer convolutional network

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
import cv2
import time
import tensorflow as tf
import model as nn
import csv
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 64, 'width and height of the input images')
flags.DEFINE_integer('batch_size', 128, 'training batch size')
flags.DEFINE_integer('max_steps', 3000, 'number of steps to run trainer')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.75, 'Keep probability for training dropout.')

flags.DEFINE_string('checkpoint_path','../output/checkpoints/7layer', 'path to checkpoint')
flags.DEFINE_string('log_dir','../output/log/7layer', 'path to log directory')
flags.DEFINE_string('output_file','../output/results/7layer/train.csv', 'path to log directory')

sess = 0

def import_data():
    """
    Returns training and evaluation data sets
    """
    train_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    eval_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)

    train_set.add_from_single_image("../data/train/10414_positives.png", 64, 64, 1, 10414, 100)
    train_set.add_from_single_image("../data/train/20038_negatives.png", 64, 64, 0, 20038, 100)
    eval_set.add_from_single_image("../data/eval/1510_positives.png", 64, 64, 1, 1510, 100)
    eval_set.add_from_single_image("../data/eval/4054_negatives.png", 64, 64, 0, 4054, 100)
    
    train_set.finalize()
    eval_set.finalize()
    
    t_total, t_pos, t_neg = train_set.info()
    e_total, e_pos, e_neg = eval_set.info()
    
    utils.print_to_file(FLAGS.output_file, 'training')
    utils.print_to_file(FLAGS.output_file, 'total, positive, negative')
    utils.print_to_file(FLAGS.output_file, str(t_total) + ',' + str(t_pos) + ',' + str(t_neg))
    
    utils.print_to_file(FLAGS.output_file, 'evaluation')
    utils.print_to_file(FLAGS.output_file, 'total, positive, negative')
    utils.print_to_file(FLAGS.output_file, str(e_total) + ',' + str(e_pos) + ',' + str(e_neg))

    return train_set, eval_set

# ============================================================= #

def evaluation(step, data_set, top_k_op, x, y_, keep_prob):
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
        top_k_op = tf.nn.in_top_k(model, y_, 1)
        
    # training ops
    with tf.name_scope('train'):
        loss = nn.loss(model, y_)
        train_step = nn.train(loss, global_step)
    
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
        if step % 500 == 0:
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
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("int32")
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    
    # use for 'network_simple' model
    model = nn.model(x, keep_prob)
    
    utils.print_to_file(FLAGS.output_file,'batch size, learning rate, drop out, image size')
    utils.print_to_file(FLAGS.output_file, str(FLAGS.batch_size) + ',' + str(FLAGS.learning_rate) + ',' + str(FLAGS.dropout) + ',' + str(FLAGS.image_size))
    
    # ---------- train model -----------------#
    train_model(model, train_set, eval_set, x, y_, keep_prob)
    
if __name__ == '__main__':
    tf.app.run()
