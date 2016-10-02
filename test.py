import utils
import input_data
import numpy
import cv2
import time
import tensorflow as tf
import csv
import os
import localizer as nn

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('image_size', 128, 'width and height of the input images')
flags.DEFINE_integer('label_size', 32, 'width and height of the input images')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/localizer', 'path to checkpoint')

def eval_test():
    size = 32 * 32
    
    mask = numpy.zeros([size])
    label = numpy.zeros([size])
    

    mask[0:36] = 0.5
    label[0:36] = 1
    
    print "sum mask: " + str(numpy.sum(mask))
    print "sum label: " + str(numpy.sum(label))
    
    
    l2_error = 0
    for i in xrange(size):
        l2_error += (mask[i] - label[i]) ** 2
        
                
    print "l2 error: " + str(l2_error)
    
    
    
    
# ========================================== #

def localizer_test():
    sess = tf.Session()
    
    # ---------- create model ----------------#
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("float", shape=[None, FLAGS.label_size * FLAGS.label_size])
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # use for 'network_simple' model
    model  = nn.create(x, keep_prob)
    # use for 'network' model
    #model  = nn.create_network(x)
    
    data = input_data.Data((FLAGS.image_size, FLAGS.image_size), (FLAGS.label_size, FLAGS.label_size))
    
    data.add_examples("../data/test/test_input.png", 4, 100, None)
    data.add_labels("../data/test/test_label.png", 4, 100)
    
    data.finalize()
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
        
    batch_xs, batch_ys = data.next_batch(4)
    feed = {x:batch_xs, y_:batch_ys, keep_prob:1.0}
    output = sess.run(model, feed_dict = feed)
        

    img = numpy.zeros([64, 128])

    for i in xrange(4):
        label = numpy.reshape(batch_ys[i], [32,32])
        mask = numpy.reshape(output[i], [32,32])
        
        label = cv2.normalize(label, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        img[0:32, i * 32 : (i+1) * 32] = label
        img[32:64, i * 32 : (i+1) * 32] = mask
    
    cv2.imwrite('../output/test.png', img)
    
    
# ========================================== #


if __name__ == '__main__':
    localizer_test()
