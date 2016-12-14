import numpy as np
import cv2
import time
import tensorflow as tf
import csv
import os
import datetime

from models import classifier as nn
from utils import utils
from utils  import input_data


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 128, 'width and height of the input images')
flags.DEFINE_integer('window_size', 64, 'width and height of the sliding window')

flags.DEFINE_string('test', 'eval_14', 'name of the test image')

flags.DEFINE_boolean('show_ground_truth', True, 'show ground truth data in output image')

flags.DEFINE_integer('step_size', 10, 'sliding window step size')
flags.DEFINE_integer('tol', 25, 'max tolerated distance error between ground truth crater and detected crater')
flags.DEFINE_integer('nms_threshold', 512, 'threshold area for the non maximum suppression')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/classifier', 'path to tensorflow checkpoint dir')
flags.DEFINE_string('ground_truth_dir','../data/eval/data/', 'path to ground truth data dir')
flags.DEFINE_string('input_dir','../data/eval/', 'path to input images')
flags.DEFINE_string('output_dir','../output/results/classifier/', 'path to output dir')

# start session
sess = tf.InteractiveSession()

def detect(model, x, keep_prob, src, delta):
    """
    object detection via sliding windows
    Args:
        model: tensorflow model which is used for detection
        x: input data placeholder
        keep_prob: keep probability placeholder (dropout)
        src: image to apply the detection
        delta: list of detection thresholds
    Returns:
        2d list of bounding boxes [(left upper x, left upper y, size, score)]
        output[i] are the positively classified sliding windows with threshold delta[i]
    """
    
    global sess
    
    height, width = src.shape
    w_size = (FLAGS.window_size, FLAGS.window_size)
    i_size = (FLAGS.image_size, FLAGS.image_size)
    
    output = []
    for i in xrange(0, len(delta)):
        output.append([])
    
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, w_size, i_size):
        feed = {x:windows, keep_prob:1.0}
        out = sess.run(tf.nn.softmax(model), feed_dict = feed)
        for i in range(0, len(out)):   
            for j in xrange(0, len(delta)):
                if out[i][1] - out[i][0] > delta[j]:
                    output[j].append((coords[i][0] - FLAGS.window_size / 2, coords[i][1] - FLAGS.window_size / 2, FLAGS.window_size, out[i][1] - out[i][0]))
    
    return output

# ============================================================= #
    
def non_maximum_suppression(bboxes, th):
    """
    applies non_maximum_suppression to the set of bounding boxes
    Args:
        bboxes: list of bounding boxes [(left upper x, left upper y, size, score)]
    Returns:
        list of objects [(center x, center y)]
    """
    
    def get_score(item):
        return item[3]
    
    # sort boxes according to detection scores
    sort = sorted(bboxes, reverse=True, key=get_score)

    for i in range(len(sort)): 
        j = i + 1
        while (j < len(sort)):
            if i >= len(sort) or j >= len(sort):
                break
            if overlap(sort[i][0], sort[i][1], sort[i][2], sort[j][0], sort[j][1], sort[j][2]) > th:
                del sort[j]
                j -= 1
            j += 1 
    return [(x + size / 2, y + size / 2) for (x,y,size,score) in sort]

# ============================================================= #

def overlap(xa1, ya1, sa, xb1, yb1, sb):
    xa2 = xa1 + sa
    ya2 = ya1 + sa
    
    xb2 = xb1 + sb
    yb2 = yb1 + sb
    
    return max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    
# ============================================================= #

def main(_):
    # ---------- create model ----------------#
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    model  = nn.create(x, keep_prob)
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    img = utils.getImage(FLAGS.input_dir + FLAGS.test + '.tif')
    img = cv2.copyMakeBorder(img, FLAGS.image_size, FLAGS.image_size, FLAGS.image_size, FLAGS.image_size, cv2.BORDER_REPLICATE) 
    
    delta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999]
    
    start = time.time()
    bboxes = detect(model, x, keep_prob, img, delta)

    elapsed = time.time() - start
    print 'detection time: %d' % (elapsed)

    # ------------- evaluation --------------#
    global_step = tf.train.global_step(sess, global_step)
    
    ground_truth_data = utils.csv_to_list(FLAGS.ground_truth_dir + FLAGS.test + '.csv', True)
    ground_truth_data = [(x + FLAGS.image_size,y + FLAGS.image_size,rad) for (x,y,rad,lbl) in ground_truth_data]
    
    
    for i in xrange(0, len(delta)):
        detected = non_maximum_suppression(bboxes[i], FLAGS.nms_threshold)
        tp, fn, fp, pr, re, f1 = utils.evaluate(ground_truth_data, detected, FLAGS.tol)
        
        # ----------------output ----------------#
        # image output
        """
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        
        for c in detected:
            cv2.circle(img_out, (c[0], c[1]), 25, [0,255,0],3)
                
        for c in ground_truth_data:
            cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
            
        for (x,y,size,score) in bboxes[i]:
            cv2.rectangle(img_out, (x,y), (x + size, y + size), [200,200,200], 2)
            
        output_file = FLAGS.test + '_' + str(global_step) + 'its_' + str(FLAGS.step_size) + 'step_' + 'threshold_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        """
        # csv output
        with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, '-', str(elapsed),
                            str(global_step), str(len(ground_truth_data)), str(delta[i]),
                            str(len(detected)), str(FLAGS.step_size), str(tp), str(fp), str(fn), 
                            str(pr), str(re), str(f1)])
        

if __name__ == '__main__':
    tf.app.run()

