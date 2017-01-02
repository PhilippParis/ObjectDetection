import numpy as np
import cv2
import time
import tensorflow as tf
import csv
import os
import datetime

from models import simple_classifier as nn
from utils import utils
from utils import input_data
from utils.rect import Rect

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('window_size', 24, 'width and height of the sliding window')
flags.DEFINE_integer('image_size', 24, 'width and height of the input images')
flags.DEFINE_integer('scale_factor', 1.25, 'scale factor')

flags.DEFINE_string('test', 'eval_1', 'name of the test image')

flags.DEFINE_boolean('show_ground_truth', True, 'show ground truth data in output image')

flags.DEFINE_integer('step_size', 10, 'sliding window step size')
flags.DEFINE_integer('tol', 25, 'max tolerated distance error between ground truth crater and detected crater')
flags.DEFINE_integer('nms_threshold', 512, 'threshold area for the non maximum suppression')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/classifier_simple_with_drop_lrn', 'path to tensorflow checkpoint dir')
flags.DEFINE_string('ground_truth_dir','../data/eval/data/', 'path to ground truth data dir')
flags.DEFINE_string('input_dir','../data/eval/', 'path to input images')
flags.DEFINE_string('output_dir','../output/results/classifier_simple_with_drop_lrn/', 'path to output dir')

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
    
    output = []
    for i in xrange(0, len(delta)):
        output.append([])
    
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, w_size, FLAGS.scale_factor, (50,50), (120,120)):
        feed = {x:windows, keep_prob:1.0}
        out = sess.run(tf.nn.softmax(model), feed_dict = feed)
        for i in range(0, len(out)):   
            for j in xrange(0, len(delta)):
                if out[i][1] - out[i][0] > delta[j]:
                    output[j].append((coords[i], out[i][1] - out[i][0]))
    
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
        return item[1]
    
    # sort boxes according to detection scores
    sort = sorted(bboxes, reverse=True, key=get_score)

    for i in range(len(sort)): 
        j = i + 1
        while (j < len(sort)):
            if i >= len(sort) or j >= len(sort):
                break
            if Rect.overlap(sort[i][0], sort[j][0]) > th:
                del sort[j]
                j -= 1
            j += 1 
            

    return [r.center() for (r,score) in sort]

# ============================================================= #

def ocv_grouping(bboxes, min_neighbours):
    
    # cluster candidate bboxes into n classes, each class represents equvalent rectangles
    class_index = [-1] * len(bboxes)
    num_classes = 0
    
    for i in xrange(len(bboxes)):
        if class_index[i] != -1:
            continue
        
        class_index[i] = num_classes
        for j in xrange(i + 1, len(bboxes)):
            if Rect.compare(bboxes[i][0], bboxes[j][0]):
                class_index[j] = num_classes
            
        num_classes += 1    
            
    # calc average bounding box of each class
    avg_bboxes = []
    class_count = []
    for i in xrange(num_classes):
        avg_bboxes.append(Rect(0,0,0,0))
        class_count.append(0)
    
    for i in xrange(len(bboxes)):
        j = class_index[i]
        avg_bboxes[j] += bboxes[i][0]
        class_count[j] += 1
    
    
    # select valid bboxes
    result = []
    for i in xrange(num_classes):    
        # reject classes with count < min_neighbours
        if class_count[i] < min_neighbours:
            continue
        
        # calc average
        avg_bboxes[i] /= class_count[i]
        
        # reject average bounding boxes which are inside other candidates
        reject = False
        for j in range(num_classes):
            if class_count[j] < min_neighbours:
                continue
            
            if i != j and avg_bboxes[j].contains(avg_bboxes[i]):
                reject = True
                break;
        
        if not reject:
            result.append(avg_bboxes[i])
            
    return result  

# ============================================================= #

def main(_):
    # ---------- create model ----------------#
    x           = tf.placeholder("float", shape=[None, FLAGS.window_size * FLAGS.window_size])
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
    img = cv2.copyMakeBorder(img, FLAGS.window_size, FLAGS.window_size, FLAGS.window_size, FLAGS.window_size, cv2.BORDER_REPLICATE) 
    
    delta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999]
    
    start = time.time()
    bboxes = detect(model, x, keep_prob, img, delta)

    elapsed_detect = time.time() - start
    print 'detection time: %d' % (elapsed_detect)

    # ------------- evaluation --------------#
    global_step = tf.train.global_step(sess, global_step)
    
    ground_truth_data = utils.csv_to_list(FLAGS.ground_truth_dir + FLAGS.test + '.csv', True)
    ground_truth_data = [(x + FLAGS.window_size,y + FLAGS.window_size,rad) for (x,y,rad,lbl) in ground_truth_data]
    
    
    for i in xrange(0, len(delta)):
        start = time.time()
        detected = non_maximum_suppression(bboxes[i], FLAGS.nms_threshold)
        elapsed_non_max = time.time() - start
        
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
            cv2.rectangle(img_out, (x,y), (x + size[0], y + size[1]), [200,200,200], 2)
            
        output_file = FLAGS.test + '_' + str(global_step) + 'its_' + str(FLAGS.step_size) + 'step_' + str(delta[i]) + 'threshold_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        """
        # csv output
        with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, str(elapsed_detect + elapsed_non_max),
                            str(global_step), str(len(ground_truth_data)), str(delta[i]),
                            str(len(detected)), str(FLAGS.step_size), str(tp), str(fp), str(fn), 
                            str(pr), str(re), str(f1)])
        

if __name__ == '__main__':
    tf.app.run()

