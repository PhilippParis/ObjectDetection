import numpy as np
import cv2
import time
import tensorflow as tf
import csv
import os
import datetime

from models import classifier as nn
from utils import utils
from utils import input_data
from utils.rect import Rect

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_size', 24, 'width and height of the cnn input')
flags.DEFINE_integer('min_window_size', 50, 'minimum window height and width')
flags.DEFINE_integer('max_window_size', 120, 'maximum window height and width')

flags.DEFINE_integer('scale_factor', 1.25, 'scale factor')
flags.DEFINE_string('test', '../data/data/eval/eval_1.png', 'test image path')

flags.DEFINE_integer('step_size', 10, 'sliding window step size')
flags.DEFINE_integer('nms_threshold', 512, 'threshold overlap for the non maximum suppression')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/classifier', 'path to tensorflow checkpoint dir')
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
        2d list of bounding boxes with the corresponding score [i][(Rect, score)]
        output[i] are all positively classified sliding windows with threshold delta[i]
    """
    
    global sess
    
    height, width = src.shape
    input_size = (FLAGS.input_size, FLAGS.input_size)
    min_window_size = (FLAGS.min_window_size, FLAGS.min_window_size)
    max_window_size = (FLAGS.max_window_size, FLAGS.max_window_size)
    
    output = []
    for i in xrange(0, len(delta)):
        output.append([])
    
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, input_size, FLAGS.scale_factor, min_window_size, max_window_size):
        feed = {x:windows, keep_prob:1.0}
        out = sess.run(tf.nn.softmax(model), feed_dict = feed)
        for i in range(0, len(out)):   
            for j in xrange(0, len(delta)):
                if out[i][1] - out[i][0] > delta[j]:
                    output[j].append((coords[i], out[i][1] - out[i][0]))
    
    return output

# ============================================================= #

def main(_):
    image_path = FLAGS.test
    csv_path = os.path.splitext(image_path)[0] + ".csv"
    
    # ---------- create model ----------------#
    x           = tf.placeholder("float", shape=[None, FLAGS.input_size * FLAGS.input_size])
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    model  = nn.create(x, keep_prob)
    
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    img = utils.getImage(image_path)
    img = cv2.copyMakeBorder(img, FLAGS.max_window_size, FLAGS.max_window_size, FLAGS.max_window_size, FLAGS.max_window_size, cv2.BORDER_REPLICATE) 
    
    delta = [0.9]
    
    start = time.time()
    bboxes = detect(model, x, keep_prob, img, delta)

    elapsed_detect = time.time() - start
    print 'detection time: %d' % (elapsed_detect)

    # ------------- evaluation --------------#
    global_step = tf.train.global_step(sess, global_step)
    ground_truth_data = utils.get_ground_truth_data(csv_path)
    ground_truth_data = [(x + FLAGS.max_window_size,y + FLAGS.max_window_size) for (x,y) in ground_truth_data]

    for i in xrange(0, len(delta)):
        start = time.time()
        detected = utils.non_maximum_suppression(bboxes[i], FLAGS.nms_threshold)
        elapsed_non_max = time.time() - start
        
        tp, fn, fp = utils.evaluate(ground_truth_data, detected)
        
        # ----------------output ----------------#
        # image output
        """
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        
        for (r,score) in bboxes[0]:
            cv2.rectangle(img_out, (r.x,r.y), (r.x2(), r.y2()), [200,200,200], 2)
        
        for r in detected:
            cv2.rectangle(img_out, (r.x,r.y), (r.x2(), r.y2()), [0,255,0], 2)
                
        for c in ground_truth_data:
            cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
                        
        output_file = "out" + '_' + str(global_step) + 'its_' + str(FLAGS.step_size) + 'step_' + str(delta[i]) + 'threshold_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        """
        # csv output
        with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, str(elapsed_detect + elapsed_non_max),
                            str(global_step), str(len(ground_truth_data)), str(delta[i]),
                            str(len(detected)), str(FLAGS.step_size), str(tp), str(fp), str(fn)])
        

if __name__ == '__main__':
    tf.app.run()

