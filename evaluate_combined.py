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
flags.DEFINE_string('test', 'eval_14', 'name of the test image')

flags.DEFINE_integer('tol', 25, 'max tolerated distance error between ground truth crater and detected crater')
flags.DEFINE_float('scaleFactor', 1.25, 'scale factor used in the multi scale detection')
flags.DEFINE_integer('minNeighbors', 20, 'minimum number of neighbours needed')
flags.DEFINE_float('delta', 0.8, 'threshold for the NN detection')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/classifier', 'path to tensorflow checkpoint dir')
flags.DEFINE_string('cascade_xml','../output/checkpoints/lbp/cascade.xml', 'path to the cascade xml file')
flags.DEFINE_string('ground_truth_dir','../data/eval/data/', 'path to ground truth data dir')
flags.DEFINE_string('input_dir','../data/eval/', 'path to input images')
flags.DEFINE_string('output_dir','../output/results/combined/', 'path to output dir')

# start session
sess = tf.InteractiveSession()

# ============================================================= #

def nn_classification(candidates, img, model, x, keep_prob, delta):
    """
    uses a neural network to classify the candidates
    Args:
        candidates: list of candidates in the provided image
        img: image
        model: tensorflow model
        x: input data placeholder
        keep_prob: keep probability placeholder (dropout)
        delta: list of detection thresholds
    Returns:
        returns a 2d-list of detections where the objects at position i represent all detected objects
        classified with threshold delta[i]
    """
    
    global sess
    
    windows = []
    detected = []
    
    for j in range(0, len(delta)):
        detected.append([])
    
    for (xc,y,w,h) in candidates:
        windows.append(cv2.resize(img[y:y+h, xc:xc+w], (FLAGS.image_size, FLAGS.image_size), interpolation=cv2.INTER_CUBIC))
    windows = np.array(windows).reshape([len(candidates), FLAGS.image_size * FLAGS.image_size])

    prev = 0
    cur = min(500, len(windows))
    
    while prev < len(windows):
        feed = {x:windows[prev:cur], keep_prob:1.0}
        out = sess.run(model, feed_dict = feed)
        
        for i in range(0, len(out)):
            for j in range(0, len(delta)):
                if (out[i][1] - out[i][0]) > delta[j]:
                    detected[j].append(candidates[i])
        prev = cur
        cur = min(cur + 500, len(windows))
    return detected

# ============================================================= #
    
def get_nn_classifier():
    """
    loads the tensorflow model and returns it
    Returns:
        tensorflow model, input data placeholder, keep probability placeholder (dropout)
    """
    # --------- load nn ----------------#
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # use for 'network_simple' model
    model  = nn.create(x, keep_prob)
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
        
    return model, x, keep_prob

# ============================================================= #

def main(_):
    # --------- load classifier ------- #
    cascade = cv2.CascadeClassifier(FLAGS.cascade_xml)
    model, x, keep_prob = get_nn_classifier()
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    img = utils.getImage(FLAGS.input_dir + FLAGS.test + '.tif')
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    delta = [-2.0, -1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999, 0.999999, 0.9999999]
    
    start = time.time()
    candidates = cascade.detectMultiScale(img, scaleFactor=FLAGS.scaleFactor, minNeighbors=FLAGS.minNeighbors, maxSize=(150,150))
    detected = nn_classification(candidates, img, model, x, keep_prob, delta)
        
    elapsed = (time.time() - start)  
    print 'detection time: %d' % elapsed

    # ------------- evaluation --------------#
        
    ground_truth_data = utils.csv_to_list(FLAGS.ground_truth_dir + FLAGS.test + '.csv', True)
    
    for j in xrange(0, len(delta)):
        detected[j] = [(x + w / 2, y + h / 2, w, h) for (x,y,w,h) in detected[j]]
        tp, fn, fp, pr, re, f1 = utils.evaluate(ground_truth_data, detected[j], FLAGS.tol)
        
        # ----------------output ----------------#
        # image output
        '''
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for (x,y,w,h) in detected[j]:
            cv2.rectangle(img_out, (x-w/2,y-h/2),(x+w/2,y+h/2), [0,255,0], 3)
                    
        for (x,y,r) in ground_truth_data:
            cv2.circle(img_out, (x, y), 3, [0,0,255],3)
                
        output_file = FLAGS.test + '_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        '''
        # csv output
        with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, "-", str(elapsed), str(len(ground_truth_data)), delta[j], FLAGS.minNeighbors, FLAGS.scaleFactor, 
                            str(len(detected[j])), str(tp), str(fp), str(fn), 
                            str(pr), str(re), str(f1)])

if __name__ == '__main__':
    tf.app.run()
