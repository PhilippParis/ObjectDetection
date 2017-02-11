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

flags.DEFINE_integer('input_size', 24, 'width and height of the input images')
flags.DEFINE_integer('max_window_size', 150, 'maximum window height and width')

flags.DEFINE_string('test', '../data/data/eval/eval_1.png', 'test image path')
flags.DEFINE_string('out', 'results', 'name of the test image')

flags.DEFINE_float('scaleFactor', 1.25, 'scale factor used in the multi scale detection')
flags.DEFINE_integer('minNeighbors', 10, 'minimum number of neighbours needed')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/classifier', 'path to tensorflow checkpoint dir')
flags.DEFINE_string('cascade_xml','../output/checkpoints/lbp/cascade.xml', 'path to the cascade xml file')
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
        windows.append(cv2.resize(img[y:y+h, xc:xc+w], (FLAGS.input_size, FLAGS.input_size), interpolation=cv2.INTER_CUBIC))
    windows = np.array(windows).reshape([len(candidates), FLAGS.input_size * FLAGS.input_size])

    prev = 0
    cur = min(500, len(windows))
    
    # run multiple batches of 500 windows at once
    while prev < len(windows):
        feed = {x:windows[prev:cur], keep_prob:1.0}
        out = sess.run(tf.nn.softmax(model), feed_dict = feed)
        
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
    x           = tf.placeholder("float", shape=[None, FLAGS.input_size * FLAGS.input_size])
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    model  = nn.create(x, keep_prob)
    
    # --------- restore model ----------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
        
    return model, x, keep_prob

# ============================================================= #

def main(_):
    image_path = FLAGS.test
    csv_path = os.path.splitext(image_path)[0] + ".csv"
    
    # --------- load classifier ------- #
    cascade = cv2.CascadeClassifier(FLAGS.cascade_xml)
    model, x, keep_prob = get_nn_classifier()
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    img = utils.getImage(image_path)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    delta = [-2, -1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    
    start = time.time()
    candidates = cascade.detectMultiScale(img, scaleFactor=FLAGS.scaleFactor, minNeighbors=FLAGS.minNeighbors, maxSize=(FLAGS.max_window_size,FLAGS.max_window_size))
    detected = nn_classification(candidates, img, model, x, keep_prob, delta)
    elapsed = (time.time() - start)  
    
    print 'detection time: %d' % elapsed

    # ------------- evaluation --------------#
        
    ground_truth_data = utils.get_ground_truth_data(csv_path)
    
    for j in xrange(0, len(delta)):
        detected[j] = [Rect(x, y, w, h) for (x,y,w,h) in detected[j]]
        tp, fn, fp = utils.evaluate(ground_truth_data, detected[j])
        
        # ----------------output ----------------#
        # image output
        """
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for (x,y,w,h) in detected[j]:
            cv2.rectangle(img_out, (x-w/2,y-h/2),(x+w/2,y+h/2), [0,255,0], 3)
                    
        for c in ground_truth_data:
            cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
                
        output_file = "out" + '_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        """
        # csv output
        with open(FLAGS.output_dir + FLAGS.out + '.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, str(elapsed), str(len(ground_truth_data)), delta[j], FLAGS.minNeighbors, FLAGS.scaleFactor, 
                            str(len(detected[j])), str(tp), str(fp), str(fn)])

if __name__ == '__main__':
    tf.app.run()
