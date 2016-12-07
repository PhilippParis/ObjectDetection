import numpy as np
import cv2
import time
import csv
import os
import datetime
import tensorflow as tf

from utils import utils
from utils import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('test', 'eval_14', 'name of the test image')
flags.DEFINE_integer('tol', 25, 'max tolerated distance error between ground truth crater and detected crater')
flags.DEFINE_float('scaleFactor', 1.25, 'scale factor used in the multi scale detection')
flags.DEFINE_integer('minNeighbors', 50, 'minimum number of neighbours needed')

flags.DEFINE_string('cascade_xml','../output/checkpoints/haar2/cascade.xml', 'path to the cascade xml file')
flags.DEFINE_string('ground_truth_dir','../data/eval/data/', 'path to ground truth data dir')
flags.DEFINE_string('input_dir','../data/eval/', 'path to input images')
flags.DEFINE_string('output_dir','../output/results/haar2/', 'path to output dir')

# ============================================================= #
    
def main():
    # --------- load classifier ------- #
    cascade = cv2.CascadeClassifier(FLAGS.cascade_xml)
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    img = utils.getImage(FLAGS.input_dir + FLAGS.test + '.tif')
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    start = time.time()
    detected = cascade.detectMultiScale(img, scaleFactor=FLAGS.scaleFactor, minNeighbors=FLAGS.minNeighbors, maxSize=(150,150))
    elapsed = (time.time() - start)
    print 'detection time: %d' % elapsed

    # ------------- evaluation --------------#
    ground_truth_data = utils.csv_to_list(FLAGS.ground_truth_dir + FLAGS.test + '.csv', True)
    detected = [(x + w / 2, y + h / 2, w, h) for (x,y,w,h) in detected]
    tp, fn, fp, pr, re, f1 = utils.evaluate(ground_truth_data, detected, FLAGS.tol)
        
    # ----------------output ----------------#
    # image output
    
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (x,y,w,h) in detected:
        cv2.rectangle(img_out, (x-w/2,y-h/2),(x+w/2,y+h/2), [0,255,0], 3)
            
    for c in ground_truth_data:
        cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
        
    output_file = FLAGS.test + '_' + str(datetime.datetime.now())
    cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
    
    # csv output
    with open(FLAGS.output_dir + 'results_new.csv', 'ab') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([FLAGS.test, "-", str(elapsed),str(len(ground_truth_data)), str(FLAGS.scaleFactor), str(FLAGS.minNeighbors), 
                         str(len(detected)), str(tp), str(fp), str(fn), str(pr), str(re), str(f1)])

if __name__ == '__main__':
    main()
