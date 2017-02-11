import numpy as np
import cv2
import time
import csv
import os
import datetime
import tensorflow as tf

from utils import utils
from utils import input_data
from utils.rect import Rect

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_window_size', 150, 'maximum window height and width')
flags.DEFINE_string('test', '../data/data/eval/eval_1.png', 'test image path')
flags.DEFINE_float('scaleFactor', 1.25, 'scale factor used in the multi scale detection')
flags.DEFINE_integer('minNeighbors', 30, 'minimum number of neighbours thresold')

flags.DEFINE_string('cascade_xml','../output/checkpoints/lbp/cascade.xml', 'path to the cascade xml file')
flags.DEFINE_string('output_dir','../output/results/lbp/', 'path to output dir')

# ============================================================= #

def main():
    image_path = FLAGS.test
    csv_path = os.path.splitext(image_path)[0] + ".csv"
    
    # ------------ load classifier ---------- #
    cascade = cv2.CascadeClassifier(FLAGS.cascade_xml)
    
    # -------------- open image --------------#
    img = utils.getImage(image_path)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    start = time.time()
    detected = cascade.detectMultiScale(img, scaleFactor=FLAGS.scaleFactor, minNeighbors=FLAGS.minNeighbors, maxSize=(FLAGS.max_window_size, FLAGS.max_window_size))
    elapsed = (time.time() - start)
    print 'detection time: %d' % elapsed
    
    # ------------- evaluation --------------#
    detected = [Rect(x, y, w, h) for (x,y,w,h) in detected]
    ground_truth_data = utils.get_ground_truth_data(csv_path)
    
    tp, fn, fp = utils.evaluate(ground_truth_data, detected)
        
    # ----------------output ----------------#
    # image output
    """
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    for c in ground_truth_data:
        cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
        
    for r in detected:
        cv2.rectangle(img_out, (r.x, r.y), (r.x2(), r.y2()), [0,255,0], 2)
        
    output_file = "out" + '_' + str(datetime.datetime.now())
    cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
    """
    # csv output
    with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([FLAGS.test, str(elapsed),str(len(ground_truth_data)), str(FLAGS.scaleFactor), 
                         str(FLAGS.minNeighbors), str(len(detected)), str(tp), str(fp), str(fn)])

if __name__ == '__main__':
    main()
