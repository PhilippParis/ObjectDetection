import numpy as np
import cv2
import time
import tensorflow as tf
import csv
import os
import datetime

from models import localizer as nn
from utils import utils
from utils import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 128, 'width and height of the input images')
flags.DEFINE_integer('window_size', 128, 'width and height of the sliding window')

flags.DEFINE_integer('label_size', 32, 'width and height of the input images')
flags.DEFINE_string('test', 'eval_8', 'name of the test image')

flags.DEFINE_boolean('show_ground_truth', True, 'show ground truth data')

flags.DEFINE_integer('step_size', 16, 'sliding window step size')
flags.DEFINE_integer('tol', 25, 'max tolerated distance error between ground truth crater and detected crater')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/localizer', 'path to thensorflow checkpoint dir')
flags.DEFINE_string('ground_truth_dir','../data/eval/data/', 'path to ground truth data dir')
flags.DEFINE_string('input_dir','../data/eval/', 'path to input images')
flags.DEFINE_string('output_dir','../output/results/localizer/', 'path to output dir')

# start session
sess = tf.InteractiveSession()

def create_mask(model, x, keep_prob, src):
    """
    object detection via sliding windows
    Args:
        model: tensorflow model which is used for detection
        x: input data placeholder
        keep_prob: keep probability placeholder (dropout)
        src: image to apply the detection
    Returns:
        image mask scaled between 0 and 255 
    """
    
    global sess
    height, width = src.shape
    mask = np.zeros((height,width), np.float32)
    w_size = (FLAGS.window_size, FLAGS.window_size)
    i_size = (FLAGS.image_size, FLAGS.image_size)
    
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, w_size, i_size):
        feed = {x:windows, keep_prob:1.0}
        out = sess.run(model, feed_dict = feed)

        for i in range(0, len(out)):
            xc = coords[i][0] - w_size[0] / 2
            yc = coords[i][1] - w_size[1] / 2
            
            out_scaled = cv2.resize(np.reshape(out[i], [32,32]), w_size, interpolation=cv2.INTER_CUBIC)
            # _,out_scaled = cv2.threshold(out_scaled,0.4,1.0,cv2.THRESH_TOZERO)
            mask[yc : yc + w_size[1], xc : xc + w_size[0]] += out_scaled
    
    # image processing
    
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #cv2.imwrite(FLAGS.output_dir + FLAGS.test + '_mask_' + str(FLAGS.step_size) + '_' + str(datetime.datetime.now()) + '.png', mask)
    return mask

# ============================================================= #

def mask_to_objects(mask, threshold):
    """
    applies a blob detection algorithm to the image
    Args:
        mask: image mask scaled between 0 and 255 
        threshold: min pixel intensity of interest
    Returns:
        list of objects [(x,y)]
    """
    
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = threshold
    params.maxThreshold = 255
    
    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 10000
    
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.blobColor = 255

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else: 
        detector = cv2.SimpleBlobDetector_create(params)
        
    keypoints = detector.detect(mask)
    
    objects = []
    for k in keypoints:
        objects.append((int(k.pt[0]), int(k.pt[1])))
    
    return objects
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
    
    start = time.time()
    
    #sliding window detection
    mask = create_mask(model, x, keep_prob, img)
    elapsed = time.time() - start
    print 'detection time: %d' % (elapsed)

    # ------------- evaluation --------------#
    global_step = tf.train.global_step(sess, global_step)
    
    ground_truth_data = utils.csv_to_list(FLAGS.ground_truth_dir + FLAGS.test + '.csv', True)
    ground_truth_data = [(x + FLAGS.image_size,y + FLAGS.image_size,rad) for (x,y,rad,lbl) in ground_truth_data]
    
    for th in xrange(10, 250, 10):
        detected = mask_to_objects(mask, th)
        tp, fn, fp, pr, re, f1 = utils.evaluate(ground_truth_data, detected, FLAGS.tol)
        
        # ----------------output ----------------#
        # image output
        '''
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        for c in detected:
            cv2.circle(img_out, (c[0], c[1]), 25, [0,255,0],3)
            
        for c in ground_truth_data:
            cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
        
        output_file = FLAGS.test + '_' + str(global_step) + 'its_' + str(FLAGS.step_size) + 'step_' + str(th) + 'threshold_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        '''
        # csv output
        with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, "-", str(elapsed), 
                            str(global_step), str(len(ground_truth_data)), str(th),
                            str(len(detected)), str(FLAGS.step_size), str(tp), str(fp), str(fn), 
                            str(pr), str(re), str(f1)])

if __name__ == '__main__':
    tf.app.run()
