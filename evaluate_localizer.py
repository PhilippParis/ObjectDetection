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
from utils.rect import Rect

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_size', 24, 'width and height of the cnn input')
flags.DEFINE_integer('min_window_size', 30, 'minimum window height and width')
flags.DEFINE_integer('max_window_size', 80, 'maximum window height and width')
flags.DEFINE_integer('label_size', 12, 'width and height of the input images')

flags.DEFINE_string('test', '../data/data/eval/eval_1.png', 'test image path')
flags.DEFINE_integer('scale_factor', 1.25, 'scale factor')
flags.DEFINE_integer('step_size', 12, 'sliding window step size')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/localizer', 'path to thensorflow checkpoint dir')
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
    input_size = (FLAGS.input_size, FLAGS.input_size)
    min_window_size = (FLAGS.min_window_size, FLAGS.min_window_size)
    max_window_size = (FLAGS.max_window_size, FLAGS.max_window_size)
    
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, input_size, FLAGS.scale_factor, min_window_size, max_window_size):
        feed = {x:windows, keep_prob:1.0}
        out = sess.run(model, feed_dict = feed)

        for i in range(0, len(out)):
            out_scaled = cv2.resize(np.reshape(out[i], [FLAGS.label_size,FLAGS.label_size]), 
                                    coords[i].size(), interpolation=cv2.INTER_CUBIC)
            
            mask[coords[i].y : coords[i].y2(), coords[i].x : coords[i].x2()] += out_scaled
    
    # image processing
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #  cv2.imwrite(FLAGS.output_dir + FLAGS.test + '_mask_' + str(FLAGS.step_size) + '_' + str(datetime.datetime.now()) + '.png', mask)
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
        objects.append(Rect(int(k.pt[0] - k.size), int(k.pt[1] - k.size), int(k.size * 2), int(k.size * 2)))
    
    return objects
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
    
    start = time.time()
    
    #sliding window detection
    mask = create_mask(model, x, keep_prob, img)
    elapsed = time.time() - start
    print 'detection time: %d' % (elapsed)

    # ------------- evaluation --------------#
    global_step = tf.train.global_step(sess, global_step)
    
    ground_truth_data = utils.get_ground_truth_data(csv_path)
    ground_truth_data = [(x + FLAGS.max_window_size,y + FLAGS.max_window_size) for (x,y) in ground_truth_data]
        
    for th in [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]:        
        detected = mask_to_objects(mask, th)
        tp, fn, fp = utils.evaluate(ground_truth_data, detected)
        
        # ----------------output ----------------#
        # image output
        """
        img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        
        for (r,score) in candidates:
            cv2.rectangle(img_out, (r.x,r.y), (r.x2(), r.y2()), [200,200,200], 2)
        
        for r in detected:
            cv2.rectangle(img_out, (r.x,r.y), (r.x2(), r.y2()), [0,255,0], 2)
            
        for c in ground_truth_data:
            cv2.circle(img_out, (c[0], c[1]), 3, [0,0,255],3)
        
        output_file = "out" + '_' + str(global_step) + 'its_' + str(FLAGS.step_size) + 'step_' + str(th) + 'threshold_' + str(datetime.datetime.now())
        cv2.imwrite(FLAGS.output_dir + output_file + '.png', img_out)
        """
        
        # csv output
        with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([FLAGS.test, str(elapsed), 
                            str(global_step), str(len(ground_truth_data)), str(th),
                            str(len(detected)), str(FLAGS.step_size), str(tp), str(fp), str(fn)])

if __name__ == '__main__':
    tf.app.run()
