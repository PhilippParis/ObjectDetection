import utils
import input_data
import numpy as np
import cv2
import time
import tensorflow as tf
import model as nn
import csv
import os
import datetime

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 64, 'width and height of the input images')
flags.DEFINE_string('test', 'eval_1', 'name of the test image')

flags.DEFINE_boolean('show_ground_truth', True, 'show ground truth data')

flags.DEFINE_integer('window_size', 64, 'sliding window size')
flags.DEFINE_integer('step_size', 10, 'sliding window step size')
flags.DEFINE_integer('tol', 25, 'tolerance')
flags.DEFINE_float('delta', 0.1, 'detection tolerance delta')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/7layer', 'path to checkpoint dir')
flags.DEFINE_string('ground_truth_dir','../data/eval/data/', 'path to ground truth data dir')
flags.DEFINE_string('input_dir','../data/eval/', 'path to input images')
flags.DEFINE_string('output_dir','../output/results/7layer/', 'path to output dir')

# start session
sess = tf.InteractiveSession()

def sliding_window_detection(model, x, keep_prob, src):
    """
    object detection via sliding windows
    Args:
        model: model which is used for detection
        x: input data placeholder
        keep_prob: keep probability placeholder
        src: image to apply the detection
    Returns:
        list of found objects [(x,y,radius)]
    
    """
    global sess
    height, width = src.shape
    mask = np.zeros((height,width,1), np.uint8)
    w_size = (FLAGS.window_size, FLAGS.window_size)
    i_size = (FLAGS.image_size, FLAGS.image_size)
    
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, w_size, i_size):
        feed = {x:windows, keep_prob:1.0}
        y = sess.run(model, feed_dict = feed)

        for i in range(0, len(y)):
            if y[i][0] < -3.0 and y[i][1] > (3.0 - FLAGS.delta): 
                cv2.circle(mask, (coords[i][0], coords[i][1]), 5, (255,255,255), -1)
    
    # image processing
    
    #cv2.imshow('object mask', cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))   
    #cv2.waitKey(0)
    
    kernel = np.ones((5,5,), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 255
    
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 10000
    
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)
    
    objects = []
    for k in keypoints:
        objects.append((int(k.pt[0]), int(k.pt[1])))
    
    return objects


# ============================================================= #
    
def evaluate(truth, detected):
    """
    evaluates the quality of the detection
    Args:
        truth: list of ground truth objects center points
        detected: list of detected objects center points
    Returns:
        true positive count, false_negative count, false positive count
    """
    t = list(truth)
    d = list(detected)
    
    fn = 0
    tp = 0
    
    for tx,ty,_ in t:
        found = False
        for i in xrange(len(d)):
            if abs(tx-d[i][0]) < FLAGS.tol and abs(ty-d[i][1]) < FLAGS.tol:
                del d[i]
                tp += 1
                found = True
                break
        if not found:
            fn += 1
    
    fp = len(d)
    return tp,fn,fp

# ============================================================= #
    
def main(_):
    # ---------- create model ----------------#
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("float", shape=[None, 2])
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # use for 'network_simple' model
    model  = nn.model(x, keep_prob)
    # use for 'network' model
    #model  = nn.create_network(x)
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
    
    # ---------- object detection ------------#    
    print 'starting detection of ' + FLAGS.test + '...'
    
    img = utils.getImage(FLAGS.input_dir + FLAGS.test + '.tif')
    start = time.time()
    
    #sliding window detection
    detected = sliding_window_detection(model, x, keep_prob, img)
    
    print 'detection time: %d' % (time.time() - start)

    # ------------- evaluation --------------#
    ground_truth_data = utils.csv_to_list(FLAGS.ground_truth_dir + FLAGS.test + '.csv', True)
    global_step = tf.train.global_step(sess, global_step)
    tp, fn, fp = evaluate(ground_truth_data, detected)
    f1_score = 0.0
    precision = 0.0
    recall = 0.0
    
    if (tp + fp) != 0:
        precision = float(tp) / float(tp + fp)
        
    if (tp + fn) != 0:    
        recall = float(tp) / float(tp + fn)

    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    # ----------------output ----------------#
    # image output
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
    for c in detected:
        cv2.circle(img, (c[0], c[1]), 25, [0,0,255],3)
        
    for c in ground_truth_data:
        cv2.circle(img, (c[0], c[1]), 20, [0,255,0],3)
    
    output_file = FLAGS.test + '_' + str(global_step) + 'its_' + str(FLAGS.delta) + 'delta_' + str(datetime.datetime.now())
    cv2.imwrite(FLAGS.output_dir + output_file + '.png', img)
    
    # csv output
    with open(FLAGS.output_dir + 'results.csv', 'ab') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([FLAGS.test, output_file, 
                         str(global_step), str(len(ground_truth_data)), 
                         str(len(detected)), str(FLAGS.delta), str(tp), str(fp), str(fn), 
                         str(precision), str(recall), str(f1_score)])

if __name__ == '__main__':
    tf.app.run()
