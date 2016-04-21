import utils
import input_data
import numpy
import cv2
import time
import tensorflow as tf
import network_simple as nn
import csv
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 28, 'width and height of the input images')
flags.DEFINE_string('test', 'Stadt', 'name of the test image')

flags.DEFINE_boolean('show_ground_truth', True, 'show ground truth data')
flags.DEFINE_boolean('candidate_detection', False, 'enable candidate detection')
flags.DEFINE_boolean('sliding_window_detection', True, 'enable sliding_window_detection')

flags.DEFINE_integer('window_size', 50, 'sliding window size')
flags.DEFINE_integer('step_size', 10, 'sliding window step size')

flags.DEFINE_string('checkpoint_path','checkpoints/simple_rotated_with_mars', 'path to checkpoint')

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
        list of found craters [(x,y,radius)]
    
    """
    global sess
    objects = []
    for windows, coords in utils.slidingWindow(src, FLAGS.step_size, (FLAGS.window_size, FLAGS.window_size), (FLAGS.image_size, FLAGS.image_size)):
        
        feed = {x:windows, keep_prob:1.0}
        y = sess.run(model, feed_dict = feed)

        for i in range(0, len(y)):
            if y[i][0] < 0.1 and y[i][1] > 0.9:
                objects.append((coords[i][0], coords[i][1], 5))
    return objects       


# ============================================================= #


def candidate_detection(model, x, keep_prob, src, candidates):
    """
    object detection via external candidate file
    Args:
        model: model which is used for detection
        x: input data placeholder
        keep_prob: keep probability placeholder
        src: source image
        candidates: list of candidates [(x,y,radius)]
    Returns
        list of found craters [(x,y,radius)]
    """
    # find max diameter
    x_border = 0
    y_border = 0
    for c in candidates:
        diameter = int(c[2] * 2)
        if diameter > x_border:
            x_border = diameter
        if diameter > y_border:
            y_border = diameter
        
    x_border = x_border / 2
    y_border = y_border / 2
    
    # add padding to image
    src = cv2.copyMakeBorder(src, x_border, y_border, x_border, y_border, cv2.BORDER_REPLICATE)
    
    images = []
    for c in candidates:
        x_pos = x_border + c[0]
        y_pos = y_border + c[1]
        diameter = int(c[2] * 2)
        sub_image = utils.getSubImage(src, x_pos, y_pos, (diameter, diameter))
        sub_image = utils.scaleImage(sub_image, (FLAGS.image_size, FLAGS.image_size))
        images.append(sub_image)
        
    images = numpy.array(images).reshape(len(candidates), FLAGS.image_size * FLAGS.image_size)
    feed = {x:images, keep_prob:1.0}
    y = sess.run(model, feed_dict = feed)
    
    objects = []
    for i in range(0, len(y)):
        if y[i][0] < 0.1 and y[i][1] > 0.9:
            objects.append(candidates[i])
    
    return objects
    
    
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
    model  = nn.create_network(x, keep_prob, FLAGS.image_size)
    # use for 'network' model
    #model  = nn.create_network(x)
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_path) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))  
    
    # ---------- object detection ------------#    
    print 'starting detection...'
    
    src = utils.getImage('../images/' + FLAGS.test + '.tif')
    start = time.time()
    
    #sliding window detection
    if FLAGS.sliding_window_detection:
        objects = sliding_window_detection(model, x, keep_prob, src)
    
    # candidate detection
    if FLAGS.candidate_detection:
        candidates = utils.csv_to_list('candidates/' + FLAGS.test + '.csv')
        objects = candidate_detection(model, x, keep_prob, src, candidates)
    
    print 'detection time: %d' % (time.time() - start)
    
    # ----------- output ---------------------#
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB) * 255
    
    # mark crater candidates
    if FLAGS.candidate_detection:
        for candidate in candidates:
            cv2.circle(src, (candidate[0], candidate[1]), candidate[2], (0,0,255), 0) # red
    
    # mark ground truth craters
    if FLAGS.show_ground_truth:
        ground_truth_data = utils.csv_to_list('../images/data/' + FLAGS.test + '.csv', True)
        for crater in ground_truth_data:
            cv2.circle(src, (crater[0], crater[1]), crater[2], (0,255,0), 0) # green
    
    # mark found objects
    for (x,y,r) in objects:
        cv2.circle(src, (x, y), r, (255,0,0), -1) #blue
    
    global_step = tf.train.global_step(sess, global_step)
    output_file = FLAGS.test + '_' + str(global_step) + 'its_' + ('cd' if FLAGS.candidate_detection else 'sw')
    cv2.imwrite('output/' + output_file + '.png', src)

if __name__ == '__main__':
    tf.app.run()
