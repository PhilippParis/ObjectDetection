from utils import utils
from utils import input_data
import numpy
import cv2
import time
import tensorflow as tf
import csv
import os
from models import classifier as nn

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('image_size', 128, 'width and height of the input images')
flags.DEFINE_integer('label_size', 32, 'width and height of the input images')

flags.DEFINE_string('checkpoint_dir','../output/checkpoints/classifier', 'path to checkpoint')

def eval_test():
    size = 32 * 32
    
    mask = numpy.zeros([size])
    label = numpy.zeros([size])
    

    mask[0:36] = 0.5
    label[0:36] = 1
    
    print "sum mask: " + str(numpy.sum(mask))
    print "sum label: " + str(numpy.sum(label))
    
    
    l2_error = 0
    for i in xrange(size):
        l2_error += (mask[i] - label[i]) ** 2
        
                
    print "l2 error: " + str(l2_error)
    

# ========================================== #

def create_cascade_pos_data():
    width = 6400
    height = 5312
    
    c = 0
    txt = ""
    for y in xrange(0, height, 64):
        for x in xrange(0, width, 64):
            txt += str(x) + " " + str(y) + " 64 64   " 
            c += 1

    print "positives/positives.png   " + str(c) + "  " + txt

# ========================================== #

def create_cascade_neg_data():
    width = 6400
    
    c = 0
    img = utils.getImage("../data/train/20038_negatives.png")
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    for y in xrange(0, 6720, 64):
        for x in xrange(0, 6400, 64):
            cv2.imwrite("negatives/" + str(c) + ".png", img[y:y+64, x:x+64])
            print "negatives/" + str(c) + ".png"
            
            c += 1
            if c >= 10000:
                return

def test_cascade():
    cascade = cv2.CascadeClassifier('../data/cascade/model/cascade.xml')
    img = utils.getImage("../data/eval/eval_1.tif")
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    craters = cascade.detectMultiScale(img, 1.3, 5)
    
    for (x,y,w,h) in craters:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    cv2.imshow('img',img)
    cv2.waitKey(0)

# ========================================== #

def localizer_test():
    sess = tf.Session()
    
    # ---------- create model ----------------#
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("float", shape=[None, FLAGS.label_size * FLAGS.label_size])
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # use for 'network_simple' model
    model  = nn.create(x, keep_prob)
    # use for 'network' model
    #model  = nn.create_network(x)
    
    data = input_data.Data((FLAGS.image_size, FLAGS.image_size), (FLAGS.label_size, FLAGS.label_size))
    
    data.add_examples("../data/test/test_input.png", 4, 100, None)
    data.add_labels("../data/test/test_label.png", 4, 100)
    
    data.finalize()
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
        
    batch_xs, batch_ys = data.next_batch(4)
    feed = {x:batch_xs, y_:batch_ys, keep_prob:1.0}
    output = sess.run(model, feed_dict = feed)
        

    img = numpy.zeros([64, 128])

    for i in xrange(4):
        label = numpy.reshape(batch_ys[i], [32,32])
        mask = numpy.reshape(output[i], [32,32])
        
        _,mask = cv2.threshold(mask,0.4,1.0,cv2.THRESH_TOZERO)
        
        label = cv2.normalize(label, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        img[0:32, i * 32 : (i+1) * 32] = label
        img[32:64, i * 32 : (i+1) * 32] = mask
    
    cv2.imwrite('../output/test_th.png', img)
    
    
# ========================================== #

def classifier_test():
    sess = tf.Session()
    
    
    # ---------- create model ----------------#
    # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("float", shape=[None, FLAGS.label_size * FLAGS.label_size])
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    model  = nn.create(x, keep_prob)
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
        
    
    img = utils.getImage('../data/detect/eval/1510_positives.png')
    
    windows = []
    
    for xc in xrange(0, 12800, 128):
        for y in range(0, 2048, 128):
            windows.append(img[y:y+128, xc:xc+128])
    
    windows = numpy.array(windows).reshape([1600, 128 * 128])

    z = 0
    prev = 0
    cur = min(500, len(windows))
    
    delta = 0.0
    
    
    while prev < len(windows):
        feed = {x:windows[prev:cur], keep_prob:1.0}
        out = sess.run(model, feed_dict = feed)
        
        for i in range(0, len(out)):
            if (out[i][1] - out[i][0]) > delta:
                xc = int((float(z) * 128.0) % 12800.0)
                xy = int((float(z) * 128.0) / 12800.0) * 128
                img[xy:xy+128, xc:xc+128] = numpy.zeros([128,128])
            z = z+1
        prev = cur
        cur = min(cur + 500, len(windows))
        
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('test.png', img)

def gauss_test():
    img = numpy.zeros([64,64])
    
    gaussian_kernel = numpy.dot(cv2.getGaussianKernel(64,10),numpy.transpose(cv2.getGaussianKernel(64,10))) * 0.9999
    img = gaussian_kernel
    
    print numpy.max(img)
    
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('test.png', img)
    
def pool_test():
    sess = tf.Session()
    
    img = numpy.zeros([4,4,3], dtype=numpy.float32)
    
    for x in xrange(0,4):
        for y in xrange(0,4):
            for c in xrange(0,3):
                img[y][x][c] = c + 1
    
    
    img[0,0,0] = 3
    img[0,3,0] = 0
    img[3,0,0] = 0
    img[3,3,0] = 0
    
    img = tf.reshape(img, [1, 4, 4, 3])
    out = tf.nn.max_pool(img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    tf.initialize_all_variables().run(session=sess)
    
    
    
    print 'input'
    print sess.run(img)
    
    
    print 'output'
    print sess.run(out)
    
def lrn_test():
    sess = tf.Session()
    
    s = 4
    
    img = numpy.zeros([s,s,5], dtype=numpy.float32)
    
    for x in xrange(0,s):
        for y in xrange(0,s):
            for c in xrange(0,5):
                img[y][x][c] = 2
    
    
    img[0,0,2] = 50
    #img[0,0,1] = 1
    #img[0,0,2] = 1
    
    #img[0,0,2] = 50
    
    img = tf.reshape(img, [1, s, s, 5])
    out = tf.nn.lrn(img, 1, bias=2.0, alpha=0.0001, beta=0.75, name='norm2')
    tf.initialize_all_variables().run(session=sess)
    
    
    print 'input'
    print sess.run(img)
    
    
    print 'output'
    print sess.run(out)
    
def import_data():
    """
    Returns training and evaluation data sets
    """
    train_set = input_data.Data((FLAGS.image_size, FLAGS.image_size), (1,1))
    eval_set = input_data.Data((FLAGS.image_size, FLAGS.image_size), (1,1))

    train_set.add_examples("../data/cascade/positives/positives2.png", 7500, 100, 1)
    train_set.add_examples("../data/detect/train/20038_negatives.png", 10000, 100, 0)
    
    eval_set.add_examples("../data/detect/eval/1510_positives.png", 1510, 100, 1)
    #eval_set.add_examples("../data/detect/eval/4054_negatives.png", 4054, 100, 0)
    eval_set.add_examples("../data/test/3709_negatives.png", 3710, 100, 0)
    
    train_set.finalize()
    eval_set.finalize()
    
    return train_set, eval_set

# ============================================================= #

    
    
def eval_cascade_training():
    
    # --------- load classifier ------- #
    cascade = cv2.CascadeClassifier('../output/checkpoints/haar3k5k/cascade.xml')
    
    train_set, eval_set = import_data()
    
    train_correct = 0
    for img,lbl,count in train_set.batches(1):
        img = numpy.reshape(img, newshape=(128, 128))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
        
        objects = cascade.detectMultiScale(img, 1.25, 5, minSize=(105,105))
        if (lbl == 0 and len(objects) == 0) or (lbl == 1 and len(objects) > 0):
            train_correct += 1
        
    eval_correct = 0
    for img,lbl,count in eval_set.batches(1):
        img = numpy.reshape(img, newshape=(128, 128))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
        
        objects = cascade.detectMultiScale(img, 1.25, 1, minSize=(105,105))
        if (lbl == 0 and len(objects) == 0) or (lbl == 1 and len(objects) > 0):
            eval_correct += 1
    
    
    train_error = 1.0 - float(train_correct)/float(train_set.count)
    eval_error = 1.0 - float(eval_correct)/float(eval_set.count)
    
    utils.print_to_file('../output/results/haar/train3k5k.csv', str(train_error) + "," + str(eval_error))

# ============================================================= #

def evaluation(sess, data_set, top_k_op, x, y_, keep_prob):
    true_count = 0
    num_examples = 0
    
    for batch_xs, batch_ys, count in data_set.batches(128):
        feed = {x:batch_xs, y_:batch_ys, keep_prob:1.0}
        predictions = sess.run(top_k_op, feed_dict = feed)
        true_count += numpy.sum(predictions)
        num_examples += count
            
    precision = float(true_count) / float(num_examples)
    return precision

# ============================================================= #


def fix_eval():
    sess = tf.Session()
    
    train_set, eval_set = import_data()
    
    # ---------- create model ----------------#
   # model input placeholder
    x           = tf.placeholder("float", shape=[None, FLAGS.image_size * FLAGS.image_size])
    # desired output placeholder
    y_          = tf.placeholder("int32")
    # keep probability placeholder
    keep_prob   = tf.placeholder("float")
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    model  = nn.create(x, keep_prob)
    top_k_op = tf.nn.in_top_k(model, y_, 1)
    
    # ---------- restore model ---------------#
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(FLAGS.checkpoint_dir) != None:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))  
        
    test_precision = evaluation(sess, eval_set, top_k_op, x, y_, keep_prob)
    print str(test_precision)
    

def count_blacks():
    
    img = utils.getImage("../data/detect/eval/4054_negatives.png")
    
    count = 0
    total = 0
    for xc in xrange(0, 12800, 128):
        for y in range(0, 5248, 128):
            total += 1
            if numpy.sum(img[y:y+128, xc:xc+128]) == 0 and total <= 4054:
                count += 1
                
    print str(count) + ", " + str(total)
    

if __name__ == '__main__':
    eval_cascade_training()
