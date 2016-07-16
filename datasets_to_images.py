import utils
import cv2
import gflags
import csv
import sys
import input_data
import numpy
import math
import time
import tensorflow as tf

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('image_size', 64, 'width and height of the input images')
gflags.DEFINE_string('eval_out_dir','../data/evaluation/data/', 'output directory evaluation data')
gflags.DEFINE_string('train_out_dir','../data/training/data/', 'output directory training data')

gflags.DEFINE_string('output_pos', 'positives.png', 'output file')
gflags.DEFINE_string('output_neg', 'negatives.png', 'output file')

def import_data():
    """
    Returns training and evaluation data sets
    """
    train_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    for i in xrange(1, 34):
        train_set.add('../data/training/data/train_' + str(i) + '.csv', '../data/training/train_' + str(i) + '.tif')
    train_set.finalize()   
    
    eval_set = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    for i in xrange(1, 22):
        eval_set.add('../data/evaluation/data/eval_' + str(i) + '.csv', '../data/evaluation/eval_' + str(i) + '.tif')
    eval_set.finalize()     
    
    print '(datasets, positive, negative)'
    print train_set.info()
    print eval_set.info()
    
    return train_set, eval_set

# ============================================================= #    

def create_img(data, count, label, imgs_per_row):
    # create output image
    width = imgs_per_row * FLAGS.image_size;
    height = int(math.ceil(float(count) / imgs_per_row) * FLAGS.image_size)
    img = numpy.zeros((height, width), numpy.float)
    
    # current position vars
    x = 0
    y = 0
    
    # copy training sub images to output image
    for i in xrange(data.count):
        if (data.labels[i] == label).all():
            train_img = data.images[i].reshape(FLAGS.image_size, FLAGS.image_size)
            img[y : y + FLAGS.image_size, x : x + FLAGS.image_size] = train_img
            x += FLAGS.image_size
            if x >= imgs_per_row * FLAGS.image_size:
                x = 0
                y += FLAGS.image_size
    return img


def create_images(data, imgs_per_row, output_dir):
    _,count_pos,count_neg = data.info()
    
    # positive examples
    img_pos = create_img(data, count_pos, [0, 1], 100)
    img_pos = cv2.normalize(img_pos, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_dir + str(count_pos) + "_" + FLAGS.output_pos, img_pos)
    
    # negative examples
    img_neg = create_img(data, count_neg, [1, 0], 100)
    img_neg = cv2.normalize(img_neg, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_dir + str(count_neg) + "_" + FLAGS.output_neg, img_neg)

def main(argv):    
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        
    # import data
    train_set, eval_set = import_data()
    
    create_images(train_set, 100, FLAGS.train_out_dir)
    create_images(eval_set, 100, FLAGS.eval_out_dir)
    
if __name__ == '__main__':
    main(sys.argv)
