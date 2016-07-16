"""
imports training examples (positives and negatives) from images (given their positions in an csv, see generate_datasets.py)
and exports the examples as two images, each one containing either all positive or all negative examples.

usage:
1. edit import_data() to import all images with their corresponding csv
2. python2 datasets_to_images.py -image_size="width/height of exported examples in pixels" -output_dir="output directory"
"""

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

gflags.DEFINE_integer('image_size', 64, 'width and height of the output example images')
gflags.DEFINE_string('output_dir','../data/', 'output directory')

gflags.DEFINE_string('output_pos', 'positives.png', 'output file')
gflags.DEFINE_string('output_neg', 'negatives.png', 'output file')

def import_data():
    """
    Returns training and evaluation data sets
    """
    data = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    for i in xrange(1, 3):
        data.add('../data/train/data/train_' + str(i) + '.csv', '../data/train/train_' + str(i) + '.tif')
    data.finalize()   

    print '(datasets, positive, negative)'
    print data.info()
    
    return data

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

# ============================================================= #    
    
def main(argv):    
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        
    # import data
    data = import_data()
    _,count_pos,count_neg = data.info()
    
     # positive examples
    img_pos = create_img(data, count_pos, [0, 1], 100)
    img_pos = cv2.normalize(img_pos, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(FLAGS.output_dir + str(count_pos) + "_" + FLAGS.output_pos, img_pos)
    
    # negative examples
    img_neg = create_img(data, count_neg, [1, 0], 100)
    img_neg = cv2.normalize(img_neg, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(FLAGS.output_dir + str(count_neg) + "_" + FLAGS.output_neg, img_neg)
    
if __name__ == '__main__':
    main(sys.argv)
