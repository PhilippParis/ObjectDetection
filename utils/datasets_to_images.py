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

gflags.DEFINE_integer('image_size', 128, 'width and height of the output example images')
gflags.DEFINE_string('output_dir','../', 'output directory')

gflags.DEFINE_string('output_pos', 'positives.png', 'output file')
gflags.DEFINE_string('output_neg', 'negatives.png', 'output file')

def images():
    """
    Returns training and evaluation data sets
    """
    
    for i in xrange(1, 22):
        image = utils.getImage('../../data/eval/eval_' + str(i) + '.tif')        
        data = utils.csv_to_list('../../data/eval/data/eval_' + str(i) + '.csv')
        
        image = cv2.copyMakeBorder(image, FLAGS.image_size,FLAGS.image_size,FLAGS.image_size,FLAGS.image_size, cv2.BORDER_REPLICATE)
        data = [(x + FLAGS.image_size, y + FLAGS.image_size, r, l) for (x,y,r,l) in data]
        
        yield image, data


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
    
    count_pos = 0
    count_neg = 0
    for img, data in images():
        for x,y,r,l in data:
            if l == 1: 
                count_pos += 1
            if l == 0:
                count_neg += 1
                
    imgs_per_row = 100
    width = imgs_per_row * FLAGS.image_size;
    
    height = int(math.ceil(float(count_pos) / imgs_per_row) * FLAGS.image_size)
    img_pos = numpy.zeros((height, width), numpy.float)
    
    height = int(math.ceil(float(count_neg) / imgs_per_row) * FLAGS.image_size)
    img_neg = numpy.zeros((height, width), numpy.float)
    
    s = FLAGS.image_size/2
    
    x_pos = s
    y_pos = s
    x_neg = s
    y_neg = s
    
    for img, data in images():
        for x,y,r,l in data:
            if l == 1:
                img_pos[y_pos-s:y_pos+s, x_pos-s:x_pos+s] = img[y-s:y+s, x-s:x+s] 
                x_pos += FLAGS.image_size
                
            if l == 0:
                img_neg[y_neg-s:y_neg+s, x_neg-s:x_neg+s] = img[y-s:y+s, x-s:x+s] 
                x_neg += FLAGS.image_size
                
            if x_pos >= imgs_per_row * FLAGS.image_size:
                x_pos = s
                y_pos += FLAGS.image_size
                
            if x_neg >= imgs_per_row * FLAGS.image_size:
                x_neg = s
                y_neg += FLAGS.image_size
     
     
    img_pos = cv2.normalize(img_pos, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(FLAGS.output_dir + str(count_pos) + "_" + FLAGS.output_pos, img_pos)
    
    img_neg = cv2.normalize(img_neg, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(FLAGS.output_dir + str(count_neg) + "_" + FLAGS.output_neg, img_neg)
    
if __name__ == '__main__':
    main(sys.argv)
