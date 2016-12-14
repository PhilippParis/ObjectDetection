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
import random
import tensorflow as tf

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('image_size', 128, 'width and height of the example images')
gflags.DEFINE_integer('mask_size', 32, 'width and height of the mask images')
gflags.DEFINE_integer('count', 200, 'amount of randomly exported sub images')

gflags.DEFINE_string('output_img_dir','../../data/detect/eval/', 'input file')

def get_labels(path, label):
    csvfile = open(path, 'rb')
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    labels = []
    for row in csv_reader:
        if int(row[3]) == label:
            pos_x = int(row[0])
            pos_y = int(row[1])
            labels.append((pos_x, pos_y))
            
    return labels

# ============================================================= #    

def get_count(path, label):
    csvfile = open(path, 'rb')
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    count = 0
    for row in csv_reader:
        if int(row[3]) == label:
            count += 1
            
    return count

# ============================================================= #    

def create_input_img(img, x, y):
    return img[y : y + FLAGS.image_size, x : x + FLAGS.image_size]

# ============================================================= #  

def create_label_img(labels, src_x, src_y):
    lbl = numpy.zeros([FLAGS.mask_size, FLAGS.mask_size])
    factor = FLAGS.image_size / FLAGS.mask_size
    
    for x,y in labels:
        if x >= src_x and x <= (src_x + FLAGS.image_size) and y >= src_y and y <= (src_y + FLAGS.image_size):
            dest_x = (x - src_x) / factor
            dest_y = (y - src_y) / factor
            
            for x in range(dest_x - 2, dest_x + 2):
                for y in range(dest_y - 2, dest_y + 2):
                    if x >= 0 and x < 32 and y >= 0 and y < 32:
                        lbl[y, x] = 1
    return lbl          
            
# ============================================================= #    

def contains_objects(labels, src_x, src_y):
    for x,y in labels:
        if x >= src_x and x <= (src_x + FLAGS.image_size) and y >= src_y and y <= (src_y + FLAGS.image_size):
            return True
    return False
            
# ============================================================= #    

def main(argv):    
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        
    # import images
    positives_count = 0
    negatives_count = 0
    
    paths = []
    for i in xrange(1, 22):
        image_path = '../../data/eval/eval_' + str(i) + '.tif'
        label_path = '../../data/eval/data/eval_' + str(i) + '.csv'
        paths.append((image_path, label_path))
        positives_count += get_count(label_path, 1)
        negatives_count += get_count(label_path, 0)
        
    
    # create output images 
    height_pos = int(math.ceil(float(positives_count) / 100) * FLAGS.image_size)
    height_neg = int(math.ceil(float(negatives_count) / 100) * FLAGS.image_size)
    height_lbl = int(math.ceil(float(positives_count) / 100) * FLAGS.mask_size)
    
    img_pos = numpy.zeros((height_pos, 100 * FLAGS.image_size), numpy.float)
    #img_neg = numpy.zeros((height_neg, 100 * FLAGS.image_size), numpy.float)
    img_lbl = numpy.zeros((height_lbl, 100 * FLAGS.mask_size), numpy.float)

    y_pos = 0
    y_neg = 0
    y_lbl = 0
    
    x_pos = 0
    x_neg = 0
    x_lbl = 0
    
    # fill image
    for (image_path, label_path) in paths:
        # import data
        src_init = utils.getImage(image_path)
        src = cv2.copyMakeBorder(src_init, FLAGS.image_size, FLAGS.image_size, FLAGS.image_size, FLAGS.image_size, cv2.BORDER_REPLICATE)
        del src_init
        
        positives = get_labels(label_path, 1)
        negatives = get_labels(label_path, 0)
        height, width = src.shape
           
        for (x, y) in positives:
            #x = x + random.randrange(-20, 20)
            #y = y + random.randrange(-20, 20)
            
            # create positive example
            
            img = create_input_img(src, x + FLAGS.image_size / 2, y + FLAGS.image_size / 2)
            img_pos[y_pos : y_pos + FLAGS.image_size, x_pos : x_pos + FLAGS.image_size] = img
            x_pos += FLAGS.image_size
            if x_pos >= 100 * FLAGS.image_size:
                x_pos = 0
                y_pos += FLAGS.image_size
            del img
            
            # create label
            lbl = create_label_img(positives, x - FLAGS.image_size / 2, y - FLAGS.image_size / 2)
            img_lbl[y_lbl : y_lbl + FLAGS.mask_size, x_lbl : x_lbl + FLAGS.mask_size] = lbl
            x_lbl += FLAGS.mask_size
            if x_lbl >= 100 * FLAGS.mask_size:
                x_lbl = 0
                y_lbl += FLAGS.mask_size
            del lbl
            
        """        
        for (x, y) in negatives:
            img = create_input_img(src, x, y)
            img_neg[y_neg : y_neg + FLAGS.image_size, x_neg : x_neg + FLAGS.image_size] = img
            x_neg += FLAGS.image_size
            if x_neg >= 100 * FLAGS.image_size:
                x_neg = 0
                y_neg += FLAGS.image_size
            del img
        """ 
        del src
        
    img_pos = cv2.normalize(img_pos, None, 0, 255, cv2.NORM_MINMAX)
    #img_neg = cv2.normalize(img_neg, None, 0, 255, cv2.NORM_MINMAX)
    img_lbl = cv2.normalize(img_lbl, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(FLAGS.output_img_dir + str(positives_count) + "_positives.png", img_pos) 
    #cv2.imwrite(FLAGS.output_img_dir + str(negatives_count) + "_negatives.png", img_neg) 
    cv2.imwrite(FLAGS.output_img_dir + str(positives_count) + "_labels.png", img_lbl)  
    
if __name__ == '__main__':
    main(sys.argv)
