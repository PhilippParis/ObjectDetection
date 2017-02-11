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
from utils import input_data
from utils import utils
import numpy
import math
import os
import time
import random
import tensorflow as tf

FLAGS = gflags.FLAGS

gflags.DEFINE_integer('object_size', 64, 'width and height of each imported example')
gflags.DEFINE_integer('image_size', 24, 'width and height of each exported example')
gflags.DEFINE_integer('mask_size', 12, 'width and height of the mask images')
gflags.DEFINE_integer('count', 200, 'amount of randomly exported sub images')

gflags.DEFINE_string('input_dir','../data/data/eval', 'output directory')
gflags.DEFINE_string('output_dir','../data/', 'output directory')

def get_labels(path, label):
    csvfile = open(path, 'rb')
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    labels = []
    for row in csv_reader:
        if int(row[2]) == label:
            pos_x = int(row[0])
            pos_y = int(row[1])
            labels.append((pos_x, pos_y))
    
    csvfile.close()
    return labels

# ============================================================= #    

def get_count(path, label):
    csvfile = open(path, 'rb')
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    count = 0
    for row in csv_reader:
        if int(row[2]) == label:
            count += 1
            
    csvfile.close()
    return count

# ============================================================= #    

def create_input_img(img, x, y):
    rad = FLAGS.object_size / 2
    example = img[y - rad : y + rad, x - rad : x + rad]
    return cv2.resize(example, (FLAGS.image_size, FLAGS.image_size))

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
        return
        
    # import images
    positives_count = 0
    negatives_count = 0
    paths = []
    
    files = [f for f in os.listdir(FLAGS.input_dir) if os.path.isfile(os.path.join(FLAGS.input_dir, f))]
    
    for f in files:
        # check if image file and label file with same name exist
        basename = os.path.splitext(f)[0]
        extension = os.path.splitext(f)[1]
        
        if extension != ".csv" and basename + ".csv" in files:
            image_path = os.path.join(FLAGS.input_dir, f)
            label_path = os.path.join(FLAGS.input_dir, basename + ".csv")
            
            paths.append((image_path, label_path))
            positives_count += get_count(label_path, 1)
            negatives_count += get_count(label_path, 0)
    

    print str(positives_count) + " positive examples imported"
    print str(negatives_count) + " negative examples imported"

    # create output images 
    height_pos = int(math.ceil(float(positives_count) / 100) * FLAGS.image_size)
    height_neg = int(math.ceil(float(negatives_count) / 100) * FLAGS.image_size)
    
    img_pos = numpy.zeros((height_pos, 100 * FLAGS.image_size), numpy.float)
    img_neg = numpy.zeros((height_neg, 100 * FLAGS.image_size), numpy.float)

    y_pos = 0
    y_neg = 0
    
    x_pos = 0
    x_neg = 0
    
    # fill image
    for (image_path, label_path) in paths:
        # import data
        src = utils.getImage(image_path)
        src = cv2.copyMakeBorder(src, FLAGS.object_size, FLAGS.object_size, FLAGS.object_size, FLAGS.object_size, cv2.BORDER_REPLICATE)
        
        positives = get_labels(label_path, 1)
        negatives = get_labels(label_path, 0)
        height, width = src.shape
           
        for (x, y) in positives:            
            # create positive example
            img = create_input_img(src, x + FLAGS.object_size, y + FLAGS.object_size)
            img_pos[y_pos : y_pos + FLAGS.image_size, x_pos : x_pos + FLAGS.image_size] = img
            x_pos += FLAGS.image_size
            if x_pos >= 100 * FLAGS.image_size:
                x_pos = 0
                y_pos += FLAGS.image_size
            del img
              
        for (x, y) in negatives:     
            # create negative example
            img = create_input_img(src, x + FLAGS.object_size, y + FLAGS.object_size)
            img_neg[y_neg : y_neg + FLAGS.image_size, x_neg : x_neg + FLAGS.image_size] = img
            x_neg += FLAGS.image_size
            if x_neg >= 100 * FLAGS.image_size:
                x_neg = 0
                y_neg += FLAGS.image_size
            del img
        del src
        
    img_pos = cv2.normalize(img_pos, None, 0, 255, cv2.NORM_MINMAX)
    img_neg = cv2.normalize(img_neg, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(FLAGS.output_dir + str(positives_count) + "_positives.png", img_pos) 
    cv2.imwrite(FLAGS.output_dir + str(negatives_count) + "_negatives.png", img_neg) 
    
if __name__ == '__main__':
    main(sys.argv)
