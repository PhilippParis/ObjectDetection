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

gflags.DEFINE_integer('image_size', 256, 'width and height of the output example images')
gflags.DEFINE_integer('count', 200, 'amount of randomly exported sub images')

gflags.DEFINE_string('output_img_dir','../data/detect/eval/input/', 'input file')
gflags.DEFINE_string('output_label_dir','../data/detect/eval/label/', 'input file')

gflags.DEFINE_string('output_dir','../data/', 'output directory')

def import_labels(path):
    csvfile = open(path, 'rb')
    csv_reader = csv.reader(csvfile, delimiter=',')
    labels = numpy.array([], dtype=numpy.uint8)    
    
    c = 0
    for row in csv_reader:
        if int(row[3]) == 1:
            pos_x = int(row[0])
            pos_y = int(row[1])
            labels = numpy.append(labels, [pos_x, pos_y])
            c = c +1
            
    labels = labels.reshape(c, 2)
    return labels

# ============================================================= #    

def create_input_img(img, x, y):
    return img[y : y + FLAGS.image_size, x : x + FLAGS.image_size]

# ============================================================= #  

def create_label_img(labels, src_x, src_y):
    lbl = numpy.zeros([32, 32])
    for x,y in labels:
        if x >= src_x and x <= (src_x + FLAGS.image_size) and y >= src_y and y <= (src_y + FLAGS.image_size):
            dest_x = (x - src_x) / 8
            dest_y = (y - src_y) / 8
            
            for x in range(dest_x - 1, dest_x + 1):
                for y in range(dest_y - 1, dest_y + 1):
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
        
    count = 0
    for i in xrange(1, 22):
        # import data
        src = utils.getImage('../data/eval/eval_' + str(i) + '.tif')
        labels = import_labels('../data/eval/data/eval_' + str(i) + '.csv')
        height, width = src.shape
            
        for j in xrange(2000):
            x = random.randrange(0, width - FLAGS.image_size)
            y = random.randrange(0, height - FLAGS.image_size)
            
            #if not contains_objects(labels, x, y):
            #    continue
            
            img = create_input_img(src, x, y)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(FLAGS.output_img_dir + str(count) + ".png", img)
            
            lbl = create_label_img(labels, x, y)
            lbl = cv2.normalize(lbl, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(FLAGS.output_label_dir + str(count) + ".png", lbl)
            
            count = count + 1
            if count >= (i * FLAGS.count):
                break
        
        
    
if __name__ == '__main__':
    main(sys.argv)
