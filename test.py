import utils
import input_data
import numpy
import cv2
import time
import tensorflow as tf
import csv
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_size', 64, 'width and height of the input images')


def import_data():
    """
    Returns a Data object which contains training data
    """
    # Import data
    data = input_data.Data(FLAGS.image_size, FLAGS.image_size)
    data.add_from_single_image("../data/eval/1510_positives.png", FLAGS.image_size, 
                               FLAGS.image_size, [0,1], 1510, 100)
    data.finalize()
    print data.info()
    
    for i in xrange(10):
        example = data.images[i].reshape(FLAGS.image_size, FLAGS.image_size)
        cv2.imshow('object mask', cv2.resize(example, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))   
        cv2.waitKey(0)


    return data

# ============================================================= #

if __name__ == '__main__':
    import_data()
