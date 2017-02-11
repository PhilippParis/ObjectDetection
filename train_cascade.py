"""
trains the cascade classifiers
needs to be executed in the same directory as positives.info and negatives.info
"""

import gflags
import cv2
import sys
import os
from subprocess import call

FLAGS =  gflags.FLAGS

gflags.DEFINE_string('positives_info', 'positives.info', 'path to positives.info file')
gflags.DEFINE_string('negatives_info', 'negatives.info', 'path to negatives.info file')
gflags.DEFINE_string('model_dir', 'model_lbp', 'cascade output path')

gflags.DEFINE_integer('image_size', 24, 'width and height of the example images in the spritesheets')

gflags.DEFINE_integer('count_positives', 7500, 'number of positive examples used for training')
gflags.DEFINE_integer('count_negatives', 10000, 'number of negative examples used for training')
gflags.DEFINE_integer('count_stages', 20, 'number of training stages')

gflags.DEFINE_string('feature_type', 'LBP', 'feature type used for training')


# ========================================== #

def convert_positives():
    call(["opencv_createsamples", "-info", FLAGS.positives_info, "-w" , str(FLAGS.image_size), 
          "-h", str(FLAGS.image_size), "-num", "8300", "-vec", "positives.vec"])

# ========================================== #

def train():
    call(["opencv_traincascade", "-data", FLAGS.model_dir, "-vec", "positives.vec", 
          "-bg", FLAGS.negatives_info, "-precalcValBufSize", "1048", "-precalcIdxBufSize", "1048",  
          "-numPos", str(FLAGS.count_positives), "-numNeg", str(FLAGS.count_negatives),
          "-numStages", str(FLAGS.count_stages), "-w", str(FLAGS.image_size), 
          "-h", str(FLAGS.image_size), "-minHitRate", "0.999", "-maxFalseAlarmRate", "0.5", 
          "-featureType", FLAGS.feature_type])

# ========================================== #

def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        return
        
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
        
    convert_positives()
    train()

# ========================================== #

if __name__ == '__main__':
    main(sys.argv)
