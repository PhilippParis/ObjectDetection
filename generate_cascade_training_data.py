"""
creates training examples in the format needed for the opencv cascade classifier

"""

import gflags
import cv2
import sys
import os
from shutil import copyfile

FLAGS =  gflags.FLAGS

gflags.DEFINE_string('positives_spritesheet','../data/train/8300_positives.png', 'path to spritesheet containing positive examples')
gflags.DEFINE_string('negatives_spritesheet','../data/train/20038_negatives.png', 'path to spritesheet containing negative examples')
gflags.DEFINE_string('output_dir','../data/cascade', 'output directory')
gflags.DEFINE_integer('image_size', 24, 'width and height of the example images in the spritesheets')

# ========================================== #

def create_cascade_pos_data():
    img = cv2.imread(FLAGS.positives_spritesheet)
    height, width, _ = img.shape
    del img
    
    c = 0
    txt = ""
    for y in xrange(0, height, FLAGS.image_size):
        for x in xrange(0, width, FLAGS.image_size):
            txt += str(x) + " " + str(y) + " " + str(FLAGS.image_size) + " " + str(FLAGS.image_size) + " "
            c += 1

    with open(FLAGS.output_dir + "/positives.info", 'w') as file:
        file.write("positives.png" + "  " + str(c) + "  " + txt)
    
    copyfile(FLAGS.positives_spritesheet, FLAGS.output_dir + "/positives.png")
    
    return c
        
# ========================================== #

def create_cascade_neg_data():
    img = cv2.imread(FLAGS.negatives_spritesheet)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    height, width, _ = img.shape
    
    c = 0
    txt = ""
    for y in xrange(0, height, FLAGS.image_size):
        for x in xrange(0, width, FLAGS.image_size):
            cv2.imwrite(FLAGS.output_dir + "/negatives/" + str(c) + ".png", img[y:y+FLAGS.image_size, x:x+FLAGS.image_size])
            txt += "negatives/" + str(c) + ".png" + "\n"
            c += 1
            
    with open(FLAGS.output_dir + "/negatives.info", 'w') as file:
        file.write(txt)        
    
    return c

# ========================================== #
            
def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        return
        
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
        
    if not os.path.exists(FLAGS.output_dir + "/negatives"):
        os.makedirs(FLAGS.output_dir + "/negatives")
    
        
    pos_count = create_cascade_pos_data()
    neg_count = create_cascade_neg_data()
    
    print str(pos_count) + " positive examples exported"
    print str(neg_count) + " negative examples exported"


if __name__ == '__main__':
    main(sys.argv)
