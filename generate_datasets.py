import cv2
import utils
import gflags
import numpy as np
import sys
import random
import csv

FLAGS =  gflags.FLAGS

gflags.DEFINE_string('input_file','../images/25_marked.tif', 'path to input file')
gflags.DEFINE_string('output_file','../images/data/25.csv', 'output file')
gflags.DEFINE_integer('rad', 25, 'radius of objects')

def findObjects(mask):
    """
    Returns a list of found objects 
    Args:
        mask = image where the objects center points are marked
        
    Returns:
        [x,y,radius,1]     
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 255
    
    params.filterByArea = True
    params.minArea = 4
    params.maxArea = 10000
    
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)
    
    blobs = []
    for k in keypoints:
        blobs.append((int(k.pt[0]), int(k.pt[1]), FLAGS.rad, 1))
    
    return blobs


# ============================================================= #


def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        
    img = cv2.imread(FLAGS.input_file)
    height, width, _ = img.shape
    
    # --------- positive objects -----------------#
    mask = cv2.inRange(img, np.array([0,0,255]), np.array([0,0,255]))
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX) 
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    objects = findObjects(mask)
    count = len(objects)
    
    # ----------- create negative objects ---------#
    for i in xrange(1000):
        x = int(50 + random.random() * (width - 50))
        y = int(50 + random.random() * (height - 50))
        
        valid = True
        for c in objects:
            if abs(x - c[0]) < 2*FLAGS.rad and abs(y - c[1]) < 2*FLAGS.rad:
                valid = False
                break
        if valid:
            objects.append((x,y, FLAGS.rad, 0))
    
    # ----------------- output ---------------------#
    print '# positive objects: ' + str(count)
    print '# negative objects: ' + str(len(objects) - count)
    
    with open(FLAGS.output_file, 'wb') as file:
        writer = csv.writer(file, delimiter=',')
        for c in objects:
            writer.writerow(c)
    
    # ----------------- show image ---------------------#
    '''
    for c in objects:
        cv2.circle(img, (c[0], c[1]), FLAGS.rad, ([0,255,0] if c[3] == 1 else [0,0,255]),3)
    
    cv2.imshow('object mask', cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC))   
    cv2.waitKey(0)
    '''
    
if __name__ == '__main__':
    main(sys.argv)
