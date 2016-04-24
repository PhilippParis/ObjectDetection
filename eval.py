import cv2
import gflags
import sys
import numpy as np
import csv
import utils

FLAGS =  gflags.FLAGS
gflags.DEFINE_string('file', 'Land_40000its_0.01delta_sw.png', 'name of the image file')
gflags.DEFINE_string('test', 'Land', 'name of the test image')
gflags.DEFINE_integer('tol', 25, 'tolerance')

def findBlobs(mask):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 255
    
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 10000
    
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector(params)
    keypoints = detector.detect(mask)
    
    blobs = []
    for k in keypoints:
        blobs.append((int(k.pt[0]), int(k.pt[1])))
    
    return blobs


def getDetected():
    img = cv2.imread('output/' + FLAGS.file)
    
    mask = cv2.inRange(img, np.array([255,0,0]), np.array([255,0,0]))
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX) 
    mask = cv2.cvtColor(mask, cv2.cv.CV_GRAY2BGR)
    
    kernel = np.ones((5,5,), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return findBlobs(mask)


def evaluate(truth, detected):
    t = list(truth)
    d = list(detected)
    
    neg = 0
    pos = 0
    
    for tx,ty,_ in t:
        found = False
        for i in xrange(len(d)):
            if abs(tx-d[i][0]) < FLAGS.tol and abs(ty-d[i][1]) < FLAGS.tol:
                del d[i]
                pos = pos + 1
                found = True
                break
        if not found:
            neg = neg +1
    
    false_pos = len(d)
    return pos, neg, false_pos
    

def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        
    img = cv2.imread('../images/' + FLAGS.test + '.tif')
        
    detected = getDetected()
    truth = utils.csv_to_list('../images/data/' + FLAGS.test + '.csv', True)
    
    # --------- output -------------#
    tp, fn, fp = evaluate(truth, detected)
    
    tpr = float(tp) / (tp + fn)
    
    print 'truth:\t\t' + str(len(truth))
    print 'detected:\t' + str(len(detected))
    print 'true positive:\t' + str(tp)
    print 'false positive:\t' + str(fp)
    print 'false negative:\t' + str(fn)
    print 'true positive rate:\t' + str(tpr)
    
    #------- image output --------#
    for c in detected:
        cv2.circle(img, (c[0], c[1]), 25, [0,0,255],3)
        
    for c in truth:
        cv2.circle(img, (c[0], c[1]), 25, [0,255,0],3)
    
    cv2.imshow('detected craters', cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC))   
    cv2.waitKey(0)
    
        
if __name__ == '__main__':
    main(sys.argv)
