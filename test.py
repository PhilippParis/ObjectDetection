import utils
import input_data
import numpy
import cv2
import time
import tensorflow as tf
import network_simple as nn
import csv




IMAGE_SIZE = 64

# test image load / show
img = utils.getImage('test.png')

utils.showImage('img', img)
utils.showImage('rotate', utils.rotateImage(img, 90))




#print image
#print utils.getSubImage(image, 100, 100, (6,6))

#test sub image
#utils.showImage('sub image', utils.getSubImage(image, 500, 500, (100, 200)))

#test resizing image
#utils.showImage('resize image', utils.scaleImage(image, (500, 500)))

#test sliding window
#for window in utils.slidingWindow(image, 100, (100,100)):
#    utils.showImage('window', window)


