import os, sys
import cv2


# returns the image at the given path
def getImage(path):
    return cv2.imread(path)

# displays the image in a window with given title
def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

# returns a subimage from 'image' with center (x,y) and (width, height) as in 'size'
def getSubImage(image, x, y, size):
    width = size[0] / 2
    height = size[1] / 2
    return image[y - height : y + height, x - width : x + width]

# returns the resized image
def scaleImage(image, size):
    return cv2.resize(image, size)

# returns subimages (windowsize) from the image starting in the left upper corner, moving to the
# bottom right corner by stepsize
def slidingWindow(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield getSubImage(image, x + windowSize[0] / 2, y + windowSize[1] / 2, windowSize)

