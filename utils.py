import os, sys
import cv2
import numpy

def getImage(path):
    """
    Args: 
        path: path to image
    Returns:
        image at path
    """
    return numpy.array(cv2.imread(path, 0))



def showImage(title, image):
    """
    displays the image in a window with given title
    
    Args:
        title: window title
        image: image to display
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)



def getSubImage(image, x, y, size):
    """
    returns a subimage from an image
    
    Args:
        image: image to retrieve the subimage from
        x,y: center coordinates for the subimage
        size: width and height of the subimage
    Returns:
        image
    """
    width = (size[0] + 1) / 2
    height = (size[1] + 1) / 2
    return numpy.array(image[y - height : y + height, x - width : x + width])



def scaleImage(image, size):
    """
    resizes an image
    
    Args:
        image: image to resize
        size: desired size
    Returns:
        resized image
    """
    return cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)



def slidingWindow(image, stepSize, windowSize):
    """
    returns subimages (windowsize) from the image starting in the left upper corner, moving to the
    bottom right corner by stepsize
    
    Args:
        image: source image
        stepSize: step size
        windowSize: width and height of the subimages
    Returns:
        subimages
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield getSubImage(image, x + windowSize[0] / 2, y + windowSize[1] / 2, windowSize)

