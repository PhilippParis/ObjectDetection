import os, sys
import cv2
import numpy
import csv

def print_to_file(path, txt):
    print txt
    with open(path, 'a') as file:
        file.write(txt + '\n')

def getImage(path):
    """
    Args: 
        path: path to image
    Returns:
        image at path
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F) 


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
        
    return image[y - height : y + height, x - width : x + width]


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


def rotateImage(image, angle):
    """
    rotates the given image by the given angle
    """
    center = tuple(numpy.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    


def slidingWindow(image, stepSize, windowSize, outputSize):
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
    count = 0
    images = []
    coords = []
    
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            img = getSubImage(image, x + windowSize[0] / 2, y + windowSize[1] / 2, windowSize)
            if img.shape != windowSize:
                continue
            if windowSize != outputSize:
                img = scaleImage(img, outputSize)
            images.append(img)
            coords.append([x + windowSize[0] / 2, y + windowSize[1] / 2])
            count = count + 1
            
            if count > 500:
                yield numpy.array(images).reshape([count, outputSize[0] * outputSize[1]]), numpy.array(coords).reshape([count, 2])
                images = []
                coords = []
                count = 0
                
    yield numpy.array(images).reshape([count, outputSize[0] * outputSize[1]]), numpy.array(coords).reshape([count, 2])


def csv_to_list(csv_file_path, onlyTrue=False):
    """
    converts the csv file at 'csv_file_path' into a list of integer triples
    Args:
        csv_file_path: path to csv file (3 columns with integer values)
        onlyTrue: returns only rows where the fourth column == 1
    Returns:
        list
    """
    candidates = []
    csv_file = open(csv_file_path, 'rb')
    for row in csv.reader(csv_file, delimiter=','):
        if len(row) < 4 or (not onlyTrue or int(row[3]) == 1):
            candidates.append((int(row[0]), int(row[1]), int(row[2])))
    return candidates
