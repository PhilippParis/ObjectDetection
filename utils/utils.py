import os, sys
import cv2
import numpy
import csv

def print_to_file(path, txt):
    print txt
    with open(path, 'a') as file:
        file.write(txt + '\n')

# ============================================================= #

def getImage(path):
    """
    Args: 
        path: path to image
    Returns:
        image at path
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F) 

# ============================================================= #

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

# ============================================================= #

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

# ============================================================= #

def rotateImage(image, angle):
    """
    rotates the given image by the given angle
    """
    center = tuple(numpy.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    

# ============================================================= #

def slidingWindow(image, stepSize, windowSize, imgSize):
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
            
            if x + windowSize[0] > image.shape[1] or y + windowSize[1] > image.shape[0]:
                continue
            
            img = image[y : y + windowSize[1], x : x + windowSize[0]]
            
            if windowSize != imgSize:
                img = cv2.resize(img, imgSize, interpolation=cv2.INTER_CUBIC)
            
            images.append(img)
            coords.append([x + windowSize[1] / 2, y + windowSize[0] / 2])
            count += 1
            
            if count > 500:
                yield numpy.array(images).reshape([count, imgSize[0] * imgSize[1]]), numpy.array(coords).reshape([count, 2])
                images = []
                coords = []
                count = 0
    
    if count > 0:
        yield numpy.array(images).reshape([count, imgSize[0] * imgSize[1]]), numpy.array(coords).reshape([count, 2])

# ============================================================= #
        
def contains(objects, x, y, tol):
    """
    return the index if the (x,y) exists in objects with tolerance tol
    otherwise returns -1
    """
    for i in xrange(len(objects)):
            if abs(x - objects[i][0]) < tol and abs(y - objects[i][1]) < tol:
                return i
    return -1
    
# ============================================================= #        
        
def evaluate(truth, detected, tol):
    """
    evaluates the quality of the detection
    Args:
        truth: list of ground truth objects center points
        detected: list of detected objects center points
    Returns:
        true positive count, false_negative count, false positive count, precision, recall, f1
    """
    t = list(truth)
    d = list(detected)
    
    fn = 0
    tp = 0
    
    for tx,ty,_,_ in t:
        index = contains(d, tx, ty, tol)
        if index >= 0:
            del d[index]
            tp += 1
        else:
            fn += 1
    
    fp = len(d)
    
    f1_score = 0.0
    precision = 0.0
    recall = 0.0
    
    if (tp + fp) != 0:
        precision = float(tp) / float(tp + fp)
    if (tp + fn) != 0:    
        recall = float(tp) / float(tp + fn)
    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return tp, fn, fp, precision, recall, f1_score

# ============================================================= #

def csv_to_list(csv_file_path, onlyTrue=False):
    """
    converts the csv file at 'csv_file_path' into a list of integer triples
    Args:
        csv_file_path: path to csv file (4 columns with integer values)
        onlyTrue: returns only rows where the fourth column == 1
    Returns:
        list
    """
    candidates = []
    csv_file = open(csv_file_path, 'rb')
    for row in csv.reader(csv_file, delimiter=','):
        if len(row) < 4 or (not onlyTrue or int(row[3]) == 1):
            candidates.append((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
    return candidates
