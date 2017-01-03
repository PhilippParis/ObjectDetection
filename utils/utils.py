import os, sys
import cv2
import numpy
import csv
from rect import Rect
from partition import partition

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

def slidingWindow(src_img, stepSize, windowSize, scale_factor, min_size, max_size):
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
    scale = 1
    count = 0
    images = []
    coords = []
    
    img_height = src_img.shape[0]
    img_width = src_img.shape[1]
    
    while windowSize[0] * scale < min_size[0] or windowSize[1] * scale < min_size[1]:
        scale *= scale_factor
    
    for i in xrange(10):        
        scaled_window = int(windowSize[0] * scale), int(windowSize[1] * scale)
        
        if scaled_window[0] > max_size[0] or scaled_window[1] > max_size[1]:
            return
    
        for y in xrange(0, img_height, stepSize):
            for x in xrange(0, img_width, stepSize):
                
                if x + scaled_window[0] > img_width or y + scaled_window[1] > img_height:
                    continue
                
                img = src_img[y : y + scaled_window[1], x : x + scaled_window[0]]
                
                if scaled_window != windowSize:
                    img = cv2.resize(img, windowSize)
                
                images.append(img)
                coords.append(Rect(x, y, scaled_window[0], scaled_window[1]))
                
                count += 1
                
                if count > 2048:
                    yield numpy.array(images).reshape([count, windowSize[0] * windowSize[1]]), coords
                    images = []
                    coords = []
                    count = 0
                    
        
        if count > 0:
            yield numpy.array(images).reshape([count, windowSize[0] * windowSize[1]]), coords
            images = []
            coords = []
            count = 0
        
        scale *= scale_factor            

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
    
    detected = list(detected)
    
    fn = 0
    tp = 0
    
    for tx,ty,_ in truth:
        index = contains(detected, tx, ty, tol)
        if index >= 0:
            del detected[index]
            tp += 1
        else:
            fn += 1
    
    fp = len(detected)
    
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
    
def non_maximum_suppression(bboxes, th):
    """
    applies non_maximum_suppression to the set of bounding boxes
    Args:
        bboxes: list of bounding boxes [(left upper x, left upper y, size, score)]
    Returns:
        list of objects [(center x, center y)]
    """
    
    def get_score(item):
        return item[1]
    
    # sort boxes according to detection scores
    sort = sorted(bboxes, reverse=True, key=get_score)

    for i in range(len(sort)): 
        j = i + 1
        while (j < len(sort)):
            if i >= len(sort) or j >= len(sort):
                break
            if Rect.overlap(sort[i][0], sort[j][0]) > th:
                del sort[j]
                j -= 1
            j += 1 
            

    return [r.center() for (r,score) in sort]

# ============================================================= #

def ocv_grouping(bboxes, min_neighbours):
    result = []
    avg_bboxes = []
    class_count = []

    # comparator function
    def compare(a, b):
        return Rect.compare(a[0], b[0])
    
    # cluster candidate bboxes into n classes, each class represents equvalent rectangles
    labels, num_classes = partition(bboxes, compare)
        
     # init vars
    for i in xrange(num_classes):
        avg_bboxes.append(Rect(0,0,0,0))
        class_count.append(0)
        
    # calc average bounding box of each class
    for i in xrange(len(bboxes)):
        j = labels[i]
        avg_bboxes[j] += bboxes[i][0]
        class_count[j] += 1
    
    # select valid bboxes
    for i in xrange(num_classes):    
        # reject classes with count < min_neighbours
        if class_count[i] < min_neighbours:
            continue
        
        # calc average
        avg_bboxes[i] /= class_count[i]
        
        # reject average bounding boxes which are inside other candidates
        reject = False
        for j in range(num_classes):
            if class_count[j] < min_neighbours:
                continue
            
            if i != j and avg_bboxes[j].contains(avg_bboxes[i]):
                reject = True
                break;
        
        # add to results if not rejected
        if not reject:
            result.append(avg_bboxes[i].center())
            
    return result  

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
