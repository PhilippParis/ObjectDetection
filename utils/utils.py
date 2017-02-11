import os, sys
import cv2
import numpy
import csv
from rect import Rect
from partition import partition

# ============================================================= #

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

def rotateImage(image, angle):
    """
    rotates the given image by the given angle
    """
    center = tuple(numpy.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    
# ============================================================= #

def slidingWindow(src_img, step_size, window_size, scale_factor, min_size, max_size):
    """
    a sliding window is moved through the src_img from upper left to lower right, each window 'step_size'
    pixels apart. the image is iterated multiple times, each time with a different window size, which
    is scaled by the 'scale_factor'
    
    Args:
        src_img: matrix, source image
        step_size: step size in x and y direction
        window_size: tuple, width and height of the output images
        scale_factor: scaling factor between each iteration of the image
        min_size: minimum window size
        max_size: maximum window size
    Returns:
        subimages of size 'window_size'. 
    """
    scale = 1
    count = 0
    images = []
    coords = []
    
    img_height = src_img.shape[0]
    img_width = src_img.shape[1]
    
    # increase scale to match minimum window size
    while window_size[0] * scale < min_size[0] or window_size[1] * scale < min_size[1]:
        scale *= scale_factor
    
    # sliding window
    while window_size[0] * scale <= max_size[0] and window_size[1] * scale <= max_size[1]:
        
        scaled_window = (int(window_size[0] * scale), int(window_size[1] * scale))
        
        for y in xrange(0, img_height, step_size):
            for x in xrange(0, img_width, step_size):
                
                if x + scaled_window[0] > img_width or y + scaled_window[1] > img_height:
                    continue
                
                img = src_img[y : y + scaled_window[1], x : x + scaled_window[0]]
                
                if scaled_window != window_size:
                    img = cv2.resize(img, window_size)
                
                images.append(img)
                coords.append(Rect(x, y, scaled_window[0], scaled_window[1]))
                
                count += 1
                
                if count > 2048:
                    yield numpy.array(images).reshape([count, window_size[0] * window_size[1]]), coords
                    images = []
                    coords = []
                    count = 0
                    
        
        if count > 0:
            yield numpy.array(images).reshape([count, window_size[0] * window_size[1]]), coords
            images = []
            coords = []
            count = 0
        
        scale *= scale_factor            

# ============================================================= #        
        
def evaluate(truth, detected):
    """
    evaluates the quality of the detection
    Args:
        truth: list of ground truth objects center points
        detected: list of detected objects bounding boxes
    Returns:
        true positive count, false_negative count, false positive count
    """
    
    # clone list of detections
    d = list(detected)
    fn = len(truth)
    tp = 0
    
    # evaluate detections
    for tx,ty in truth:
        bounding_box = next((x for x in d if x.contains(tx, ty)), None)
        
        if bounding_box != None:
            d.remove(bounding_box)
            tp += 1
            fn -= 1
            
    fp = len(d)
    
    return tp, fn, fp

# ============================================================= #
    
def non_maximum_suppression(bboxes, th):
    """
    applies non_maximum_suppression to the set of bounding boxes
    Args:
        bboxes: list of tuples of bounding boxes od the corresponding confidence score [(rect, score)]
    Returns:
        list of bounding boxes [Rect]
    """
    
    # sort boxes according to detection scores
    sort = sorted(bboxes, reverse=True, key=(lambda x: x[1]))

    for i in range(len(sort)): 
        j = i + 1
        while (j < len(sort)):
            if i >= len(sort) or j >= len(sort):
                break
            if Rect.overlap(sort[i][0], sort[j][0]) > th:
                del sort[j]
                j -= 1
            j += 1 
            
    return [r for (r,score) in sort]

# ============================================================= #

def ocv_grouping(bboxes, min_neighbours):
    """
    filtering/grouping of overlapping detections (bounding boxes) as done in opencv::detectMultiScale.
    can be used instead of non maximum suppression. test implementation, slow performance
    
    Args:
        bboxes: list of tuples of bounding boxes od the corresponding confidence score [(rect, score)]
        min_neighbours: minimum neighbours threshold
    Returns:
        list of bounding boxes [Rect]
    """
    
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

def get_ground_truth_data(csv_file_path):
    """
    converts the csv file at 'csv_file_path' into a list of integer triples
    Args:
        csv_file_path: path to csv file (2 columns with integer values)
    Returns:
        list (x,y)
    """
    candidates = []
    csv_file = open(csv_file_path, 'rb')
    for row in csv.reader(csv_file, delimiter=','):
        if int(row[2]) == 1:
            candidates.append((int(row[0]), int(row[1])))
    return candidates
