import utils
import cv2
import csv
import numpy
import tensorflow as tf

class Data:
    'Data class for handling input data'
    
    images = numpy.array([], dtype=numpy.uint8)     # input data: images
    labels = numpy.array([], dtype=numpy.uint8)     # desired output data: [0,1] = crater; [1,0] = no crater
    coords = numpy.array([], dtype=numpy.uint8)     # coordinates of the crater in the src image if available
    size = (0,0)                                    # size of the images
    num_examples = 0                                # number of examples stored in this class
    
    def __init__(self, width, height):
        """
        
        Sets the input image size
        Args:
            width: input image width
            height: input image height
        """
        self.size = (width, height)

    def finalize(self):
        """
        finalizes the datasets; must be called after adding all data
        """
        
        self.images = self.images.reshape(self.num_examples, self.size[0] * self.size[1])
        self.labels = self.labels.reshape(self.num_examples, 2)
        self.coords = self.coords.reshape(self.num_examples, 3)
        
    def add(self, data_path, image_path):
        """
        adds datasets from a single image 
        
        Args:
            data_path:  path to the csv file containing the crater coordinates (x,y) in the src image, 
                        the crater diameter and the label (1 = crater) in this order
            image_path: path to the src image
        """
        
        src = utils.getImage(image_path)
        
        csvfile = open(data_path, 'rb')
        reader = csv.reader(csvfile, delimiter=',')
        
        # get max diameter and add border to src image
        x_border = 0
        y_border = 0
        
        for row in reader:
            diameter = int(row[2])
            if diameter > x_border:
                x_border = diameter
            if diameter > y_border:
                y_border = diameter
        
        x_border = x_border / 2
        y_border = y_border / 2

        src = cv2.copyMakeBorder(src, x_border, y_border, x_border, y_border, cv2.BORDER_REPLICATE)
        
        csvfile = open(data_path, 'rb')
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            label = numpy.array([0,1]) if int(row[3]) == 1 else numpy.array([1,0])
            self.add_dataset(src, x_border + int(row[0]), y_border + int(row[1]), int(row[2]), label)
            
                
    def add_dataset(self, src_image, x, y, diameter, label):
        """
        adds a single dataset
        
        Args:
            src_image: source image
            x: x coordinate of the crater in the src image (in pixel)
            y: y coordinate of the crater in the src image (in pixel)
            diameter: diameter of the crater in pixels
            label: label of the crater [0,1] = crater; [1,0] = no crater
        """
        
        # get subimage
        sub_image = utils.getSubImage(src_image, x, y, (diameter, diameter))
        sub_image = utils.scaleImage(sub_image, self.size)
        
        for i in xrange(4):
            # add dataset
            self.images = numpy.append(self.images, sub_image)
            self.labels = numpy.append(self.labels, label)
            self.coords = numpy.append(self.coords, [x, y, diameter])
            self.num_examples = self.num_examples + 1
            
            # rotate image
            sub_image = utils.rotateImage(sub_image, 90)
        
                
    def next_batch(self, batch_size):
        """
        Args:
            batch_size: number of examples 
        Returns:
            a randomly selected batch of examples (input,labels)
        """
        
        # shuffle data
        perm = numpy.arange(self.num_examples) 
        numpy.random.shuffle(perm)
        self.images = self.images[perm]
        self.labels = self.labels[perm]
        
        return (self.images[0:batch_size], self.labels[0:batch_size])
    
    def info(self):
        """
        Returns:
            (num_examples, number of positive datasets, number of negative datasets)
        """
        positive = numpy.sum(numpy.equal([0,1], self.labels).astype(int))
        negative = numpy.sum(numpy.equal([1,0], self.labels).astype(int))
        
        return self.num_examples, positive / 2, negative / 2
                
    
