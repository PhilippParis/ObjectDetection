import utils
import cv2
import csv
import numpy
import tensorflow as tf

class Data:
    'Data class for handling input data'
    images = numpy.array([], dtype=numpy.uint8)     # images
    labels = numpy.array([], dtype=numpy.uint8)     # desired output data: [0,1] = crater; [1,0] = no crater
    img_size = (0,0)                                # size of the images
    count = 0                                       # number of images stored in this class
    
    def __init__(self, width, height):
        """
        
        Sets the input image size
        Args:
            width: input image width
            height: input image height
        """
        self.img_size = (width, height)
        
    # ============================================================= #    

    def finalize(self):
        """
        finalizes the datasets; must be called after adding all data
        """
        self.images = self.images.reshape(self.count, self.img_size[0] * self.img_size[1])
        self.labels = self.labels.reshape(self.count, 2)
        
    # ============================================================= #    
        
    def get_image_with_border(self, data_path, image_path):
        """
        returns the image with a border around the image
        
        Args:
            data_path:  path to the csv file containing the crater coordinates (x,y) in the src image, 
                        the crater diameter and the label (1 = crater) in this order
            image_path: path to the src image
        Returns: 
            image with border, border size
        """
        src_img = utils.getImage(image_path)
        csvfile = open(data_path, 'rb')
        reader = csv.reader(csvfile, delimiter=',')
        
        # get max diameter
        max_diameter = 0
        for row in reader:
            if int(row[2]) > max_diameter:
                max_diameter = int(row[2])
        
        # add border
        border = max_diameter / 2
        return cv2.copyMakeBorder(src_img, border, border, border, border, cv2.BORDER_REPLICATE), border
        
    # ============================================================= #    
        
    def add(self, data_path, image_path):
        """
        adds images to the training data set
        
        Args:
            data_path:  path to the csv file containing the crater coordinates (x,y) in the src image, 
                        the crater diameter and the label (1 = crater) in this order
            image_path: path to the src image
        """
        src_image, border = self.get_image_with_border(data_path, image_path)
        csv_file = open(data_path, 'rb')
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            # read row
            pos_x = border + int(row[0])
            pos_y = border + int(row[1])
            diameter = int(row[2])
            
            # get label
            label = numpy.array([0,1]) if int(row[3]) == 1 else numpy.array([1,0])
            
            # get subimage
            sub_image = utils.getSubImage(src_image, pos_x, pos_y, (diameter, diameter))
            sub_image = utils.scaleImage(sub_image, self.img_size)
            
            # add dataset
            self.images = numpy.append(self.images, sub_image)
            self.labels = numpy.append(self.labels, label)
            self.count = self.count + 1
        
        # free memory
        del src_image
        csv_file.close()
        
    # ============================================================= #    
                
    def next_batch(self, batch_size):
        """
        Args:
            batch_size: number of examples 
        Returns:
            a randomly selected batch of examples (input,labels)
        """
        
        # shuffle data
        perm = numpy.arange(self.count) 
        numpy.random.shuffle(perm)
        self.images = self.images[perm]
        self.labels = self.labels[perm]
        
        return (self.images[0:batch_size], self.labels[0:batch_size])
    
    # ============================================================= #    
    
    def info(self):
        """
        Returns:
            (count, number of positive datasets, number of negative datasets)
        """
        positive = numpy.sum(numpy.equal([0,1], self.labels).astype(int))
        negative = numpy.sum(numpy.equal([1,0], self.labels).astype(int))
        
        return self.count, positive / 2, negative / 2
                
    
