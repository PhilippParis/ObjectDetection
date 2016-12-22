"""
implements Data class which handles the training/evaluation examples
"""

import utils
import cv2
import csv
import numpy
import tensorflow as tf

class Data:
    'Data class for handling input data'
    
    def __init__(self, input_size, label_size):
        self.images = []     # images
        self.labels = []     # desired output data: [0,1] = crater; [1,0] = no crater
        self.count = 0                                       # number of images stored in this class
        self.index_in_epoch = 0                              # current start index
        self.img_size = input_size
        self.lbl_size = label_size
    # ============================================================= #    

    def finalize(self):
        """
        finalizes the datasets; must be called after adding all data
        """                
        size_input = self.img_size[0] * self.img_size[1]
        size_label = self.lbl_size[0] * self.lbl_size[1]
        
        
        self.images = numpy.reshape(self.images, newshape=(self.count, size_input))
        
        if size_label == 1:
            self.labels = numpy.reshape(self.labels, newshape=(self.count))
        else:
            self.labels = numpy.reshape(self.labels, newshape=(self.count, size_label))
            
        self.shuffle()
        
    # ============================================================= #    

    def add_examples(self, image_path, count, imgs_per_row, label):
        src_image = utils.getImage(image_path)
        x = 0
        y = 0
        
        for i in xrange(count):
            example = src_image[y : y + self.img_size[1], x : x + self.img_size[0]]
            
            for j in xrange(4):
                example = utils.rotateImage(example, 90)
                
                # add normal
                self.images.append(example)  
                # add inverted
                self.images.append(1.0 - example) 
                
                if not label is None:
                    # add label 
                    self.labels.append(label)
                    # add label for inverted
                    self.labels.append(label)
                    
                self.count += 2
                
            x += self.img_size[0]
            if x >= imgs_per_row * self.img_size[0]:
                x = 0
                y += self.img_size[1]
        del src_image
        
    # ============================================================= #
        
    def add_labels(self, image_path, count, imgs_per_row):
        src_image = utils.getImage(image_path)
        
        x = 0
        y = 0
        for i in xrange(count):
            label = src_image[y : y + self.lbl_size[1], x : x + self.lbl_size[0]]
            
            for j in xrange(4):
                label = utils.rotateImage(label, 90)
                
                # add label twice for inverted image
                self.labels.append(label.flatten('C'))
                self.labels.append(label.flatten('C'))
                
            x += self.lbl_size[0]
            if x >= imgs_per_row * self.lbl_size[0]:
                x = 0
                y += self.lbl_size[1]
        del src_image
        
    # ============================================================= #
    
    def shuffle(self):
        """
        shuffles the data
        """
        perm = numpy.arange(self.count) 
        numpy.random.shuffle(perm)
        self.images = self.images[perm]
        self.labels = self.labels[perm]
        self.index_in_epoch = 0
        
    # ============================================================= #    
                
    def next_batch(self, batch_size):
        """
        Args:
            batch_size: number of examples 
        Returns:
            returns the next batch of examples (input,labels)
            shuffles the examples at the end of an epoch (all examples returned)
        """
        
        if self.index_in_epoch > self.count:
            self.shuffle()
            
        start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch = end
        
        if end > self.count:
            end = self.count
        
        return (self.images[start:end], self.labels[start:end])
    
    # ============================================================= #   
    
    def batches(self, batch_size):
        steps = int(self.count / batch_size)
        
        start = 0
        end = batch_size
        for i in xrange(steps):
            yield (self.images[start:end], self.labels[start:end], batch_size)
            start += batch_size
            end += batch_size
        
        if start < self.count:
            yield (self.images[start:self.count], self.labels[start:self.count], self.count - start)
        
    # ============================================================= #  
                
    
