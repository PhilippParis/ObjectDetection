# ObjectDetection
Object Detection in satellite and arial imagery in python using tensorflow. 

The main application of this project, which is part of an Bachelorthesis at the University of Technology in Vienna, is to detect bomb craters in arial images of the second world war.

This project is very much a work in progress!

## Installation
Needed:
* python 2.7
* opencv
* tensorflow

## Usage
### Training Data
the training data has to be organized as an image file (e.g. arial image) and an csv file (positions of the objects in the image that should be used for training). 
The csv file has to be organized as follows: "center x, center y, radius, label (1, 0)".

generate_datasets.py generates the csv file from an image where the center positions of the objects are marked with red dots.

### Training
To train the network edit the training parameters and the import of training data in train.py and run it.

### Detection
Edit the parameters in detect.py to start the object detection.
