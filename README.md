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
### Training/Evaluation Data
Training/Evaluation examples can be imported in two ways:

#### import from image + csv
 the data is organized as image file (e.g. arial image of mars) and a corresponding csv file (which holds the positions and size of the examples in the image: "center x, center y, radius, label (1, 0)", e.g. position of craters in the image)
-> input_data.add("path to csv", "path to image")

#### import from Spritesheets
the data is organized as "spritesheet": two images, one containing all positive, the other all negative training examples side by side, row by row 
-> input_data.add_from_single_image(...)

generate_datasets.py can be used to generate the csv file from an image where the center positions of the objects are marked with red dots.
datasets_to_images.py can be used to create two "spritesheets" from imported training data (image/csv)

### Training
To train the network edit the training parameters and the import of training data in train_....py and run it.

### Detection
Edit the parameters in detect.py to start the object detection.
