import utils

# test image load / show
image = utils.getImage('../images/Land.jpg')
utils.showImage('full image', image)

#test sub image
#utils.showImage('sub image', utils.getSubImage(image, 500, 500, (100, 200)))

#test resizing image
#utils.showImage('resize image', utils.scaleImage(image, (500, 500)))

#test sliding window
#for window in utils.slidingWindow(image, 100, (100,100)):
#    utils.showImage('window', window)

