# USAGE
# python test_network.py --model avcnet.model --image images/examp/avc.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from operator import itemgetter

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
##ap.add_argument("-s", "--file", required=True,
##	help="path to output file")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()
##image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##image = cv2.equalizeHist(image)


# pre-process the image for classification
##image = cv2.resize(image, (28, 28))
image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"],custom_objects={"tf": tf})

# classify the input image
(stop, left,right,fly) = model.predict(image)[0]
my_dict = {'stop':stop, 'left':left, 'right':right,'fly':fly}
print my_dict
maxPair = max(my_dict.iteritems(), key=itemgetter(1))
label=maxPair[0]
proba=maxPair[1]

##lijst=[stop,left,right,fly]
##print lijst 
##idx=lijst.index(max(lijst))
##print idx, lijst[idx]
##if idx==0:
##    label="stop"
##if idx==1:
##    label="left"
##if idx==2:
##    label="right"
##if idx==3:
##    label="fly"

# build the label

##proba = lijst[idx]
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
##cv2.imwrite(args["file"],output)
cv2.waitKey(0)
