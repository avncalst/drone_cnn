# USAGE
# python train_network.py --dataset imag --model avcnet.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
##from pyimagesearch.lenet import LeNet
from avctech.dronet import DroNet
from avctech.avc_1net import avc_1Net
from avctech.resnet import ResNet
from keras.utils import plot_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
#EPOCHS = 25
##EPOCHS = 30
EPOCHS = 50
INIT_LR = 1e-3 # smaller value lowers error peaks
##INIT_LR = 5e-4 # smaller value lowers error peaks
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
##	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##	image = cv2.equalizeHist(image)
	image = cv2.resize(image, (64, 64))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
##	print label
	if label == "stop":
                label=0
	if label == "left":
                label = 1 
	if label == "right":
                label = 2 
	if label == "fly":
                label = 3 
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print labels

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
##print trainX
##print testX

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=False, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
##model = DroNet.build(width=64, height=64, depth=3, classes=2)
model = avc_1Net.build(width=64, height=64, depth=3, classes=4)
##model = avc_1Net.build(width=64, height=64, depth=1, classes=2)
# model = ResNet.build(img_width=64,img_height=64,img_channels=3, output_dim=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(loss="binary_crossentropy", optimizer=opt,
#	metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss

checkpoint = ModelCheckpoint("./avcnet_best_6.hdf5", monitor="val_loss",
  save_best_only=True, verbose=1,save_weights_only=False,mode='auto',period=1)
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, callbacks=callbacks,verbose=1)

# save the model to disk
print("[INFO] serializing network...")
print ('summary',model.summary())
##plot_model(model, to_file='avc_1net.png')
model.save(args["model"])


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on fly/stop")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
