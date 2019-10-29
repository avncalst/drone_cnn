# USAGE
# python train_network.py --dataset imag --model avcnet.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import tensorflow as tf
# from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from avctech.avc_1netCV_tf import avc_1Net
from tensorflow.keras.utils import plot_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
ap.add_argument("-d", "--dataset", type=str, default='./imag_2',
	help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
ap.add_argument("-m", "--model", type=str, default='./TF_model/fin_model_tf.h5',
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="./TF_model/plot_tf.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
# EPOCHS = 1
EPOCHS = 30
INIT_LR = 1e-3 # smaller value lowers error peaks
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
                label = 0
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
print(labels)

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
aug = ImageDataGenerator(samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=False, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = avc_1Net.build(width=64, height=64, depth=3, classes=4)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(loss="binary_crossentropy", optimizer=opt,
#	metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss

checkpoint = ModelCheckpoint("./TF_model/my_model_tf.h5", monitor="val_loss",
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

model.save(args["model"],)
model.save('./TF_model/tf_model/', save_format='tf')
# m = tf.keras.models.load_model(args["model"])
# m.save('./TF_model/', save_format='tf')
# tf.saved_model.save(m, './TF_model/tf_model/')




# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
