# USAGE
# python train_unsupervised_autoencoder.py --dataset output/images.pickle --model output/autoencoder.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf

# import the necessary packages
from imutils import paths,resize
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

#========================================================================================================
def sampling(args):
    mu,sigma=args
    eps=K.random_normal(shape=K.shape(mu))
    return mu+K.exp(sigma/2)*eps

#========================================================================================================

#==================
#parameters
#==================
width=128
height=128
depth=3
filters=(32,64)
# filters=(32,64,64)
latentDim=200
autoenc=True
EPOCHS = 30
INIT_LR = .5e-4 # small values (<1e-3) avoids exploding losses especially when larger latent vectors are used
BS = 128
BETA = 1e-4 # smaller numbers=better reconstruction images


# initialize the input shape to be "channels last" along with
# the channels dimension itself
# channels dimension itself

#==================
# encoder
#==================


inputShape = (height, width, depth)
chanDim = -1

# define the input to the encoder
inputs = Input(shape=inputShape)
x = inputs

# loop over the number of filters
for f in filters:
    # apply a CONV => RELU => BN operation
    x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

# flatten the network and then construct our latent vector
volumeSize = K.int_shape(x)
x = Flatten()(x)

if autoenc:

    mu = Dense(latentDim, name='latent_mu')(x)
    # sigma = Dense(latentDim, name='latent_sigma',kernel_initializer='zeros')(x)
    sigma = Dense(latentDim, name='latent_sigma')(x)
    z = Lambda(sampling,name='z')([mu,sigma])
    # build the encoder model
    encoder = Model(inputs, [mu,sigma,z], name="encoder")
    


else:    
    latent = Dense(latentDim)(x)
    # build the encoder model
    encoder = Model(inputs, latent, name="encoder")

encoder.summary()

#==================
# decoder
#==================

# start building the decoder model which will accept the
# output of the encoder as its inputs
latentInputs = Input(shape=(latentDim,))
x = Dense(np.prod(volumeSize[1:]))(latentInputs)
x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

# loop over our number of filters again, but this time in
# reverse order
for f in filters[::-1]:
    # apply a CONV_TRANSPOSE => RELU => BN operation
    x = Conv2DTranspose(f, (3, 3), strides=2,
        padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

# apply a single CONV_TRANSPOSE layer used to recover the
# original depth of the image
x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
outputs = Activation("sigmoid")(x)

# build the decoder model
decoder = Model(latentInputs, outputs, name="decoder")
decoder.summary()
if autoenc:
    out = decoder(encoder(inputs)[2]) # z as input decoder
    # out = decoder(encoder(inputs)[0]) # mu as input decoder
else:
    out = decoder(encoder(inputs))

#==================
# autoencoder
#==================
    
# our autoencoder is the encoder + decoder
autoencoder = Model(inputs, out, name="autoencoder")
autoencoder.summary()



#========================================================================================================
def visualize_predictions(decoded, gt, samples=10):
	# initialize our list of output images
	outputs = None

	# loop over our number of output samples
	for i in range(0, samples):
		# grab the original image and reconstructed image
		original = (gt[i] * 255).astype("uint8")
		recon = (decoded[i] * 255).astype("uint8")

		# stack the original and reconstructed image side-by-side
		output = np.hstack([original, recon])

		# if the outputs array is empty, initialize it as the current
		# side-by-side image display
		if outputs is None:
			outputs = output

		# otherwise, vertically stack the outputs
		else:
			outputs = np.vstack([outputs, output])

	# return the output images
	return outputs

#========================================================================================================
def kl_reconstruction_loss(true, pred):
    # Reconstruction loss
    reconstruction_loss = mean_squared_error(true,pred)
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum((1 + sigma - tf.math.pow(mu, 2) - tf.math.exp(sigma)), axis=1))
    return K.mean(reconstruction_loss + BETA*kl_loss)

#========================================================================================================

# initialize the number of epochs to train for, initial learning rate,


train_datagen = ImageDataGenerator(rescale=1./255,
                                    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    './mycar/vae/imagFace',
    target_size=(height,width),
    batch_size=BS,
    class_mode='input',
    subset='training') # set as training data


validation_generator = train_datagen.flow_from_directory(
    './mycar/vae/imagFace', # same directory as training data
    target_size=(height,width),
    batch_size=BS,
    class_mode='input',
    subset='validation') # set as validation data



# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
if autoenc:
    autoencoder.compile(loss=kl_reconstruction_loss, optimizer=opt,experimental_run_tf_function=False)
else:    
    autoencoder.compile(loss="mse", optimizer=opt)

# train the convolutional autoencoder
# H = autoencoder.fit(
# 	trainX, trainX,
# 	validation_data=(testX, testX),
# 	epochs=EPOCHS,
# 	batch_size=BS)

H = autoencoder.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BS,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BS,
    epochs = EPOCHS)

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk
print("[INFO] making predictions...")
example_batch = next(train_generator)
example_batch = example_batch[0]
testX = example_batch[:10]
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
vis = cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)
cv2.imwrite('./mycar/vae/recon_vis.png', vis)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("./mycar/vae/plot.png")

# # serialize the image data to disk
# print("[INFO] saving image data...")
# # f = open(args["dataset"], "wb")
# f = open("./vae/dataset", "wb")
# f.write(pickle.dumps(trainX))
# f.close()

# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
# autoencoder.save(args["model"], save_format="h5")
autoencoder.save("./mycar/vae/autoencoder.h5", save_format="h5")
encoder.save('./mycar/vae/encoder.h5',save_format='h5')
decoder.save('./mycar/vae/decoder.h5',save_format='h5')
