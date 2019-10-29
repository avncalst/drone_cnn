# CNN model

import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import InputLayer
from keras.layers import Permute
from keras.layers import Reshape
##from keras.layers import output_shape
from keras.layers.merge import add
from keras.layers.core import Lambda
from keras import regularizers
from keras import backend as K
import tensorflow as tf


class avc_1Net:
    @staticmethod

 
    def build(width, height, depth, classes):
        """
        Define model architecture avc_1Net.
        
        # Arguments
           width: Target image widht.
           height: Target image height.
           depth: Target image channels.
           classes: Dimension of model output.
           
        # Returns
           model: A Model instance.
        """
##        import tensorflow as tf
        # Input
        img_input = Input(shape=(height, width, depth),name='img_input')
##        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(img_input)

        x1 = Conv2D(24, (5, 5), padding='same')(img_input)
##        x1 = Conv2D(24, (5, 5), padding='same')(x)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x1)
##        x1 = Dropout(0.2)(x1)

        x2 = Conv2D(48, (5, 5), padding='same')(x1)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x2)
##        x2 = Dropout(0.2)(x2)

        x3 = Conv2D(96, (5, 5), padding='same')(x2)
        x3 = Activation('relu')(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x3)
##        x3 = Dropout(0.2)(x3)

##        x4 = Flatten()(x3) # OpenCV/dnn does not support flatten
        a, b, c, d = x3.shape # returns dimension
##        print(x3.shape)
        a = b*c*d
        x4 = Permute([1, 2, 3])(x3)
        x4 = Reshape((int(a),))(x4) # convert dim -> int
        
        x4 = Dense(500)(x4)
	# x4 = Dense(300)(x4)
   ##	x4 = Dense(256)(x4)
	# x4 = Dense(128)(x4)
        x4 = Activation('relu')(x4)
        x4 = Dropout(0.2)(x4)

        uit = Dense(classes)(x4)
        uit = Activation('softmax')(uit)

        model = Model(inputs=[img_input], outputs=[uit])

##        print(model.summary())
##        plot_model(model, to_file='avc_1net.png')

        return model

    
    
