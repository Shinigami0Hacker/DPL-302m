import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

np.random.seed(1)


class FoundationCNN_v2:
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    @staticmethod
    def run_test_exercise1():
        happy_model = FoundationCNN_v2.happyModel()
        # Print a summary for each layer
        for layer in test_utils.summary(happy_model):
            print(layer)
            
        output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
                    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
                    ['BatchNormalization', (None, 64, 64, 32), 128],
                    ['ReLU', (None, 64, 64, 32), 0],
                    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
                    ['Flatten', (None, 32768), 0],
                    ['Dense', (None, 1), 32769, 'sigmoid']]
            
        test_utils.comparator(test_utils.summary(happy_model), output)

    @staticmethod
    def happyModel():
        """
        Implements the forward propagation for the binary classification model:
        ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
        
        Note that for simplicity and grading purposes, you'll hard-code all the values
        such as the stride and kernel (filter) sizes. 
        Normally, functions should take these values as function parameters.
        
        Arguments:
        None

        Returns:
        model -- TF Keras model (object containing the information for the entire training process) 
        """
        model = tf.keras.Sequential([
                tf.keras.layers.ZeroPadding2D(padding = (3, 3), input_shape = (64, 64, 3)),
                tf.keras.layers.Conv2D(32, (7, 7)),
                tf.keras.layers.BatchNormalization(axis = -1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])
        
        return model
    
    @staticmethod
    def run_test_exercise2():
        conv_model = FoundationCNN_v2.convolutional_model((64, 64, 3))
        conv_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        conv_model.summary()
            
        output = [['InputLayer', [(None, 64, 64, 3)], 0],
                ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
                ['ReLU', (None, 64, 64, 8), 0],
                ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
                ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
                ['ReLU', (None, 8, 8, 16), 0],
                ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
                ['Flatten', (None, 64), 0],
                ['Dense', (None, 6), 390, 'softmax']]
            
        comparator(summary(conv_model), output)
    
    @staticmethod
    def convolutional_model(input_shape):
        """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
        
        Note that for simplicity and grading purposes, you'll hard-code some values
        such as the stride and kernel (filter) sizes. 
        Normally, functions should take these values as function parameters.
        
        Arguments:
        input_img -- input dataset, of shape (input_shape)

        Returns:
        model -- TF Keras model (object containing the information for the entire training process) 
        """

        input_img = tf.keras.Input(shape=input_shape)

        Z1 = tfl.Conv2D(8, 4, activation='linear', padding="same", strides=1)(input_img)
        A1 = tfl.ReLU()(Z1)
        P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)
        Z2 = tfl.Conv2D(16, 2, activation='linear', padding="same", strides=1)(P1)
        A2 = tfl.ReLU()(Z2)
        P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)
        F = tfl.Flatten()(P2)
        outputs = tfl.Dense(6, activation='softmax')(F)
        
        model = tf.keras.Model(inputs=input_img, outputs=outputs)
        return model

