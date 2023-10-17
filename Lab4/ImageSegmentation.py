import tensorflow as tf
import numpy as np

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout 
from keras.layers import Conv2DTranspose
from keras.layers import concatenate

from primary_src.W3A2.test_utils import summary, comparator

import os
import numpy as np
import pandas as pd

import imageio

import matplotlib.pyplot as plt

from utils import Menu

from primary_src import W3A2
class ImageSegmentation:
    path = ''
    image_path = os.path.join(path, './primary_src/W3A2/data/CameraRGB/')
    mask_path = os.path.join(path, './primary_src/W3A2/data/CameraMask/')

    def process_data():
        """
        
        
        """
        image_list_orig = os.listdir(ImageSegmentation.image_path)
        
        global image_list, mask_list

        image_list = [ImageSegmentation.image_path + i for i in image_list_orig]
        mask_list = [ImageSegmentation.image_path + i for i in image_list_orig]
        image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
        mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)
        
        image_filenames = tf.constant(image_list)
        masks_filenames = tf.constant(mask_list)

        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
        def process_path(image_path, mask_path):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)

            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=3)
            mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
            return img, mask

        def preprocess(image, mask):
            input_image = tf.image.resize(image, (96, 128), method='nearest')
            input_mask = tf.image.resize(mask, (96, 128), method='nearest')

            return input_image, input_mask
        
        global image_ds, processed_image_ds
        image_ds = dataset.map(process_path)
        processed_image_ds = image_ds.map(preprocess)

    def display_sample():
        N = 2
        img = imageio.imread(image_list[N])
        mask = imageio.imread(mask_list[N])
        #mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

        fig, arr = plt.subplots(1, 2, figsize=(14, 10))
        arr[0].imshow(img)
        arr[0].set_title('Image')
        arr[1].imshow(mask[:, :, 0])
        arr[1].set_title('Segmentation')
        plt.show()

    @staticmethod
    def process_path(image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        return img, mask

    def preprocess(image, mask):
        input_image = tf.image.resize(image, (96, 128), method='nearest')
        input_mask = tf.image.resize(mask, (96, 128), method='nearest')

        return input_image, input_mask

    @staticmethod
    def run_test_exercise1():
        """
        
        """
        input_size=(96, 128, 3)
        n_filters = 32
        inputs = Input(input_size)
        cblock1 = ImageSegmentation.conv_block(inputs, n_filters * 1)
        model1 = tf.keras.Model(inputs=inputs, outputs=cblock1)

        output1 = [['InputLayer', [(None, 96, 128, 3)], 0],
                    ['Conv2D', (None, 96, 128, 32), 896, 'same', 'relu', 'HeNormal'],
                    ['Conv2D', (None, 96, 128, 32), 9248, 'same', 'relu', 'HeNormal'],
                    ['MaxPooling2D', (None, 48, 64, 32), 0, (2, 2)]]

        print('Block 1:')
        for layer in summary(model1):
            print(layer)

        comparator(summary(model1), output1)

        inputs = Input(input_size)
        cblock1 = ImageSegmentation.conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
        model2 = tf.keras.Model(inputs=inputs, outputs=cblock1)

        output2 = [['InputLayer', [(None, 96, 128, 3)], 0],
                    ['Conv2D', (None, 96, 128, 1024), 28672, 'same', 'relu', 'HeNormal'],
                    ['Conv2D', (None, 96, 128, 1024), 9438208, 'same', 'relu', 'HeNormal'],
                    ['Dropout', (None, 96, 128, 1024), 0, 0.1],
                    ['MaxPooling2D', (None, 48, 64, 1024), 0, (2, 2)]]
                
        print('\nBlock 2:')   
        for layer in summary(model2):
            print(layer)
            
        comparator(summary(model2), output2)

    @staticmethod
    def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
        """
        Convolutional downsampling block
        
        Arguments:
            inputs -- Input tensor
            n_filters -- Number of filters for the convolutional layers
            dropout_prob -- Dropout probability
            max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
        Returns: 
            next_layer, skip_connection --  Next layer and skip connection outputs
        """

        conv = Conv2D(filters=n_filters, # Number of filters
                    kernel_size=3,   # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(inputs)

        conv = Conv2D(filters=n_filters, # Number of filters
                    kernel_size=3,   # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv)
        
        # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)

        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        if max_pooling:
            ### START CODE HERE
            next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
            ### END CODE HERE
            
        else:
            next_layer = conv
            
        skip_connection = conv
        
        return next_layer, skip_connection

    @staticmethod
    def run_test_exercise2():
        input_size1=(12, 16, 256)
        input_size2 = (24, 32, 128)
        n_filters = 32
        expansive_inputs = Input(input_size1)
        contractive_inputs =  Input(input_size2)
        cblock1 = ImageSegmentation.upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
        model1 = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)

        output1 = [['InputLayer', [(None, 12, 16, 256)], 0],
                    ['Conv2DTranspose', (None, 24, 32, 32), 73760],
                    ['InputLayer', [(None, 24, 32, 128)], 0],
                    ['Concatenate', (None, 24, 32, 160), 0],
                    ['Conv2D', (None, 24, 32, 32), 46112, 'same', 'relu', 'HeNormal'],
                    ['Conv2D', (None, 24, 32, 32), 9248, 'same', 'relu', 'HeNormal']]

        print('Block 1:')
        for layer in summary(model1):
            print(layer)

        comparator(summary(model1), output1)

    @staticmethod
    def upsampling_block(expansive_input, contractive_input, n_filters=32):
        """
        Convolutional upsampling block
        
        Arguments:
            expansive_input -- Input tensor from previous layer
            contractive_input -- Input tensor from previous skip layer
            n_filters -- Number of filters for the convolutional layers
        Returns: 
            conv -- Tensor output
        """
        
        ### START CODE HERE
        up = Conv2DTranspose(
                    n_filters,    # number of filters
                    3,    # Kernel size
                    strides=2,
                    padding='same')(expansive_input)
        
        # Merge the previous output and the contractive_input
        merge = concatenate([up, contractive_input], axis=3)
        conv = Conv2D(n_filters,   # Number of filters
                    3,     # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(merge)
        conv = Conv2D(n_filters,  # Number of filters
                    3,   # Kernel size
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv)
        
        return conv

    @staticmethod
    def run_test_exercise3():
        img_height = 96
        img_width = 128
        num_channels = 3

        unet = ImageSegmentation.unet_model((img_height, img_width, num_channels))
        comparator(summary(unet), W3A2.outputs.unet_model_output)
    
    @staticmethod
    def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
        """
        Unet model
        
        Arguments:
            input_size -- Input shape 
            n_filters -- Number of filters for the convolutional layers
            n_classes -- Number of output classes
        Returns: 
            model -- tf.keras.Model
        """
        inputs = Input(input_size)
        # Contracting Path (encoding)
        # Add a conv_block with the inputs of the unet_ model and n_filters
        cblock1 = ImageSegmentation.conv_block(inputs, n_filters)
        # Chain the first element of the output of each block to be the input of the next conv_block. 
        # Double the number of filters at each new step
        cblock2 = ImageSegmentation.conv_block(cblock1[0], n_filters*2)
        cblock3 = ImageSegmentation.conv_block(cblock2[0], n_filters*4)
        cblock4 = ImageSegmentation.conv_block(cblock3[0], n_filters*8, dropout_prob=0.3) # Include a dropout of 0.3 for this layer
        # Include a dropout of 0.3 for this layer, and avoid the max_pooling layer
        cblock5 = ImageSegmentation.conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
        
        # Expanding Path (decoding)
        # Add the first upsampling_block.
        # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
        ### START CODE HERE
        ublock6 = ImageSegmentation.upsampling_block(cblock5[0], cblock4[1],  n_filters * 8)
        # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
        # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
        # At each step, use half the number of filters of the previous block 
        ublock7 = ImageSegmentation.upsampling_block(ublock6, cblock3[1],  n_filters * 4)
        ublock8 = ImageSegmentation.upsampling_block(ublock7, cblock2[1],  n_filters * 2)
        ublock9 = ImageSegmentation.upsampling_block(ublock8, cblock1[1],  n_filters)

        conv9 = Conv2D(n_filters,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(ublock9)

        # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
        ### START CODE HERE
        conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
        ### END CODE HERE
        
        model = tf.keras.Model(inputs=inputs, outputs=conv10)

        return model

    @staticmethod
    def model_initialize():
        img_height = 96
        img_width = 128
        num_channels = 3

        global unet
        unet = ImageSegmentation.unet_model((img_height, img_width, num_channels))
        unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    @staticmethod
    def model_summary():
        print(unet.summary())

    @staticmethod
    def display_input_and_mask(display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    @staticmethod
    def unet_training():
        EPOCHS = 5
        VAL_SUBSPLITS = 5
        BUFFER_SIZE = 500
        BATCH_SIZE = 32
        train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        print(processed_image_ds.element_spec)
        global model_history
        model_history = unet.fit(train_dataset, epochs=EPOCHS)

    @staticmethod
    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis] 
        return pred_mask[0]

    @staticmethod
    def evaluation():
        plt.plot(model_history.history["accuracy"])
    
    @staticmethod
    def show_predictions(dataset=None, num=1):
        """
        Displays the first image of each of the num batches
        """
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = unet.predict(image)
                ImageSegmentation.display_input_and_mask([image[0], mask[0], create_mask(pred_mask)])
        else:
            ImageSegmentation.display_input_and_mask([sample_image, sample_mask,
                ImageSegmentation.create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
def main():
    content = [
        "",
        "",
        "",
        "",
        "",
        "Predict on test image"
    ]
    menu = Menu.Menu(content)
    while True:
        menu.print()
        ImageSegmentation.process_data()
        try:
            choice = int(input("Enter your choice: "))
        except TypeError as err:
            print("Please enter the integer")
        validation = menu.validate(choice)
        if not validation:
            continue
        if choice == 1:
            ImageSegmentation.display_sample()
        elif choice == 2:
            ImageSegmentation.run_test_exercise1()
        elif choice == 3:
            ImageSegmentation.run_test_exercise2()
        elif choice == 4:
            ImageSegmentation.run_test_exercise3()
        elif choice == 5:
            pass
        elif choice == 6:
            out_scores, out_boxes, out_classes = ImageSegmentation.predict("./primary_src/W3A2/test.jpg")
        elif choice == 7:
            exit()

if __name__ == '__main__':
    main()