import tensorflow as tf
import numpy as np
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model, load_model
from primary_src import W2A1
from keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from utils import Menu

class ResidualsNetwork:
    """
    Submission of Residual network which is lab 1 in week 2 on convolution course.

    About the data:
    The data is about SIGNS dataset which is 6 classes:
    y = 0 => hand O
    y = 1 => one finger up
    y = 2 => hi (V)
    y = 3 => three finger count
    y = 4 => four finger count
    y = 5 => five finger count
    """


    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = W2A1.resnets_utils.load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = W2A1.resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = W2A1.resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

    

    @staticmethod
    def print_data_info():
        """
        Print the data infomation
        """
        print ("number of training examples = " + str(ResidualsNetwork.X_train.shape[0]))
        print ("number of test examples = " + str(ResidualsNetwork.X_test.shape[0]))
        print ("X_train shape: " + str(ResidualsNetwork.X_train.shape))
        print ("Y_train shape: " + str(ResidualsNetwork.Y_train.shape))
        print ("X_test shape: " + str(ResidualsNetwork.X_test.shape))
        print ("Y_test shape: " + str(ResidualsNetwork.Y_test.shape))

    @staticmethod
    def run_test_exercise1():
        tf.keras.backend.set_learning_phase(False) # Turn on the inference mode, use like test mode, not update the weight and bias

        np.random.seed(1)
        tf.random.set_seed(2)
        X1 = np.ones((1, 4, 4, 3)) * -1
        X2 = np.ones((1, 4, 4, 3)) * 1
        X3 = np.ones((1, 4, 4, 3)) * 3

        X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

        A3 = ResidualsNetwork.identity_block(X, f=2, filters=[4, 4, 3],
                        initializer=lambda seed=0:constant(value=1))
        print('With training=False\n')
        A3np = A3.numpy()
        print(np.around(A3.numpy()[:,(0,-1),:,:].mean(axis = 3), 5))
        resume = A3np[:,(0,-1),:,:].mean(axis = 3)
        print(resume[1, 1, 0])

        tf.keras.backend.set_learning_phase(True) # Reverse of the inference mode, update weight

        print('With training=True\n')
        np.random.seed(1)
        tf.random.set_seed(2)
        A4 = ResidualsNetwork.identity_block(X, f=2, filters=[3, 3, 3],
                        initializer=lambda seed=0:constant(value=1))
        print(np.around(A4.numpy()[:,(0,-1),:,:].mean(axis = 3), 5))

        W2A1.public_tests.identity_block_test(ResidualsNetwork.identity_block)

    @staticmethod
    def identity_block(X, f, filters, initializer=random_uniform):
        """
        Build the single identiy block.
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
        
        Returns:
        X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
        """      
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # valid - no padding, same - keep dims when convolution

        # First component of main path include (Conv2D - valid, BatchNormalization, and actiavation relu)
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X) # Default axis
        X = Activation('relu')(X)
        
        ## Second component of main path include (Conv2D - same, BatchNormalization, and actiavation relu)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        # Third component of main path include (Conv2D - valid, BatchNormalization)
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        
        # Final step (Plus the main path with the short path and apply ReLu activation on the sum)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    
    @staticmethod
    def run_test_exercise2():
        W2A1.public_tests.convolutional_block_test(ResidualsNetwork.convolutional_block)

    @staticmethod
    def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
        """
        Implementation of the convolutional block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        s -- Integer, specifying the stride to be used
        initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                    also called Xavier uniform initializer.
        
        Returns:
        X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
        """
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X
        
        # First component of main path glorot_uniform(seed=0)
        X = Conv2D(filters = F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        ## Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X) 
        X = Activation('relu')(X)

        ## Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        
        # The short path
        X_shortcut = Conv2D(filters = F3, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

        # Final step (Plus the main path with the short path and apply ReLu activation on the sum)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X
    
    @staticmethod
    def run_test_exercise3():
        tf.keras.backend.set_learning_phase(True)
        model = ResidualsNetwork.ResNet50(input_shape = (64, 64, 3), classes = 6)
        print(model.summary())

        model = ResidualsNetwork.ResNet50(input_shape = (64, 64, 3), classes = 6)

        W2A1.test_utils.comparator(W2A1.test_utils.summary(model), W2A1.outputs.ResNet50_summary)

    @staticmethod
    def ResNet50(input_shape = (64, 64, 3), classes = 6, training=False):
        """
        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = ResidualsNetwork.convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
        X = ResidualsNetwork.identity_block(X, 3, [64, 64, 256])
        X = ResidualsNetwork.identity_block(X, 3, [64, 64, 256])

        ### START CODE HERE
        
        # Use the instructions above in order to implement all of the Stages below
        # Make sure you don't miss adding any required parameter
        
        ## Stage 3
        # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = ResidualsNetwork.convolutional_block(X, f = 3, filters = [128,128,512], s = 2)
        
        # the 3 `identity_block` with correct values of `f` and `filters` for this stage
        X = ResidualsNetwork.identity_block(X, f=3, filters=[128,128,512])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[128,128,512])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[128,128,512])

        # Stage 4
        # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = ResidualsNetwork.convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
        
        # the 5 `identity_block` with correct values of `f` and `filters` for this stage
        X = ResidualsNetwork.identity_block(X, f=3, filters=[256, 256, 1024])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[256, 256, 1024])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[256, 256, 1024])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[256, 256, 1024])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[256, 256, 1024])

        # Stage 5 
        # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
        X = ResidualsNetwork.convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
        
        # the 2 `identity_block` with correct values of `f` and `filters` for this stage
        X = ResidualsNetwork.identity_block(X, f=3, filters=[512, 512, 2048])
        X = ResidualsNetwork.identity_block(X, f=3, filters=[512, 512, 2048])

        # AVGPOOL
        X = AveragePooling2D()(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
        model = Model(inputs = X_input, outputs = X)

        return model

    @staticmethod
    def training():
        global model
        model = ResidualsNetwork.ResNet50(input_shape = (64, 64, 3), classes = 6)
        np.random.seed(1)
        tf.random.set_seed(2)
        opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(ResidualsNetwork.X_train, ResidualsNetwork.Y_train, epochs = 10, batch_size = 32)
        return model

    @staticmethod
    def evaluation():
        global model
        if (model):
            preds = model.evaluate(ResidualsNetwork.X_test, ResidualsNetwork.Y_test)
            print ("Loss = " + str(preds[0]))
            print ("Test Accuracy = " + str(preds[1]))
        else:
            model = ResidualsNetwork.training()
            ResidualsNetwork.evaluation()

    @staticmethod
    def pre_trained_model():
        pre_trained_model = load_model('./primary_src/W2A1/resnet50.h5')
        preds = pre_trained_model.evaluate(ResidualsNetwork.X_test, ResidualsNetwork.Y_test)
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

def main():
    content = [
        "Identity block",
        "Convolutional block",
        "ResNet50",
        "Training",
        "Evaludation",
        "Pre-traiend model evaluation",
        "Exit"
    ]
    
    menu = Menu.Menu(content=content)
    while True:
        menu.print()
        try:
            choice = int(input("Enter your choice: "))
        except TypeError as err:
            print("Please enter the integer.")
            continue
        validation = menu.validate(choice)
        if not validation:
            continue
        if choice == 1:
            ResidualsNetwork.run_test_exercise1()
        elif choice == 2:
            ResidualsNetwork.run_test_exercise2()
        elif choice == 3:
            ResidualsNetwork.run_test_exercise3()
        elif choice == 4:
            ResidualsNetwork.training()
        elif choice == 5:
            ResidualsNetwork.evaluation()
        elif choice == 6:
            ResidualsNetwork.pre_trained_model()
        elif choice == 7:
            exit()
if __name__ == '__main__':
    main()