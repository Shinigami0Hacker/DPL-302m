import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.layers as tfl

from keras.utils.image_dataset import image_dataset_from_directory
from keras.layers import RandomFlip, RandomRotation
import os
from utils import Menu
from primary_src import W2A2
import keras

class TransferLearning:
    """
    
    """
    
    IMG_SIZE = (160, 160)
    BATCH_SIZE = 32
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_learning_rate = 0.001

    initial_epochs = 5
    @staticmethod
    def load_dataset():
        """
        Load the dataset
        """

        directory = "./primary_src/W2A2/dataset/"

        TransferLearning.train_dataset = image_dataset_from_directory(directory,
                                                    shuffle=True, # Randomly change position
                                                    batch_size=TransferLearning.BATCH_SIZE, # Divieded into the batch
                                                    image_size=TransferLearning.IMG_SIZE, # Resize the image
                                                    validation_split=0.2, # Declare the size of the validation data
                                                    subset='training',
                                                    seed=42)

        TransferLearning.validation_dataset = image_dataset_from_directory(directory,
                                                    shuffle=True,
                                                    batch_size=TransferLearning.BATCH_SIZE,
                                                    image_size=TransferLearning.IMG_SIZE,
                                                    validation_split=0.2,
                                                    subset='validation',
                                                    seed=42)

    @staticmethod
    def show_dataset():
        class_names = TransferLearning.train_dataset.class_names
        plt.figure(figsize=(10, 10))
        for images, labels in TransferLearning.train_dataset.take(1): # Randomly take a batch of sample now is 32 sample per batch.
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

        plt.show()

    @staticmethod
    def visual_test_exrcise1():
        global data_augmentation
        data_augmentation = TransferLearning.data_augmenter()

        for image, _ in TransferLearning.train_dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')
        plt.show()

    @staticmethod
    def run_test_exercise1():
        augmenter = TransferLearning.data_augmenter()

        assert(augmenter.layers[0].name.startswith('random_flip')), "First layer must be RandomFlip"
        assert augmenter.layers[0].mode == 'horizontal', "RadomFlip parameter must be horizontal"
        assert(augmenter.layers[1].name.startswith('random_rotation')), "Second layer must be RandomRotation"
        assert augmenter.layers[1].factor == 0.2, "Rotation factor must be 0.2"
        assert len(augmenter.layers) == 2, "The model must have only 2 layers"

        print('All tests passed!')

    @staticmethod
    def data_augmenter():
        '''
        Create a Sequential model composed of 2 layers
        Returns:
            tf.keras.Sequential
        '''
        ### START CODE HERE
        data_augmentation = tf.keras.Sequential()
        data_augmentation.add(RandomFlip('horizontal'))
        data_augmentation.add(RandomRotation(0.2))
        ### END CODE HERE
        
        return data_augmentation

    @staticmethod
    def run_test_exercise2():
        global model2
        model2 = TransferLearning.alpaca_model(TransferLearning.IMG_SIZE, data_augmentation)

        alpaca_summary = [['InputLayer', [(None, 160, 160, 3)], 0],
                    ['Sequential', (None, 160, 160, 3), 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['Functional', (None, 5, 5, 1280), 2257984],
                    ['GlobalAveragePooling2D', (None, 1280), 0],
                    ['Dropout', (None, 1280), 0, 0.2],
                    ['Dense', (None, 1), 1281, 'linear']] #linear is the default activation

        W2A2.test_utils.comparator(W2A2.test_utils.summary(model2), alpaca_summary)

        for layer in W2A2.test_utils.summary(model2):
            print(layer)

    @staticmethod
    def alpaca_model(image_shape=(160, 160), data_augmentation=None):
        ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
        Arguments:
            image_shape -- Image width and height
            data_augmentation -- data augmentation function
        Returns:
        Returns:
            tf.keras.model
        '''
        data_augmentation = TransferLearning.data_augmenter()

        input_shape = image_shape + (3,)
        
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False, # <== Important!!!!
                                                    weights='imagenet') # From imageNet
        
        # freeze the base model by making it non trainable
        base_model.trainable = False

        # create the input layer (Same as the imageNetv2 input size)
        inputs = tf.keras.Input(shape=input_shape) 
        
        # apply data augmentation to the inputs
        x = data_augmentation(inputs)
        
        # data preprocessing using the same weights the model was trained on
        x = TransferLearning.preprocess_input(x) 
        
        # set training to False to avoid keeping track of statistics in the batch norm layer
        x = base_model(x, training=False) 
        
        # add the new Binary classification layers
        # use global avg pooling to summarize the info in each channel
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # include dropout with probability of 0.2 to avoid overfitting
        x = tf.keras.layers.Dropout(0.2)(x)
            
        # use a prediction layer with one neuron (as a binary classifier only needs one)
        prediction_layer = tfl.Dense(1)
        outputs = prediction_layer(x)
        
        ### END CODE HERE
        
        model = tf.keras.Model(inputs, outputs)
        
        return model
    
    @staticmethod
    def training_alpaca_model():
        
        model2.compile(optimizer=tf.keras.optimizers.Adam(lr=TransferLearning.base_learning_rate),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        global history
        history = model2.fit(TransferLearning.train_dataset, validation_data=TransferLearning.validation_dataset, epochs=TransferLearning.initial_epochs)

    @staticmethod
    def plot_alpaca_result():
        global acc, val_acc, loss, val_loss
        acc = [0.] + history.history['accuracy']
        val_acc = [0.] + history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
    
    @staticmethod
    def run_test_exercise3(loss_function, optimizer, metrics):
        assert type(loss_function) == keras.losses.BinaryCrossentropy, "Not the correct layer"
        assert loss_function.from_logits, "Use from_logits=True"
        assert type(optimizer) == keras.optimizers.Adam, "This is not an Adam optimizer"
        assert optimizer.lr == TransferLearning.base_learning_rate / 10, "Wrong learning rate"
        assert metrics[0] == 'accuracy', "Wrong metric"

        print('All tests passed!')

    @staticmethod
    def fine_tuning(run_test =  False):
        base_model = model2.layers[4]
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 120

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = True
            
        # Define a BinaryCrossentropy loss function. Use from_logits=True
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
        optimizer = tf.keras.optimizers.Adam(lr=0.1 * TransferLearning.base_learning_rate)
        # Use accuracy as evaluation metric
        metrics= ['accuracy']

        model2.compile(loss=loss_function,
                    optimizer = optimizer,
                    metrics=metrics)
        
        if run_test:
            TransferLearning.run_test_exercise3(loss_function, optimizer, metrics)

    @staticmethod
    def fine_tune_training():
        fine_tune_epochs = 5
        total_epochs =  TransferLearning.initial_epochs + fine_tune_epochs

        global history_fine
        history_fine = model2.fit(TransferLearning.train_dataset,
                                epochs=total_epochs,
                                initial_epoch=history.epoch[-1],
                                validation_data=TransferLearning.validation_dataset)
    
    @staticmethod
    def fine_tune_evaluation():
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.plot([TransferLearning.initial_epochs-1,TransferLearning.initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([TransferLearning.initial_epochs-1,TransferLearning.initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

def main():
    content = [
        "Load dataset",
        "Dataset visualization",
        "Data augmenter",
        "Alpaca model",
        "Train alpace model",
        "Evaluate alpca model",
        "Fine tuning",
        "Train fine tuning",
        "Evaluate fine tuning",
        "Exit"
    ]

    menu = Menu.Menu(content)
    
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
            TransferLearning.load_dataset()
        elif choice == 2:
            TransferLearning.visual_test_exrcise1()
        elif choice == 3:
            TransferLearning.run_test_exercise1()
        elif choice == 4:
            TransferLearning.run_test_exercise2()
        elif choice == 5:
            TransferLearning.training_alpaca_model()
        elif choice == 6:
            TransferLearning.plot_alpaca_result()
        elif choice == 7:
            TransferLearning.fine_tuning(run_test=True)
        elif choice == 8:
            TransferLearning.fine_tune_training()
        elif choice == 9:
            TransferLearning.fine_tune_evaluation()
        elif choice == 10:
            exit()

if __name__ == '__main__':
    main()
