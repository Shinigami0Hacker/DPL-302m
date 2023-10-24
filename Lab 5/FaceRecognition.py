from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate
from keras.layers import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Layer
from keras import backend as K
from keras.models import model_from_json
from utils.Menu import Menu
from primary_src import W4A1

K.set_image_data_format('channels_last')

import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL

class FaceRecognition:
    json_file = open('./primary_src/W4A1/keras-facenet-h5/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('./primary_src/W4A1/keras-facenet-h5/model.h5')

    @staticmethod
    def run_test_exercise1():
        tf.random.set_seed(1)
        y_true = (None, None, None) # It is not used
        y_pred = (tf.keras.backend.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
                tf.keras.backend.random_normal([3, 128], mean=1, stddev=1, seed = 1),
                tf.keras.backend.random_normal([3, 128], mean=3, stddev=4, seed = 1))
        loss = FaceRecognition.triplet_loss(y_true, y_pred)

        assert type(loss) == tf.python.framework.ops.EagerTensor, "Use tensorflow functions"
        print("loss = " + str(loss))

        y_pred_perfect = ([[1., 1.]], [[1., 1.]], [[1., 1.,]])
        loss = FaceRecognition.triplet_loss(y_true, y_pred_perfect, 5)
        assert loss == 5, "Wrong value. Did you add the alpha to basic_loss?"
        y_pred_perfect = ([[1., 1.]],[[1., 1.]], [[0., 0.,]])
        loss = FaceRecognition.triplet_loss(y_true, y_pred_perfect, 3)
        assert loss == 1., "Wrong value. Check that pos_dist = 0 and neg_dist = 2 in this example"
        y_pred_perfect = ([[1., 1.]],[[0., 0.]], [[1., 1.,]])
        loss = FaceRecognition.triplet_loss(y_true, y_pred_perfect, 0)
        assert loss == 2., "Wrong value. Check that pos_dist = 2 and neg_dist = 0 in this example"
        y_pred_perfect = ([[0., 0.]],[[0., 0.]], [[0., 0.,]])
        loss = FaceRecognition.triplet_loss(y_true, y_pred_perfect, -2)
        assert loss == 0, "Wrong value. Are you taking the maximum between basic_loss and 0?"
        y_pred_perfect = ([[1., 0.], [1., 0.]],[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]])
        loss = FaceRecognition.triplet_loss(y_true, y_pred_perfect, 3)
        assert loss == 2., "Wrong value. Are you applying tf.reduce_sum to get the loss?"
        y_pred_perfect = ([[1., 1.], [2., 0.]], [[0., 3.], [1., 1.]], [[1., 0.], [0., 1.,]])
        loss = FaceRecognition.triplet_loss(y_true, y_pred_perfect, 1)
        if (loss == 4.):
            raise Exception('Perhaps you are not using axis=-1 in reduce_sum?')
        assert loss == 5, "Wrong value. Check your implementation"

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha = 0.2):
        """
        Implementation of the triplet loss as defined by formula (3)
        
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)
        
        Returns:
        loss -- real number, value of the loss
        """
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        
        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0), axis = -1)
        
        return loss
    
    @staticmethod
    def img_to_encoding(image_path, model):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = model.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)
    
    @staticmethod
    def test():
        danielle = tf.keras.preprocessing.image.load_img("./primary_src/W4A1/images/danielle.png", target_size=(160, 160))
        kian = tf.keras.preprocessing.image.load_img("images/kian.jpg", target_size=(160, 160))

    @staticmethod
    def run_test_exercise2():
        distance, door_open_flag = FaceRecognition.verify("./primary_src/W4A1/images/camera_0.jpg", "younes", database, FRmodel)
        assert np.isclose(distance, 0.5992949), "Distance not as expected"
        assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
        print("(", distance, ",", door_open_flag, ")")
    
    @staticmethod
    def verify(image_path, identity, database, model):
        """
        Function that verifies if the person on the "image_path" image is "identity".
        
        Arguments:
            image_path -- path to an image
            identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
            database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
            model -- your Inception model instance in Keras
        
        Returns:
            dist -- distance between the image_path and the image of "identity" in the database.
            door_open -- True, if the door should open. False otherwise.
        """
        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = FaceRecognition.img_to_encoding(image_path, model)
        # Step 2: Compute distance with identity's image (≈ 1 line)
        dist = np.linalg.norm(encoding - database[identity])
        # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
        if dist < 0.7:
            print("It's " + str(identity) + ", welcome in!")
            door_open = True
        else:
            print("It's not " + str(identity) + ", please go away")
            door_open = False   
        return dist, door_open

    @staticmethod
    def verification_test():
        FaceRecognition.verify("images/camera_2.jpg", "kian", database, FRmodel)

    @staticmethod
    def run_test_exercise3():
        distance, door_open_flag = FaceRecognition.verify("images/camera_0.jpg", "younes", database, FRmodel)
        assert np.isclose(distance, 0.5992949), "Distance not as expected"
        assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
        print("(", distance, ",", door_open_flag, ")")

    @staticmethod
    def load_database():
        global FRmodel
        FRmodel = FaceRecognition.model
        global database
        database = {}
        database["danielle"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/danielle.png", FRmodel)
        database["younes"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/younes.jpg", FRmodel)
        database["tian"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/tian.jpg", FRmodel)
        database["andrew"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/andrew.jpg", FRmodel)
        database["kian"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/kian.jpg", FRmodel)
        database["dan"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/dan.jpg", FRmodel)
        database["sebastiano"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/sebastiano.jpg", FRmodel)
        database["bertrand"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/bertrand.jpg", FRmodel)
        database["kevin"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/kevin.jpg", FRmodel)
        database["felix"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/felix.jpg", FRmodel)
        database["benoit"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/benoit.jpg", FRmodel)
        database["arnaud"] = FaceRecognition.img_to_encoding("./primary_src/W4A1/images/arnaud.jpg", FRmodel)

    @staticmethod
    def run_test_exercise3():
        FaceRecognition.who_is_it("./primary_src/W4A1/images/camera_0.jpg", database, FRmodel)

        # Test 2 with Younes pictures 
        test1 = FaceRecognition.who_is_it("./primary_src/W4A1/images/camera_0.jpg", database, FRmodel)
        assert np.isclose(test1[0], 0.5992946)
        assert test1[1] == 'younes'

        # Test 3 with Younes pictures 
        test2 = FaceRecognition.who_is_it("./primary_src/W4A1/images/younes.jpg", database, FRmodel)
        assert np.isclose(test2[0], 0.0)
        assert test2[1] == 'younes'
        
    @staticmethod
    def who_is_it(image_path, database, model):
        """
        Implements face recognition for the office by finding who is the person on the image_path image.
        
        Arguments:
            image_path -- path to an image
            database -- database containing image encodings along with the name of the person on the image
            model -- your Inception model instance in Keras
        
        Returns:
            min_dist -- the minimum distance between image_path encoding and the encodings from the database
            identity -- string, the name prediction for the person on image_path
        """
        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        encoding =  FaceRecognition.img_to_encoding(image_path, model)
        
        ## Step 2: Find the closest encoding ##
        
        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
            
            # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
            dist = np.linalg.norm(encoding - db_enc)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        if min_dist > 0.7:
            print("Not in the database.")
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        return min_dist, identity

def main():
    content = [
        "Compute Tripleloss",
        "Load database",
        "Verification",
        "Verification real-tes",
        "Recognition",
        "Exit"

    ]
    menu = Menu(content)
    while True:
        menu.print()
        try:
            choice = int(input("Enter your choice: "))
        except TypeError as err:
            print("Please enter your choice: ")
            continue
        is_valid = menu.validate(choice)
        if not is_valid:
            print("Please enter the integer")
            continue
        if choice == 1:
            FaceRecognition.run_test_exercise1()
        elif choice == 2:
            FaceRecognition.load_database()
        elif choice == 3:
            FaceRecognition.run_test_exercise2()
        elif choice == 4:
            FaceRecognition.verification_test()
        elif choice == 5:
            FaceRecognition.run_test_exercise2()
        elif choice == 6:
            exit()

if __name__ == "__main__":
    main()