from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL

from tensorflow.keras.models import model_from_json

json_file = open('C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\keras-facenet-h5\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\keras-facenet-h5\\model.h5')

#EXERCISE 1
# UNQ_C1(UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: triplet_loss

def triplet_loss(y_true, y_pred, alpha = 0.2):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

# BEGIN UNIT TEST
tf.random.set_seed(1)
y_true = (None, None, None) # It is not used
y_pred = (tf.keras.backend.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
          tf.keras.backend.random_normal([3, 128], mean=1, stddev=1, seed = 1),
          tf.keras.backend.random_normal([3, 128], mean=3, stddev=4, seed = 1))
loss = triplet_loss(y_true, y_pred)

assert type(loss) == tf.python.framework.ops.EagerTensor, "Use tensorflow functions"
print("loss = " + str(loss))

y_pred_perfect = ([[1., 1.]], [[1., 1.]], [[1., 1.,]])
loss = triplet_loss(y_true, y_pred_perfect, 5)
assert loss == 5, "Wrong value. Did you add the alpha to basic_loss?"
y_pred_perfect = ([[1., 1.]],[[1., 1.]], [[0., 0.,]])
loss = triplet_loss(y_true, y_pred_perfect, 3)
assert loss == 1., "Wrong value. Check that pos_dist = 0 and neg_dist = 2 in this example"
y_pred_perfect = ([[1., 1.]],[[0., 0.]], [[1., 1.,]])
loss = triplet_loss(y_true, y_pred_perfect, 0)
assert loss == 2., "Wrong value. Check that pos_dist = 2 and neg_dist = 0 in this example"
y_pred_perfect = ([[0., 0.]],[[0., 0.]], [[0., 0.,]])
loss = triplet_loss(y_true, y_pred_perfect, -2)
assert loss == 0, "Wrong value. Are you taking the maximum between basic_loss and 0?"
y_pred_perfect = ([[1., 0.], [1., 0.]],[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]])
loss = triplet_loss(y_true, y_pred_perfect, 3)
assert loss == 2., "Wrong value. Are you applying tf.reduce_sum to get the loss?"
y_pred_perfect = ([[1., 1.], [2., 0.]], [[0., 3.], [1., 1.]], [[1., 0.], [0., 1.,]])
loss = triplet_loss(y_true, y_pred_perfect, 1)
if (loss == 4.):
    raise Exception('Perhaps you are not using axis=-1 in reduce_sum?')
assert loss == 5, "Wrong value. Check your implementation"

# END UNIT TEST

FRmodel = model
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)
database = {}
database["danielle"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\danielle.png", FRmodel)
database["younes"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\younes.jpg", FRmodel)
database["tian"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\kian.jpg", FRmodel)
database["dan"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\arnaud.jpg", FRmodel)

#EXERCISE 2
def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding-database[identity])
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False      
    return dist, door_open

# BEGIN UNIT TEST
distance, door_open_flag = verify("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\camera_0.jpg", "younes", database, FRmodel)
assert np.isclose(distance, 0.5992949), "Distance not as expected"
assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
print("(", distance, ",", door_open_flag, ")")
# END UNIT TEST

verify("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\camera_2.jpg", "kian", database, FRmodel)

#EXERCISE 3
def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

# BEGIN UNIT TEST
# Test 1 with Younes pictures 
who_is_it("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\camera_0.jpg", database, FRmodel)

# Test 2 with Younes pictures 
test1 = who_is_it("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\camera_0.jpg", database, FRmodel)
assert np.isclose(test1[0], 0.5992946)
assert test1[1] == 'younes'

# Test 3 with Younes pictures 
test2 = who_is_it("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Face_Recognition\\images\\younes.jpg", database, FRmodel)
assert np.isclose(test2[0], 0.0)
assert test2[1] == 'younes'
# END UNIT TEST

