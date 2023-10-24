import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint

tf.random.set_seed(272)
pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Art_Generation_with_Neural_Style_Transfer\\pretrained-model\\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False
pp.pprint(vgg)

#  picture of the Louvre.
content_image = Image.open('C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Art_Generation_with_Neural_Style_Transfer\\images\\louvre.jpg')
print("The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.")
# compute_content_cost
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]  # a_C  tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G = generated_output[-1] #a_G  tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, (m, n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(a_C, (m, -1, n_C))
    # compute the cost with tensorflow 
    J_content =  tf.reduce_sum((a_C-a_G)**2)/(4*n_H*n_W*n_C)
    return J_content
# example style  image
example = Image.open("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Art_Generation_with_Neural_Style_Transfer\\images\\monet.jpg")
# Compute gram matrix
def gram_matrix(A): 
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    # a_S tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C])) 
    # a_G tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))
    # Computing gram_matrices for both images S and G 
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    # Computing the loss 
    J_style_layer =(tf.reduce_sum((GS-GG)**2)) / (4*(n_C**2)*((n_H*n_W)**2))     
    return J_style_layer

#Style weights
#start list of layer
for layer in vgg.layers:
    print(layer.name)
# choose layers to represent the style of the image and assign style costs
vgg.get_layer('block5_conv4').output
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# compute_style_cost
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
   # initialize the overall style cost
    J_style = 0
    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]
    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style

# compute total cost
# UNQ_C4
# GRADED FUNCTION: total_cost
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    # J_content  content cost 
    # J_style  style cost 
    # alpha  hyperparameter weighting the importance of the content cost
    # beta  hyperparameter weighting the importance of the style cost 
    J = alpha*J_content+beta*J_style
    return J

#load, reshape, and normalize  "content" image C (the Louvre museum picture)
content_image = np.array(Image.open("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Art_Generation_with_Neural_Style_Transfer\\images\\louvre_small.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
print(content_image.shape)
imshow(content_image[0])
plt.show()
# load, reshape and normalize your "style" image (Claude Monet's painting)
style_image =  np.array(Image.open("C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Art_Generation_with_Neural_Style_Transfer\\images\\monet.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
print(style_image.shape)
imshow(style_image[0])
plt.show()

# randomly Initialize the Image to be Generated
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()

# Creates a vgg model that returns a list of intermediate output values.
def get_layer_outputs(vgg, layer_names):
    
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# define the content layer 
content_layer = [('block5_conv4', 1)]
# build the model
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
# Save the outputs
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder

# Compute Total Cost
# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

# Compute the Style image Encoding (a_S) 
# Assign the input of the model to be the "style" image 
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)


# display the images generated by the style transfer model
def clip_0_1(image): # Truncate all the pixels in the tensor to be between 0 and 1
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):

   # Converts the given tensor into a PIL image
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# optimizer with adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:   
        # Compute a_G as the vgg_model_outputs for the current generated image    
        a_G = vgg_model_outputs(generated_image)     
        # Compute the style cost       
        J_style = compute_style_cost(a_S, a_G)
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style)       
    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J
generated_image = tf.Variable(generated_image)
# Train the Model
epochs = 2501
for i in range(epochs):
    
    train_step(generated_image)  
    if i % 10 == 0:
        print(f"Epoch {i} ")
    if i % 10 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"C:\\Users\\phuon\\Documents\\FPT\\CN5\\DPL302m\\FolderforcloningGithub\\DPL-302m\\Lab4_2\\Art_Generation_with_Neural_Style_Transfer\\output\\image_{i}.jpg")
     
      

# Show the 3 images in a row
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()