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
from primary_src.W4A2.public_tests import *
from utils.Menu import Menu
tf.random.set_seed(272)
pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(input_shape=(img_size, img_size, 3), include_top= False,
                                weights='./primary_src/W4A2/pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False

#Load content
content_image = np.array(Image.open("./primary_src/images/louvre_small.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

#Load style
style_image =  np.array(Image.open("./primary_src/W4A2/images/monet.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

print(style_image.shape)
imshow(style_image[0])

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

content_image = np.array(Image.open("images/louvre_small.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

class NeuralStyleTransfer:
    @staticmethod
    def run_test_exercise1():
        compute_content_cost_test(NeuralStyleTransfer.compute_content_cost)

    @staticmethod
    def compute_content_cost(content_output, generated_output):
        """
        Computes the content cost
        
        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
        
        Returns: 
        J_content -- scalar that you compute using equation 1 above.
        """
        a_C = content_output[-1]
        a_G = generated_output[-1]
        
        # Retrieve dimensions from a_G (≈1 line)
        _, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape 'a_C' and 'a_G' (≈2 lines)
        # DO NOT reshape 'content_output' or 'generated_output'
        a_C_unrolled = tf.reshape(tf.transpose(a_C), (n_C, (n_H * n_W)))
        a_G_unrolled = tf.reshape(tf.transpose(a_G), (n_C, (n_H * n_W)))
        
        # compute the cost with tensorflow (≈1 line)
        J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))) / (4 * n_H *n_W * n_C)
        
        return J_content
    
    @staticmethod
    def run_test_exercise2():
        gram_matrix_test(NeuralStyleTransfer.gram_matrix)
    
    @staticmethod
    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)
        
        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """  
        GA = tf.matmul(A, tf.transpose(A))
        return GA
    
    @staticmethod
    def run_test_exercise3():
        compute_layer_style_cost_test(NeuralStyleTransfer.compute_layer_style_cost)

    @staticmethod
    def compute_layer_style_cost(a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
        
        Returns: 
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W) (≈2 lines)
        a_S = tf.reshape(tf.transpose(a_S,perm=[0,3,1,2]),(n_C,n_H*n_W))
        a_G = tf.reshape(tf.transpose(a_G,perm=[0,3,1,2]),(n_C,n_H*n_W))

        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = NeuralStyleTransfer.gram_matrix(a_S)
        GG = NeuralStyleTransfer.gram_matrix(a_G)
        J_style_layer =  tf.reduce_sum(tf.square(GS - GG)) / (4 * n_C * n_C * n_H * n_H * n_W * n_W)
        
        return J_style_layer

    @staticmethod
    def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers
        
        Arguments:
        style_image_output -- our tensorflow model
        generated_image_output --
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
        
        Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        
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
            J_style_layer = NeuralStyleTransfer.compute_layer_style_cost(a_S[i], a_G[i])

            # Add weight * J_style_layer of this layer to overall style cost
            J_style += weight[1] * J_style_layer

        return J_style

    @staticmethod
    def run_test_exercise4():
        total_cost_test(NeuralStyleTransfer.total_cost)

    @staticmethod
    @tf.function()
    def total_cost(J_content, J_style, alpha = 10, beta = 40):
        """
        Computes the total cost function
        
        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost
        
        Returns:
        J -- total cost as defined by the formula above.
        """

        J = alpha * J_content + beta * J_style

        return J

    @staticmethod
    def clip_0_1(image):
        """
        Truncate all the pixels in the tensor to be between 0 and 1
        
        Arguments:
        image -- Tensor
        J_style -- style cost coded above

        Returns:
        Tensor
        """
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def tensor_to_image(tensor):
        """
        Converts the given tensor into a PIL image
        
        Arguments:
        tensor -- Tensor
        
        Returns:
        Image: A PIL image
        """
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
    
    @staticmethod
    @tf.function()
    def train_step(generated_image):
        # Assign the input of the model to be the "style" image 
        preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
        a_S = vgg_model_outputs(preprocessed_style)
        content_layer = [('block5_conv4', 1)]
        def get_layer_outputs(vgg, layer_names):
            """ Creates a vgg model that returns a list of intermediate output values."""
            outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

            model = tf.keras.Model([vgg.input], outputs)
            return model
        preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        vgg_model_outputs = get_layer_outputs(NeuralStyleTransfer.vgg, NeuralStyleTransfer.STYLE_LAYERS + content_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        with tf.GradientTape() as tape:
            # In this function you must use the precomputed encoded images a_S and a_C
            # Compute a_G as the vgg_model_outputs for the current generated image
            a_G = vgg_model_outputs(generated_image)
            a_C = vgg_model_outputs(preprocessed_content)
            # Compute the style cost
            J_style = NeuralStyleTransfer.compute_style_cost(a_S, a_G)

            # Compute the content cost
            J_content = NeuralStyleTransfer.compute_content_cost(a_C, a_G)
            # Compute the total cost
            J = NeuralStyleTransfer.total_cost(J_content, J_style, 10, 40)
                
        grad = tape.gradient(J, generated_image)

        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(NeuralStyleTransfer.clip_0_1(generated_image))
        # For grading purposes
        return J

    @staticmethod
    def training():
        epochs = 2000
        for i in range(epochs):
            NeuralStyleTransfer.train_step(NeuralStyleTransfer.generated_image)
            if i % 250 == 0:
                print(f"Epoch {i} ")

    @staticmethod
    def show_result():
        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(1, 3, 1)
        imshow(NeuralStyleTransfer.content_image[0])
        ax.title.set_text('Content image')
        ax = fig.add_subplot(1, 3, 2)
        imshow(NeuralStyleTransfer.style_image[0])
        ax.title.set_text('Style image')
        ax = fig.add_subplot(1, 3, 3)
        imshow(NeuralStyleTransfer.generated_image[0])
        ax.title.set_text('Generated image')
        plt.show()

def main():
    content = [
        "Compute content cost",
        "Gram matrix",
        "Compute layer cost",
        "Total cost",
        "Training",
        "Show result",
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
            NeuralStyleTransfer.run_test_exercise1()
        elif choice == 2:
            NeuralStyleTransfer.run_test_exercise2()
        elif choice == 3:
            NeuralStyleTransfer.run_test_exercise3()
        elif choice == 4:
            NeuralStyleTransfer.run_test_exercise4()
        elif choice == 5:
            NeuralStyleTransfer.training()
        elif choice == 6:
            NeuralStyleTransfer.show_result()
        elif choice == 7:
            exit()
if __name__ == "__main__":
    main()