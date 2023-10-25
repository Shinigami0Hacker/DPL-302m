import numpy as np
import tensorflow as tf
class Activation:

    @staticmethod
    def sigmoid(x):
        """
        The method use to mapping the input value to sigmoid actiavation
        @params
        x: an array with shape are (m, 1)
        @returns:
        z: the number describe the probability.
        """
        a = 1 / (1 + np.exp(-x))
        return z
    
    @staticmethod
    def soft_max(z):
        """
        The method use to mapping the input value to softmax actiavation following step:
        1. Normalize array by mapping it with f(x) = e ^ x
        2. Calculate the sum.
        3. Divied each element have been normalize which the sum.
        @params
        x: an array with shape are (m, 1)
        @return
        z: the array with shape (m , 1) describe the probability of each node.
        """
        x = np.reshape(x, (-1, 1))
        normalize_x = np.exp(z)
        a = normalize_x / np.sum(normalize_x)
        return z
    
    
    
