import tensorflow as tf

def normalize(image):
        """
        Transform an image into a tensor of shape (64 * 64 * 3, )
        and normalize its components.
        
        Arguments
        image - Tensor.
        
        Returns: 
        result -- Transformed tensor 
        """
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [-1,])
        return image

def one_hot_matrix(label, C=6):
    """
    Computes the one hot encoding for a single label
    
    Arguments:
        label --  (int) Categorical labels
        C --  (int) Number of different classes that label can take
    
    Returns:
        one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """

    one_hot = tf.reshape(tf.one_hot(label, C, axis = 0), shape = [C, ])

    return one_hot
