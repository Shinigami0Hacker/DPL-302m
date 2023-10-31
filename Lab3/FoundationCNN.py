import numpy as np
import matplotlib.pyplot as plt
from primary_src import W1A1
from utils import Menu

class FoundationCNN:
    """
    The week 1 of CNN which build the CNN model step by step include 6 graded function are:
    1. Zero padding using numpy.
    2. Convolution single step.
    3. Convolution forward.
    4. Pool forward.
    5. Convolution backward.
    6. Mask from window
    7. Distribute value
    """

    @staticmethod
    def run_test_exercise1():
        np.random.seed(1)
        x = np.random.randn(4, 3, 3, 2)
        x_pad = FoundationCNN.zero_pad(x, 3)
        print ("x.shape =\n", x.shape)
        print ("x_pad.shape =\n", x_pad.shape)
        print ("x[1,1] =\n", x[1, 1])
        print ("x_pad[1,1] =\n", x_pad[1, 1])

        fig, axarr = plt.subplots(1, 2)
        axarr[0].set_title('x')
        axarr[0].imshow(x[0, :, :, 0])
        axarr[1].set_title('x_pad')
        axarr[1].imshow(x_pad[0, :, :, 0])
        W1A1.public_tests.zero_pad_test(FoundationCNN.zero_pad)
    @staticmethod
    def zero_pad(X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
        as illustrated in Figure 1.
        
        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions
        
        Returns:
        X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
        """
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        return X_pad
    
    @staticmethod
    def run_test_exercise2():
        np.random.seed(1)
        a_slice_prev = np.random.randn(4, 4, 3)
        W = np.random.randn(4, 4, 3)
        b = np.random.randn(1, 1, 1)

        Z = FoundationCNN.conv_single_step(a_slice_prev, W, b)
        print("Z =", Z)
        W1A1.public_tests.conv_single_step_test(FoundationCNN.conv_single_step)

        assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
        assert np.isclose(Z, -6.999089450680221), "Wrong value"


    @staticmethod
    def conv_single_step(a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
        of the previous layer.
        
        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        
        Returns:
        Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
        """

        s = np.multiply(a_slice_prev, W)
        Z = np.sum(s)
        Z += float(b)
    
        return Z
    
    @staticmethod
    def run_test_exercise3():
        np.random.seed(1)
        A_prev = np.random.randn(2, 5, 7, 4)
        W = np.random.randn(3, 3, 4, 8)
        b = np.random.randn(1, 1, 1, 8)
        hparameters = {"pad" : 1,
                    "stride": 2}
        
        Z, cache_conv = FoundationCNN.conv_forward(A_prev, W, b, hparameters)
        z_mean = np.mean(Z)
        z_0_2_1 = Z[0, 2, 1]
        cache_0_1_2_3 = cache_conv[0][1][2][3]
        print("Z's mean =\n", z_mean)
        print("Z[0,2,1] =\n", z_0_2_1)
        print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)

        W1A1.public_tests.conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
        W1A1.public_tests.conv_forward_test_2(FoundationCNN.conv_forward)

    @staticmethod
    def conv_forward(A_prev, W, b, hparameters):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer, 
            numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape
        
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        n_H = int((n_H_prev+ (2 * pad) - f) / stride) + 1
        n_W = int((n_W_prev+(2 * pad) - f ) / stride) + 1
        
        Z =  np.zeros((m, n_H, n_W, n_C))
        
        A_prev_pad = FoundationCNN.zero_pad(A_prev, pad)
        
        for i in range(m):               # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]          # Select ith training example's padded activation
            for h in range(n_H):           # loop over vertical axis of the output volume
                vert_start = stride * h 
                vert_end = vert_start  + f
                
                for w in range(n_W):       # loop over horizontal axis of the output volume
                    # Find the horizontal start and end of the current "slice" (≈2 lines)
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    for c in range(n_C):   # loop over channels (= #filters) of the output volume
                                            
                        a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        
                        weights = W[:, :, :, c]
                        biases  = b[:, :, :, c]
                        Z[i, h, w, c] = FoundationCNN.conv_single_step(a_slice_prev, weights, biases)
        
        cache = (A_prev, W, b, hparameters)
        
        return Z, cache
    
    @staticmethod
    def run_test_exercise4():
        print("CASE 1:\n")
        np.random.seed(1)
        A_prev_case_1 = np.random.randn(2, 5, 5, 3)
        hparameters_case_1 = {"stride" : 1, "f": 3}

        A, cache = FoundationCNN.pool_forward(A_prev_case_1, hparameters_case_1, mode = "max")
        print("mode = max")
        print("A.shape = " + str(A.shape))
        print("A[1, 1] =\n", A[1, 1])
        A, cache = FoundationCNN.pool_forward(A_prev_case_1, hparameters_case_1, mode = "average")
        print("mode = average")
        print("A.shape = " + str(A.shape))
        print("A[1, 1] =\n", A[1, 1])

        W1A1.public_tests.pool_forward_test_1(pool_forward)

        # Case 2: stride of 2
        print("\n\033[0mCASE 2:\n")
        np.random.seed(1)
        A_prev_case_2 = np.random.randn(2, 5, 5, 3)
        hparameters_case_2 = {"stride" : 2, "f": 3}

        A, cache = FoundationCNN.pool_forward(A_prev_case_2, hparameters_case_2, mode = "max")
        print("mode = max")
        print("A.shape = " + str(A.shape))
        print("A[0] =\n", A[0])
        print()

        A, cache = FoundationCNN.pool_forward(A_prev_case_2, hparameters_case_2, mode = "average")
        print("mode = average")
        print("A.shape = " + str(A.shape))
        print("A[1] =\n", A[1])

        W1A1.public_tests.pool_forward_test_2(FoundationCNN.pool_forward)

    @staticmethod
    def pool_forward(A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        f = hparameters["f"]
        stride = hparameters["stride"]

        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        A = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):                         # loop over the training examples
            a_prev_slice = A_prev[i]
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                vert_start = stride * h 
                vert_end = vert_start + f
                for w in range(n_W):        # loop on the horizontal axis of the output volume
                    horiz_start = stride * w 
                    horiz_end = horiz_start + f
                    
                    for c in range (n_C):            # loop over the channels of the output volume
                        a_slice_prev = a_prev_slice[vert_start:vert_end,horiz_start:horiz_end,c]
                        
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_slice_prev)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_slice_prev)

        cache = (A_prev, hparameters)

        return A, cache
    
    @staticmethod
    def run_test_exercise5():
        np.random.seed(1)
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        hparameters = {"pad" : 2,
                    "stride": 2}
        Z, cache_conv = FoundationCNN.conv_forward(A_prev, W, b, hparameters)

        # Test conv_backward
        dA, dW, db = FoundationCNN.conv_backward(Z, cache_conv)

        print("dA_mean =", np.mean(dA))
        print("dW_mean =", np.mean(dW))
        print("db_mean =", np.mean(db))

        assert type(dA) == np.ndarray, "Output must be a np.ndarray"
        assert type(dW) == np.ndarray, "Output must be a np.ndarray"
        assert type(db) == np.ndarray, "Output must be a np.ndarray"
        assert dA.shape == (10, 4, 4, 3), f"Wrong shape for dA  {dA.shape} != (10, 4, 4, 3)"
        assert dW.shape == (2, 2, 3, 8), f"Wrong shape for dW {dW.shape} != (2, 2, 3, 8)"
        assert db.shape == (1, 1, 1, 8), f"Wrong shape for db {db.shape} != (1, 1, 1, 8)"
        assert np.isclose(np.mean(dA), 1.4524377), "Wrong values for dA"
        assert np.isclose(np.mean(dW), 1.7269914), "Wrong values for dW"
        assert np.isclose(np.mean(db), 7.8392325), "Wrong values for db"

        print("\033[92m All tests passed.")

    @staticmethod
    def conv_backward(dZ, cache):
        """
        Implement the backward propagation for a convolution function
        
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
        
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
            numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
            numpy array of shape (1, 1, 1, n_C)
        """    
    
        (A_prev, W, b, hparameters) = cache
        
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        (f, f, n_C_prev, n_C) = W.shape
        
        (m, n_H, n_W, n_C) = dZ.shape
        
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros(W.shape)
        db = np.zeros(b.shape)
        
        A_prev_pad = FoundationCNN.zero_pad(A_prev, pad)
        dA_prev_pad = FoundationCNN.pool_forwardzero_pad(dA_prev, pad)
        
        for i in range(m):
            
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = stride * h
                        vert_end = vert_start + f
                        
                        horiz_start = stride * w
                        horiz_end = horiz_start + f
                        
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                        
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        return dA_prev, dW, db
    
    @staticmethod
    def run_test_exercise6():
        np.random.seed(1)
        x = np.random.randn(2, 3)
        mask = FoundationCNN.create_mask_from_window(x)
        print('x = ', x)
        print("mask = ", mask)

        x = np.array([[-1, 2, 3],
                    [2, -3, 2],
                    [1, 5, -2]])

        y = np.array([[False, False, False],
            [False, False, False],
            [False, True, False]])
        mask = FoundationCNN.create_mask_from_window(x)

        assert type(mask) == np.ndarray, "Output must be a np.ndarray"
        assert mask.shape == x.shape, "Input and output shapes must match"
        assert np.allclose(mask, y), "Wrong output. The True value must be at position (2, 1)"

        print("\033[92m All tests passed.")

    @staticmethod
    def create_mask_from_window(x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """    
        mask = (x == np.max(x))
        return mask
    
    @staticmethod
    def run_test_exercise7():
        a = FoundationCNN.distribute_value(2, (2, 2))
        print('distributed value =', a)


        assert type(a) == np.ndarray, "Output must be a np.ndarray"
        assert a.shape == (2, 2), f"Wrong shape {a.shape} != (2, 2)"
        assert np.sum(a) == 2, "Values must sum to 2"

        a = FoundationCNN.distribute_value(100, (10, 10))
        assert type(a) == np.ndarray, "Output must be a np.ndarray"
        assert a.shape == (10, 10), f"Wrong shape {a.shape} != (10, 10)"
        assert np.sum(a) == 100, "Values must sum to 100"

        print("All tests passed.")

    @staticmethod
    def distribute_value(dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
        
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """    

        # YOUR CODE STARTS HERE
        (n_H, n_W) = shape
        average = np.prod(shape)
        a = (dz / average) * np.ones(shape)

        return a
    
    @staticmethod
    def run_test_exercise8():
        np.random.seed(1)
        A_prev = np.random.randn(5, 5, 3, 2)
        hparameters = {"stride" : 1, "f": 2}
        A, cache = FoundationCNN.pool_forward(A_prev, hparameters)
        print(A.shape)
        print(cache[0].shape)
        dA = np.random.randn(5, 4, 2, 2)

        dA_prev1 = FoundationCNN.pool_backward(dA, cache, mode = "max")
        print("mode = max")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev1[1,1] = ', dA_prev1[1, 1])  
        print()
        dA_prev2 = FoundationCNN.pool_backward(dA, cache, mode = "average")
        print("mode = average")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev2[1,1] = ', dA_prev2[1, 1]) 

        assert type(dA_prev1) == np.ndarray, "Wrong type"
        assert dA_prev1.shape == (5, 5, 3, 2), f"Wrong shape {dA_prev1.shape} != (5, 5, 3, 2)"
        assert np.allclose(dA_prev1[1, 1], [[0, 0], 
                                            [ 5.05844394, -1.68282702],
                                            [ 0, 0]]), "Wrong values for mode max"
        assert np.allclose(dA_prev2[1, 1], [[0.08485462,  0.2787552], 
                                            [1.26461098, -0.25749373], 
                                            [1.17975636, -0.53624893]]), "Wrong values for mode average"
        print("All tests passed.")

    @staticmethod
    def pool_backward(dA, cache, mode = "max"):
        """
        Implements the backward pass of the pooling layer
        
        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        (A_prev, hparameters) = cache
        
        stride = hparameters['stride']
        f = hparameters['f']
        
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):
            a_prev = A_prev[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = stride * h
                        vert_end = vert_start + f
                        horiz_start = stride * w
                        horiz_end = horiz_start + f
                        if mode == 'max':
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = FoundationCNN.create_mask_from_window(a_prev_slice)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        elif mode == 'average':
                            da = dA[i, h, w, c]
                            shape = (f, f)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += FoundationCNN.distribute_value(da, shape)
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_prev

def main():
    content = [
        "Zero padding using numpy",
        "Convolution single step",
        "Convolution forward",
        "Pool backward"
        "Convolution backward",
        "Mask from window",
        "Distribute value",
        "Pool backward"
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
        match choice:
            case 1:
                FoundationCNN.run_test_exercise1()
            case 2:
                FoundationCNN.run_test_exercise2()
            case 3:
                FoundationCNN.run_test_exercise3()
            case 4:
                FoundationCNN.run_test_exercise4()
            case 5:
                FoundationCNN.run_test_exercise5()
            case 6:
                FoundationCNN.run_test_exercise6()
            case 7:
                FoundationCNN.run_test_exercise7()
            case 8:
                FoundationCNN.run_test_exercise7()
            case 9:
                FoundationCNN.run_test_exercise8()
            case 10:
                exit()

if __name__ == '__main__':
    pass