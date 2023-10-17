import tensorflow as tf
import numpy as np
from primary_src import W3A1
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.framework.ops import EagerTensor
import h5py
from utils import normalize, one_hot_matrix
from utils import Menu
import matplotlib.pyplot as plt

class TensorflowIntroduction:

    train_dataset = h5py.File('./primary_src/W3A1/datasets/train_signs.h5', "r")
    test_dataset = h5py.File('./primary_src/W3A1/datasets/test_signs.h5', "r")

    x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    new_train = x_train.map(normalize)
    new_test = x_test.map(normalize)

    new_y_test = y_test.map(one_hot_matrix)
    new_y_train = y_train.map(one_hot_matrix)

    @staticmethod
    def run_test_exercise1():
        result =  TensorflowIntroduction.linear_function()
        print(result)

        assert np.allclose(result, [[-2.15657382], [ 2.95891446], [-1.08926781], [-0.84538042]]), "Error"
        print("All test passed")

    @staticmethod
    def linear_function():
        """
        Implements a linear function: 
                Initializes X to be a random tensor of shape (3,1)
                Initializes W to be a random tensor of shape (4,3)
                Initializes b to be a random tensor of shape (4,1)
        Returns: 
        result -- Y = WX + b 
        """

        np.random.seed(1)
        
        X = tf.constant(np.random.randn(3, 1), name = "X")
        W = tf.Variable(np.random.randn(4, 3), name = "W")
        b = tf.Variable(np.random.randn(4, 1), name = "b")
        Y = tf.add(tf.matmul(W, X), b)

        return Y
    
    @staticmethod
    def run_test_exercise2():
        result = TensorflowIntroduction.sigmoid(-1)
        print ("type: " + str(type(result)))
        print ("dtype: " + str(result.dtype))
        print ("sigmoid(-1) = " + str(result))
        print ("sigmoid(0) = " + str(TensorflowIntroduction.sigmoid(0.0)))
        print ("sigmoid(12) = " + str(TensorflowIntroduction.sigmoid(12)))

        def sigmoid_test(target):
            result = target(0)
            assert(type(result) == EagerTensor)
            assert (result.dtype == tf.float32)
            assert TensorflowIntroduction.sigmoid(0) == 0.5, "Error"
            assert TensorflowIntroduction.sigmoid(-1) == 0.26894143, "Error"

            print("All test passed")

        sigmoid_test(TensorflowIntroduction.sigmoid)
    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z
        
        Arguments:
        z -- input value, scalar or vector
        
        Returns: 
        a -- (tf.float32) the sigmoid of z
        """


        z = tf.cast(z, tf.float32)
        a = tf.keras.activations.sigmoid(z)

        return a
  
    @staticmethod
    def run_test_exercise3():
        def one_hot_matrix_test(target):
            label = tf.constant(1)
            C = 4
            result = target(label, C)
            print("Test 1:",result)
            assert result.shape[0] == C, "Use the parameter C"
            assert np.allclose(result, [0., 1. ,0., 0.] ), "Wrong output. Use tf.one_hot"
            label_2 = [2]
            result = target(label_2, C)
            print("Test 2:", result)
            assert result.shape[0] == C, "Use the parameter C"
            assert np.allclose(result, [0., 0. ,1., 0.] ), "Wrong output. Use tf.reshape as instructed"
            
            print("All test passed")

        one_hot_matrix_test(TensorflowIntroduction.one_hot_matrix)
    
    @staticmethod
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
    
    @staticmethod
    def run_test_exercise4():
        def initialize_parameters_test(target):
            parameters = target()

            values = {"W1": (25, 12288),
                    "b1": (25, 1),
                    "W2": (12, 25),
                    "b2": (12, 1),
                    "W3": (6, 12),
                    "b3": (6, 1)}

            for key in parameters:
                print(f"{key} shape: {tuple(parameters[key].shape)}")
                assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
                assert np.abs(np.mean(parameters[key].numpy())) < 0.5,  f"{key}: Use the GlorotNormal initializer"
                assert np.std(parameters[key].numpy()) > 0 and np.std(parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"

            print("All test passed")

        initialize_parameters_test(TensorflowIntroduction.initialize_parameters)

    @staticmethod
    def initialize_parameters():
        """
        Initializes parameters to build a neural network with TensorFlow. The shapes are:
                            W1 : [25, 12288]
                            b1 : [25, 1]
                            W2 : [12, 25]
                            b2 : [12, 1]
                            W3 : [6, 12]
                            b3 : [6, 1]
        
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
                                
        initializer = tf.keras.initializers.GlorotNormal(seed=1)   

        W1 = tf.Variable(initializer(shape=(25,12288)))
        b1 = tf.Variable(initializer(shape=(25,1)))
        W2 = tf.Variable(initializer(shape=(12,25)))
        b2 = tf.Variable(initializer(shape=(12,1)))
        W3 = tf.Variable(initializer(shape=(6,12)))
        b3 = tf.Variable(initializer(shape=(6,1)))

        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                    "b3": b3}
        
        return parameters
    
    @staticmethod
    def run_test_exercise5():
        def forward_propagation_test(target, examples):
            minibatches = examples.batch(2)
            parametersk = TensorflowIntroduction.initialize_parameters()
            W1 = parametersk['W1']
            b1 = parametersk['b1']
            W2 = parametersk['W2']
            b2 = parametersk['b2']
            W3 = parametersk['W3']
            b3 = parametersk['b3']
            index = 0
            minibatch = list(minibatches)[0]
            with tf.GradientTape() as tape:
                forward_pass = target(tf.transpose(minibatch), parametersk)
                print(forward_pass)
                fake_cost = tf.reduce_mean(forward_pass - np.ones((6,2)))

                assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
                assert forward_pass.shape == (6, 2), "Last layer must use W3 and b3"
                assert np.allclose(forward_pass, 
                                [[-0.13430887,  0.14086473],
                                    [ 0.21588647, -0.02582335],
                                    [ 0.7059658,   0.6484556 ],
                                    [-1.1260961,  -0.9329492 ],
                                    [-0.20181894, -0.3382722 ],
                                    [ 0.9558965,   0.94167566]]), "Output does not match"
            index = index + 1
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(fake_cost, trainable_variables)
            assert not(None in grads), "Wrong gradients. It could be due to the use of tf.Variable whithin forward_propagation"
            print("All test passed")

        forward_propagation_test(TensorflowIntroduction.forward_propagation, TensorflowIntroduction.new_train)
    def forward_propagation(X, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
        
        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                    the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']
        
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.keras.activations.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.keras.activations.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)

        return Z3
    
    

    @staticmethod
    def run_test_exercise6():
        def compute_total_loss_test(target, Y):
            pred = tf.constant([[ 2.4048107,   5.0334096 ],
                    [-0.7921977,  -4.1523376 ],
                    [ 0.9447198,  -0.46802214],
                    [ 1.158121,    3.9810789 ],
                    [ 4.768706,    2.3220146 ],
                    [ 6.1481323,   3.909829  ]])
            minibatches = Y.batch(2)
            for minibatch in minibatches:
                result = target(pred, tf.transpose(minibatch))
                break
                
            print(result)
            assert(type(result) == EagerTensor), "Use the TensorFlow API"
            assert (np.abs(result - (0.50722074 + 1.1133534) / 2.0) < 1e-7), "Test does not match. Did you get the reduce sum of your loss functions?"

            print("All test passed")

        compute_total_loss_test(TensorflowIntroduction.compute_total_loss, TensorflowIntroduction.new_y_train)

    @staticmethod
    def compute_total_loss(logits, labels):
        """
        Computes the total loss
        
        Arguments:
        logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
        labels -- "true" labels vector, same shape as Z3
        
        Returns:
        total_loss - Tensor of the total loss value
        """

        total_loss =  tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))  

        return total_loss
    
    @staticmethod
    def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
        """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
        
        Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 10 epochs
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        costs = []                                        # To keep track of the cost
        train_acc = []
        test_acc = []
        
        # Initialize your parameters
        #(1 line)
        parameters = TensorflowIntroduction.initialize_parameters()

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # The CategoricalAccuracy will track the accuracy for this multiclass problem
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        dataset = tf.data.Dataset.zip((X_train, Y_train))
        test_dataset = tf.data.Dataset.zip((X_test, Y_test))
        
        # We can get the number of elements of a dataset using the cardinality method
        m = dataset.cardinality().numpy()
        
        minibatches = dataset.batch(minibatch_size).prefetch(8)
        test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
        #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
        #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster 

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_total_loss = 0.
            
            #We need to reset object to start measuring from 0 the accuracy each epoch
            train_accuracy.reset_states()
            
            for (minibatch_X, minibatch_Y) in minibatches:
                
                with tf.GradientTape() as tape:
                    # 1. predict
                    Z3 = TensorflowIntroduction.forward_propagation(tf.transpose(minibatch_X), parameters)

                    # 2. loss
                    minibatch_total_loss = TensorflowIntroduction.compute_total_loss(Z3, tf.transpose(minibatch_Y))

                # We accumulate the accuracy of all the batches
                train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
                
                trainable_variables = [W1, b1, W2, b2, W3, b3]
                grads = tape.gradient(minibatch_total_loss, trainable_variables)
                optimizer.apply_gradients(zip(grads, trainable_variables))
                epoch_total_loss += minibatch_total_loss
            
            # We divide the epoch total loss over the number of samples
            epoch_total_loss /= m

            # Print the cost every 10 epochs
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
                print("Train accuracy:", train_accuracy.result())
                
                # We evaluate the test set every 10 epochs to avoid computational overhead
                for (minibatch_X, minibatch_Y) in test_minibatches:
                    Z3 = TensorflowIntroduction.forward_propagation(tf.transpose(minibatch_X), parameters)
                    test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
                print("Test_accuracy:", test_accuracy.result())

                costs.append(epoch_total_loss)
                train_acc.append(train_accuracy.result())
                test_acc.append(test_accuracy.result())
                test_accuracy.reset_states()


        return parameters, costs, train_acc, test_acc
    
def main():
    content = [
        "Linear function",
        "Sigmoid",
        "One hot matrix",
        "Initialize parameters",
        "Forward propagation",
        "Compute total loss",
        "Training",
        "Exit"
    ]

    menu = Menu(content=content)

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
                TensorflowIntroduction.run_test_exercise1()
            case 2:
                TensorflowIntroduction.run_test_exercise2()
            case 3:
                TensorflowIntroduction.run_test_exercise3()
            case 4:
                TensorflowIntroduction.run_test_exercise4()
            case 5: 
                TensorflowIntroduction.run_test_exercise5()
            case 6:
                TensorflowIntroduction.run_test_exercise6()
            case 7:
                parameters, costs, train_acc, test_acc = TensorflowIntroduction.model(TensorflowIntroduction.new_train, 
                                                                                      TensorflowIntroduction.new_y_train,
                                                                                      TensorflowIntroduction.new_test,
                                                                                      TensorflowIntroduction.new_y_test, num_epochs=100)
                
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per fives)')
                plt.title("Learning rate =" + str(0.0001))
                plt.show()

                plt.plot(np.squeeze(train_acc))
                plt.ylabel('Train Accuracy')
                plt.xlabel('iterations (per fives)')
                plt.title("Learning rate =" + str(0.0001))
                
                plt.plot(np.squeeze(test_acc))
                plt.ylabel('Test Accuracy')
                plt.xlabel('iterations (per fives)')
                plt.title("Learning rate =" + str(0.0001))
                plt.show()
            case 8:
                exit()
        continue
if __name__ == '__main__':
   main()
