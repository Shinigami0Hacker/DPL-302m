import numpy as np
import matplotlib.pyplot as plt
from primary_src import W2A1
from utils import Menu
import math
class Optimazation:

    @staticmethod
    def run_test_exercise1():
        parameters, grads, learning_rate = W2A1.testCases.update_parameters_with_gd_test_case()
        learning_rate = 0.01
        parameters = Optimazation.update_parameters_with_gd(parameters, grads, learning_rate)

        print("W1 =\n" + str(parameters["W1"]))
        print("b1 =\n" + str(parameters["b1"]))
        print("W2 =\n" + str(parameters["W2"]))
        print("b2 =\n" + str(parameters["b2"]))

        W2A1.public_tests.update_parameters_with_gd_test(Optimazation.update_parameters_with_gd)

    @staticmethod
    def update_parameters_with_gd(parameters, grads, learning_rate):
        """
        Update parameters using one step of gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters to be updated:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients to update each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar.
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        L = len(parameters) // 2
        for l in range(1, L + 1):
            parameters['W' + str(l)] -= grads['dW' + str(l)] * learning_rate
            parameters['b' + str(l)] -= grads['db' + str(l)] * learning_rate
        return parameters

    @staticmethod
    def run_test_exercise2_v1():
        np.random.seed(1)
        mini_batch_size = 64
        nx = 12288
        m = 148
        X = np.array([x for x in range(nx * m)]).reshape((m, nx)).T
        Y = np.random.randn(1, m) < 0.5

        mini_batches = Optimazation.random_mini_batches(X, Y, mini_batch_size)
        n_batches = len(mini_batches)

        assert n_batches == math.ceil(m / mini_batch_size), f"Wrong number of mini batches. {n_batches} != {math.ceil(m / mini_batch_size)}"
        for k in range(n_batches - 1):
            assert mini_batches[k][0].shape == (nx, mini_batch_size), f"Wrong shape in {k} mini batch for X"
            assert mini_batches[k][1].shape == (1, mini_batch_size), f"Wrong shape in {k} mini batch for Y"
            assert np.sum(np.sum(mini_batches[k][0] - mini_batches[k][0][0], axis=0)) == ((nx * (nx - 1) / 2 ) * mini_batch_size), "Wrong values. It happens if the order of X rows(features) changes"
        if ( m % mini_batch_size > 0):
            assert mini_batches[n_batches - 1][0].shape == (nx, m % mini_batch_size), f"Wrong shape in the last minibatch. {mini_batches[n_batches - 1][0].shape} != {(nx, m % mini_batch_size)}"

        assert np.allclose(mini_batches[0][0][0][0:3], [294912,  86016, 454656]), "Wrong values. Check the indexes used to form the mini batches"
        assert np.allclose(mini_batches[-1][0][-1][0:3], [1425407, 1769471, 897023]), "Wrong values. Check the indexes used to form the mini batches"

        print("All tests passed!")

    @staticmethod
    def run_test_exercise2_v2():
        t_X, t_Y, mini_batch_size = W2A1.testCases.random_mini_batches_test_case()
        mini_batches = Optimazation.random_mini_batches(t_X, t_Y, mini_batch_size)

        print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
        print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
        print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
        print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
        print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
        print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
        print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

        W2A1.public_tests.random_mini_batches_test(Optimazation.random_mini_batches)

    @staticmethod
    def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        np.random.seed(seed)            
        m = X.shape[1]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))
        
        inc = mini_batch_size

        num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            
            mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]       
            
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:

            mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]       

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    @staticmethod
    def run_test_exercise3():
        parameters = W2A1.testCases.initialize_velocity_test_case()

        v = Optimazation.initialize_velocity(parameters)
        print("v[\"dW1\"] =\n" + str(v["dW1"]))
        print("v[\"db1\"] =\n" + str(v["db1"]))
        print("v[\"dW2\"] =\n" + str(v["dW2"]))
        print("v[\"db2\"] =\n" + str(v["db2"]))

        W2A1.public_tests.initialize_velocity_test(Optimazation.initialize_velocity)

    @staticmethod
    def initialize_velocity(parameters):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        
        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """
        
        L = len(parameters) // 2 
        v = {}

        for l in range(1, L + 1):
 
            v["dW" + str(l)] = np.zeros_like(parameters['W' + str(l)])
            v["db" + str(l)] = np.zeros_like(parameters['b' + str(l)])
            
        return v

    @staticmethod
    def run_test_exercise4():
        parameters, grads, v = W2A1.testCases.update_parameters_with_momentum_test_case()

        parameters, v = Optimazation.update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
        print("W1 = \n" + str(parameters["W1"]))
        print("b1 = \n" + str(parameters["b1"]))
        print("W2 = \n" + str(parameters["W2"]))
        print("b2 = \n" + str(parameters["b2"]))
        print("v[\"dW1\"] = \n" + str(v["dW1"]))
        print("v[\"db1\"] = \n" + str(v["db1"]))
        print("v[\"dW2\"] = \n" + str(v["dW2"]))
        print("v[\"db2\"] = v" + str(v["db2"]))

        W2A1.public_tests.update_parameters_with_momentum_test(Optimazation.update_parameters_with_momentum)
    @staticmethod
    def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
        L = len(parameters) // 2

        for l in range(1, L + 1):

            v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads['dW' + str(l)]
            v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads['db' + str(l)]

            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

        return parameters, v

    @staticmethod
    def run_test_exercise5():
        parameters = W2A1.testCasesinitialize_adam_test_case()

        v, s = Optimazation.initialize_adam(parameters)
        print("v[\"dW1\"] = \n" + str(v["dW1"]))
        print("v[\"db1\"] = \n" + str(v["db1"]))
        print("v[\"dW2\"] = \n" + str(v["dW2"]))
        print("v[\"db2\"] = \n" + str(v["db2"]))
        print("s[\"dW1\"] = \n" + str(s["dW1"]))
        print("s[\"db1\"] = \n" + str(s["db1"]))
        print("s[\"dW2\"] = \n" + str(s["dW2"]))
        print("s[\"db2\"] = \n" + str(s["db2"]))

        W2A1.public_tests.initialize_adam_test(Optimazation.initialize_adam)

    @staticmethod
    def initialize_adam(parameters) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters["W" + str(l)] = Wl
                        parameters["b" + str(l)] = bl
        
        Returns: 
        v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """
        
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}

        for l in range(1, L + 1):

            v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
            v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
            s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
            s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

        return v, s
    
    @staticmethod
    def run_test_exercise6():
        parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon =  W2A1.testCases.update_parameters_with_adam_test_case()

        parameters, v, s, vc, sc  = Optimazation.update_parameters_with_adam(parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon)
        print(f"W1 = \n{parameters['W1']}")
        print(f"W2 = \n{parameters['W2']}")
        print(f"b1 = \n{parameters['b1']}")
        print(f"b2 = \n{parameters['b2']}")

        W2A1.public_tests.update_parameters_with_adam_test(Optimazation.update_parameters_with_adam)

    @staticmethod
    def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                    beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        """
        Update parameters using Adam
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        t -- Adam variable, counts the number of taken steps
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """
        
        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary

        for l in range(1, L + 1):

            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))

            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2) 
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)
            
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))
            
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

        return parameters, v, s, v_corrected, s_corrected
    
    @staticmethod
    def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
              beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):
        """
        3-layer neural network model which can be run in different optimizer modes.
        
        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        optimizer -- the optimizer to be passed, gradient descent, momentum or adam
        layers_dims -- python list, containing the size of each layer
        learning_rate -- the learning rate, scalar.
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """

        L = len(layers_dims)             # number of layers in the neural networks
        costs = []                       # to keep track of the cost
        t = 0                            # initializing the counter required for Adam update
        seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
        m = X.shape[1]                   # number of training examples
        
        # Initialize parameters
        parameters = Optimazation.initialize_parameters(layers_dims)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = Optimazation.initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = Optimazation.initialize_adam(parameters)
        
        # Optimization loop
        for i in range(num_epochs):
            
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = Optimazation.random_mini_batches(X, Y, mini_batch_size, seed)
            cost_total = 0
            
            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                a3, caches = (minibatch_X, parameters)

                # Compute cost and add to the cost total
                cost_total += W2A1.opt_utils_v1a.compute_cost(a3, minibatch_Y)

                # Backward propagation
                grads = W2A1.opt_utils_v1a.backward_propagation(minibatch_X, minibatch_Y, caches)

                # Update parameters
                if optimizer == "gd":
                    parameters = Optimazation.update_parameters_with_gd(parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    parameters, v = Optimazation.update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s, _, _ = Optimazation.update_parameters_with_adam(parameters, grads, v, s,
                                                                t, learning_rate, beta1, beta2,  epsilon)
            cost_avg = cost_total / m
            
            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            if print_cost and i % 100 == 0:
                costs.append(cost_avg)
                    
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        return parameters
    
    @staticmethod
    def run_test_exercise7():
        learning_rate = 0.5
        print("Original learning rate: ", learning_rate)
        epoch_num = 2
        decay_rate = 1
        learning_rate_2 = Optimazation.update_lr(learning_rate, epoch_num, decay_rate)

        print("Updated learning rate: ", learning_rate_2)

        W2A1.public_tests.update_lr_test(Optimazation.update_lr)

    @staticmethod
    def update_lr(learning_rate0, epoch_num, decay_rate):
        """
        Calculates updated the learning rate using exponential weight decay.
        
        Arguments:
        learning_rate0 -- Original learning rate. Scalar
        epoch_num -- Epoch number. Integer
        decay_rate -- Decay rate. Scalar

        Returns:
        learning_rate -- Updated learning rate. Scalar 
        """
 
        learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate0

        return learning_rate
    
    @staticmethod
    def training():
        # train 3-layer model
        layers_dims = [Optimazation.train_X.shape[0], 5, 2, 1]
        parameters = Optimazation.model(Optimazation.train_X, Optimazation.train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=update_lr)

        # Predict
        predictions = W2A1.opt_utils_v1a.predict(Optimazation.train_X, Optimazation.train_Y, parameters)

        # Plot decision boundary
        plt.title("Model with Gradient Descent optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        W2A1.opt_utils_v1a.plot_decision_boundary(lambda x: W2A1.opt_utils_v1a.predict_dec(parameters, x.T), Optimazation.train_X, Optimazation.train_Y)

    @staticmethod
    def run_test_exercise8():
        learning_rate = 0.5
        print("Original learning rate: ", learning_rate)

        epoch_num_1 = 10
        epoch_num_2 = 100
        decay_rate = 0.3
        time_interval = 100
        learning_rate_1 = Optimazation.schedule_lr_decay(learning_rate, epoch_num_1, decay_rate, time_interval)
        learning_rate_2 = Optimazation.schedule_lr_decay(learning_rate, epoch_num_2, decay_rate, time_interval)
        print("Updated learning rate after {} epochs: ".format(epoch_num_1), learning_rate_1)
        print("Updated learning rate after {} epochs: ".format(epoch_num_2), learning_rate_2)

        W2A1.public_tests.schedule_lr_decay_test(Optimazation.schedule_lr_decay)
    
    @staticmethod
    def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
        """
        Calculates updated the learning rate using exponential weight decay.
        
        Arguments:
        learning_rate0 -- Original learning rate. Scalar
        epoch_num -- Epoch number. Integer.
        decay_rate -- Decay rate. Scalar.
        time_interval -- Number of epochs where you update the learning rate.

        Returns:
        learning_rate -- Updated learning rate. Scalar 
        """

        learning_rate = 1 / (1 + decay_rate * np.floor(epoch_num / time_interval)) * learning_rate0

        return learning_rate

    def training_with_lr_decay():
        layers_dims = [Optimazation.train_X.shape[0], 5, 2, 1]
        parameters = Optimazation.model(Optimazation.train_X, Optimazation.train_Y, layers_dims, optimizer = "momentum", learning_rate = 0.1, num_epochs=5000, decay=Optimazation.schedule_lr_decay)

        # Predict
        predictions = W2A1.opt_utils_v1a.predict(Optimazation.train_X, Optimazation.train_Y, parameters)
        
        plt.title("Model with Gradient Descent with momentum optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5,2.5])
        axes.set_ylim([-1,1.5])
        W2A1.opt_utils_v1a.plot_decision_boundary(lambda x: W2A1.opt_utils_v1a.predict_dec(parameters, x.T), Optimazation.train_X, Optimazation.train_Y)
def main():
    content = [
        "Update parameters with gradient descend",
        "Random mini batches v1",
        "Random mini batches v2",
        "Initialize velocity",
        "Initialize Adam",
        "Update parameters with Adam",
        ""
    ]

    menu = Menu(content)

    train_X, train_Y = W2A1.opt_utils_v1a.load_dataset()
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
                Optimazation.run_test_exercise1()
            case 2:
                Optimazation.run_test_exercise2_v1()
            case 3:
                Optimazation.run_test_exercise2_v2()
            case 4:
                Optimazation.run_test_exercise3()
            case 5:
                Optimazation.run_test_exercise4()
            case 6:
                Optimazation.run_test_exercise5()
            case 7:
                Optimazation.run_test_exercise6()
            case 8:
                Optimazation.run_test_exercise7()
            case 9:
                Optimazation.run_test_exercise8()
            case 10:
                exit()

if __name__ == '__main__':
        main()

