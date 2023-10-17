from primary_src import W1A2
import matplotlib.pyplot as plt
import numpy as np
from utils import Menu
import warnings
import os

os.environ["PYTHONWARNINGS"] = "ignore"

class Regularization:

    @staticmethod
    def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
        """
        Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
        learning_rate -- learning rate of the optimization
        num_iterations -- number of iterations of the optimization loop
        print_cost -- If True, print the cost every 10000 iterations
        lambd -- regularization hyperparameter, scalar
        keep_prob - probability of keeping a neuron active during drop-out, scalar.
        
        Returns:
        parameters -- parameters learned by the model. They can then be used to predict.
        """
        
        grads = {}
        costs = []                            # to keep track of the cost
        m = X.shape[1]                        # number of examples
        layers_dims = [X.shape[0], 20, 3, 1]
        W1A2.reg_utils
        # Initialize parameters dictionary.
        parameters = W1A2.reg_utils.initialize_parameters(layers_dims)

        # Loop (gradient descent)

        for i in range(0, num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            if keep_prob == 1:
                a3, cache = W1A2.reg_utils.forward_propagation(X, parameters)
            elif keep_prob < 1:
                a3, cache = Regularization.forward_propagation_with_dropout(X, parameters, keep_prob)
            
            # Cost function
            if lambd == 0:
                cost = W1A2.reg_utils.compute_cost(a3, Y)
            else:
                cost = Regularization.compute_cost_with_regularization(a3, Y, parameters, lambd)
                
            # Backward propagation.
            assert (lambd == 0 or keep_prob == 1)   # it is possible to use both L2 regularization and dropout, 
                                                    # but this assignment will only explore one at a time
            if lambd == 0 and keep_prob == 1:
                grads = W1A2.reg_utils.backward_propagation(X, Y, cache)
            elif lambd != 0:
                grads = Regularization.backward_propagation_with_regularization(X, Y, cache, lambd)
            elif keep_prob < 1:
                grads = Regularization.backward_propagation_with_dropout(X, Y, cache, keep_prob)
            
            # Update parameters.
            parameters = W1A2.reg_utils.update_parameters(parameters, grads, learning_rate)

            # Print the loss every 10000 iterations
            if print_cost and i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            if print_cost and i % 1000 == 0:
                costs.append(cost)
        
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
    
    @staticmethod
    def run_test_excercise1():
        A3, t_Y, parameters = W1A2.testCases.compute_cost_with_regularization_test_case()
        
        cost = Regularization.compute_cost_with_regularization(A3, t_Y, parameters, lambd=0.1)
        
        print("cost = " + str(cost))
        
        W1A2.public_tests.compute_cost_with_regularization_test(Regularization.compute_cost_with_regularization)

    @staticmethod
    def compute_cost_with_regularization(A3, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization. See formula (2) above.
        
        Arguments:
        A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model
        
        Returns:
        cost - value of the regularized loss function (formula (2))
        """
        m = Y.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        W3 = parameters["W3"]
        
        cross_entropy_cost = W1A2.reg_utils.compute_cost(A3, Y)

        L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
        
        cost = cross_entropy_cost + L2_regularization_cost
        
        return cost
    
    @staticmethod
    def run_test_excercise2():
        t_X, t_Y, cache = W1A2.testCases.backward_propagation_with_regularization_test_case()

        grads = Regularization.backward_propagation_with_regularization(t_X, t_Y, cache, lambd = 0.7)
        
        print ("dW1 = \n"+ str(grads["dW1"]))
        print ("dW2 = \n"+ str(grads["dW2"]))
        print ("dW3 = \n"+ str(grads["dW3"]))

        W1A2.public_tests.backward_propagation_with_regularization_test(Regularization.backward_propagation_with_regularization)

    @staticmethod
    def backward_propagation_with_regularization(X, Y, cache, lambd):
        """
        Implements the backward propagation of our baseline model to which we added an L2 regularization.
        
        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambd * W3) / m
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))

        dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd * W2) / m

        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))

        dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd * W1) / m 

        db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                    "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients

    @staticmethod
    def run_test_excercise3():
        t_X, parameters = W1A2.testCases.forward_propagation_with_dropout_test_case()

        A3, cache = Regularization.forward_propagation_with_dropout(t_X, parameters, keep_prob=0.7)
        print ("A3 = " + str(A3))

        W1A2.public_tests.forward_propagation_with_dropout_test(Regularization.forward_propagation_with_dropout)
            
    @staticmethod
    def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
        """
        Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
        
        Arguments:
        X -- input dataset, of shape (2, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (20, 2)
                        b1 -- bias vector of shape (20, 1)
                        W2 -- weight matrix of shape (3, 20)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns:
        A3 -- last activation value, output of the forward propagation, of shape (1,1)
        cache -- tuple, information stored for computing the backward propagation
        """
        
        np.random.seed(1)
        
        # retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        Z1 = np.dot(W1, X) + b1
        A1 = W1A2.reg_utils.relu(Z1)

        D1 = np.random.rand(A1.shape[0], A1.shape[1])
        D1 = D1 < keep_prob
        A1 = A1 * D1
        A1 /= keep_prob
        # YOUR CODE ENDS HERE
        Z2 = np.dot(W2, A1) + b2
        A2 = W1A2.reg_utils.relu(Z2)

        D2 = np.random.rand(A2.shape[0], A2.shape[1])
        D2 = D2 < keep_prob
        A2 = A2 * D2
        A2 /= keep_prob
        
        # YOUR CODE ENDS HERE
        Z3 = np.dot(W3, A2) + b3
        A3 = W1A2.reg_utils.sigmoid(Z3)
        
        cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
        
        return A3, cache

    @staticmethod
    def run_test_exercise4():
        t_X, t_Y, cache = W1A2.testCases.backward_propagation_with_dropout_test_case()

        gradients = Regularization.backward_propagation_with_dropout(t_X, t_Y, cache, keep_prob=0.8)

        print ("dA1 = \n" + str(gradients["dA1"]))
        print ("dA2 = \n" + str(gradients["dA2"]))

        W1A2.public_tests.backward_propagation_with_dropout_test(Regularization.backward_propagation_with_dropout)

    @staticmethod
    def backward_propagation_with_dropout(X, Y, cache, keep_prob):
        """
        Implements the backward propagation of our baseline model to which we added dropout.
        
        Arguments:
        X -- input dataset, of shape (2, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation_with_dropout()
        keep_prob - probability of keeping a neuron active during drop-out, scalar
        
        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        
        m = X.shape[1]
        (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1./m * np.dot(dZ3, A2.T)
        db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
        dA2 = np.dot(W3.T, dZ3)

        dA2 = dA2 * D2
        dA2 /= keep_prob


        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1./m * np.dot(dZ2, A1.T)
        db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(W2.T, dZ2)
        
        dA1 = dA1 * D1
        dA1 /= keep_prob

        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1./m * np.dot(dZ1, X.T)
        db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                    "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients

def main():
    content = [
        "Compute cost with regularization",
        "Backward propagation with regularization",
        "Forward propagation with dropout",
        "Backward propagation with dropout",
        "Training with dropout",
        "Exit"
    ]
    train_X, train_Y, test_X, test_Y = W1A2.reg_utils.load_2D_dataset()
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
                Regularization.run_test_excercise1()
            case 2:
                Regularization.run_test_excercise2()
            case 3:
                Regularization.run_test_excercise3()
            case 4:
                Regularization.run_test_exercise4()
            case 5:
                parameters = Regularization.model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
                print ("On the train set:")
                predictions_train = W1A2.reg_utils.predict(train_X, train_Y, parameters)
                print ("On the test set:")
                predictions_test = W1A2.reg_utils.predict(test_X, test_Y, parameters)
            case 6:
                exit()
if __name__ == '__main__':
    main()