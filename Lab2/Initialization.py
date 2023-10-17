import matplotlib.pyplot as plt
import numpy as np
from primary_src import W1A1
from utils import Menu


class Initalization:

    @staticmethod
    def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
        grads = {}
        costs = []
        m = X.shape[1]
        layers_dims = [X.shape[0], 10, 5, 1]
        
        if initialization == "zeros":
            parameters = Initalization.initialize_parameters_zeros(layers_dims)
        elif initialization == "random":
            parameters = Initalization.initialize_parameters_random(layers_dims)
        elif initialization == "he":
            parameters = Initalization.initialize_parameters_he(layers_dims)

        for i in range(num_iterations):

            # Forward propagation:  RELU(LINEAR) -> RELU(LINEAR) -> SIGMOID(LINEAR)
            a3, cache = W1A1.init_utils.forward_propagation(X, parameters)
            
            # Loss
            cost = W1A1.init_utils.compute_loss(a3, Y)

            # Backward propagation.
            grads = W1A1.init_utils.backward_propagation(X, Y, cache)
            
            # Update parameters.
            parameters = W1A1.init_utils.update_parameters(parameters, grads, learning_rate)
            
            # Print the loss every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
                costs.append(cost)
                
        # plot the loss
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters

    @staticmethod
    def run_test_exercise1():
        W1A1.public_tests.initialize_parameters_zeros_test(Initalization.initialize_parameters_zeros)

    @staticmethod
    def initialize_parameters_zeros(layers_dims, run_test = False):
        """
            Arguments:
            layer_dims -- python array (list) containing the size of each layer.
            
            Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                            b1 -- bias vector of shape (layers_dims[1], 1)
                            ...
                            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                            bL -- bias vector of shape (layers_dims[L], 1)
        """
        parameters = {}
        L = len(layers_dims)
        if run_test:
            Initalization.run_test_exercise1()
        for l in range(1, L):

            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters

        
    @staticmethod
    def run_test_exercise2():
        W1A1.public_tests.initialize_parameters_random_test(Initalization.initialize_parameters_random)

    @staticmethod
    def initialize_parameters_random(layers_dims, run_test = False):
        """
            Arguments:
            layer_dims -- python array (list) containing the size of each layer.
            
            Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                            b1 -- bias vector of shape (layers_dims[1], 1)
                            ...
                            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                            bL -- bias vector of shape (layers_dims[L], 1)
        """
        np.random.seed(3)               
        parameters = {}
        L = len(layers_dims)            
        if run_test:
            Initalization.run_test_exercise2()
        for l in range(1, L):
            
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters

    @staticmethod
    def run_test_exercise2():
        W1A1.public_tests.initialize_parameters_he_test(Initalization.initialize_parameters_he)

    @staticmethod
    def initialize_parameters_he(layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the size of each layer.
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
        """
        np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1 # integer representing the number of layers
        
        for l in range(1, L + 1):

            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / (layers_dims[l - 1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            
        return parameters
def main():
    content = [
        "Initalize with zero",
        "Test model with zero initalization",
        "Initialize with random number",
        "Test model with ranndom initalization",
        "Initialize with he",
        "Test model with he initalization",
        "Exit"
    ]

    menu = Menu(content=content)
    
    train_X, train_Y, test_X, test_Y = W1A1.init_utils.load_dataset()

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
                parameters = Initalization.initialize_parameters_zeros([3, 2, 1], run_test=True)
                
                print("-" * 30)
                print("W1 = " + str(parameters["W1"]))
                print("b1 = " + str(parameters["b1"]))
                print("W2 = " + str(parameters["W2"]))
                print("b2 = " + str(parameters["b2"]))
            case 2:
                parameters = Initalization.model(train_X, train_Y, initialization = "zeros")

                print ("On the train set:")
                predictions_train = W1A1.init_utils.predict(train_X, train_Y, parameters)   
                print ("On the test set:")
                predictions_test = W1A1.init_utils.predict(test_X, test_Y, parameters)

                print(f"Count prediction on train set is: {np.unique(np.array(predictions_train), return_counts=True)}")
                print(f"Count prediction on test set is: {np.unique(np.array(predictions_test), return_counts=True)}")

            case 3:
                parameters = Initalization.initialize_parameters_random(layers_dims=[3, 2, 1], run_test=True)

                print("-" * 30)
                print("W1 = " + str(parameters["W1"]))
                print("b1 = " + str(parameters["b1"]))
                print("W2 = " + str(parameters["W2"]))
                print("b2 = " + str(parameters["b2"]))

            case 4:
                parameters = Initalization.model(train_X, train_Y, initialization = "random")

                print ("On the train set:")
                predictions_train = W1A1.init_utils.predict(train_X, train_Y, parameters)   
                print ("On the test set:")
                predictions_test = W1A1.init_utils.predict(test_X, test_Y, parameters)

                print(f"Count prediction on train set is: {np.unique(np.array(predictions_train), return_counts=True)}")
                print(f"Count prediction on test set is: {np.unique(np.array(predictions_test), return_counts=True)}")
            case 5: 
                parameters = Initalization.initialize_parameters_random(layers_dims=[2, 4, 1], run_test=True)
                print("-" * 30)
                print("W1 = " + str(parameters["W1"]))
                print("b1 = " + str(parameters["b1"]))
                print("W2 = " + str(parameters["W2"]))
                print("b2 = " + str(parameters["b2"]))
            case 6:
                parameters = Initalization.model(train_X, train_Y, initialization = "he")

                print ("On the train set:")
                predictions_train = W1A1.init_utils.predict(train_X, train_Y, parameters)   
                print ("On the test set:")
                predictions_test = W1A1.init_utils.predict(test_X, test_Y, parameters)

                print(f"Count prediction on train set is: {np.unique(np.array(predictions_train), return_counts=True)}")
                print(f"Count prediction on test set is: {np.unique(np.array(predictions_test), return_counts=True)}")
            case 7:
                exit()
        continue
if __name__ == '__main__':
    main()
     

