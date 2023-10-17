import numpy as np
from utils.Activatioin import Activation
from primary_src import W1A3 
from utils import Menu

class GradientChecking:

    @staticmethod
    def run_test_exercise1():
        x, theta = 2, 4
        J = GradientChecking.forward_propagation(x, theta)
        print ("J = " + str(J))
        W1A3.public_tests.forward_propagation_test(GradientChecking.forward_propagation)
    
    #Exercise 1
    @staticmethod
    def forward_propagation(x, theta):
        J = theta * x
        return 
    
    @staticmethod
    def run_test_exercise2  ():
        x, theta = 3, 4
        dtheta = GradientChecking.backward_propagation(x, theta)
        print ("dtheta = " + str(dtheta))
        W1A3.public_tests.backward_propagation_test(GradientChecking.backward_propagation)

    #Exercise 2
    @staticmethod
    def backward_propagation(x, theta):
        dtheta = x
        return dtheta
    
    def run_test_exercise3():
        x, theta = 3, 4
        difference = GradientChecking.gradient_check(x, theta, print_msg=True)
    
    #Exercise 3
    @staticmethod
    def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
        """
        """
        # A very small range from the theta in two directions
        theta_plus =  theta + epsilon
        theta_minus = theta - epsilon

        J_plus = GradientChecking.forward_propagation(x, theta_plus)
        J_minus = GradientChecking.forward_propagation(x, theta_minus)

        # Gradient approximation
        gradapprox = (J_plus - J_minus) / (2 * epsilon)

        grad = GradientChecking.backward_propagation(x, theta)

        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox) 
        difference = numerator / denominator
        if print_msg:
            if difference > 2e-7:
                print ("There is a mistake in the backward propagation! difference = " + str(difference))
            else:
                print ("Your backward propagation works perfectly fine! difference = " + str(difference))
    
        return difference

    @staticmethod
    def forward_propagation_n(X, Y, parameters):
        m = X.shape[1]
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        # Relu(LINEAR) -> Relu(LINEAR) -> SIGMOID(LINEAR)
        Z1 = np.dot(W1, X) + b1
        A1 = W1A3.gc_utils.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = W1A3.gc_utils.relu(Z1)
        Z3 = np.dot(W3, A2) + b3
        A3 = W1A3.gc_utils.relu(Z1)

        # Calculate the cost
        log_probs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
        cost = 1. / m * np.sum(log_probs)
        
        cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
        
        return cost, cache
    
    @staticmethod
    def backward_propagation_n(X, Y, cache):   
        m = X.shape[1]
        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
        
        dZ3 = A3 - Y
        dW3 = 1. / m * np.dot(dZ3, A2.T)
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
        
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1. / m * np.dot(dZ1, X.T)
        db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)
        
        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                    "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                    "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
        
        return gradients
    
    @staticmethod
    def run_test_exercise4():
        X, Y, parameters = W1A3.testCases.gradient_check_n_test_case()
        
        cost, cache = GradientChecking.forward_propagation_n(X, Y, parameters)
        
        gradients = GradientChecking.backward_propagation_n(X, Y, cache)
        difference = GradientChecking.gradient_check_n(parameters, gradients, X, Y, 1e-7, True)
        expected_values = [0.2850931567761623, 1.1890913024229996e-07]
        assert not(type(difference) == np.ndarray), "You are not using np.linalg.norm for numerator or denominator"
        assert np.any(np.isclose(difference, expected_values)), "Wrong value. It is not one of the expected values"

    @staticmethod
    def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
        
        # Set-up variables
        parameters_values, _ = W1A3.gc_utils.dictionary_to_vector(parameters)
        
        grad = W1A3.gc_utils.dictionary_to_vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        
        # Compute gradapprox
        for i in range(num_parameters):

            theta_plus = np.copy(parameters_values)
            theta_plus[i] = theta_plus[i] + epsilon
            J_plus[i], _ = GradientChecking.forward_propagation_n(X, Y, W1A3.gc_utils.vector_to_dictionary(theta_plus))

            theta_minus = np.copy(parameters_values)
            theta_minus[i] = theta_minus[i] - epsilon
            J_minus[i], _ = GradientChecking.forward_propagation_n(X, Y, W1A3.gc_utils.vector_to_dictionary(theta_minus))

            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
        if print_msg:
            if difference > 2e-7:
                print ("There is a mistake in the backward propagation! difference = " + str(difference))
            else:
                print ("Your backward propagation works perfectly fine! difference = " + str(difference))

        return difference

def main():
    content = [
        "Forward propagation",
        "Backward propagation",
        "Gradient check",
        "Backward propagation",
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
        match choice:
            case 1:
                GradientChecking.run_test_exercise1()
            case 2:
                GradientChecking.run_test_exercise2()
            case 3:
                GradientChecking.run_test_exercise3()
            case 4:
                GradientChecking.run_test_exercise4()
            case 5:
                exit()

if __name__ == '__main__':
    main()