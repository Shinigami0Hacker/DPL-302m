from SNN.source import *
from SNN.W3A1 import *

def Exercise1():
    X, Y = load_planar_dataset()
    shape_X = X.shape
    shape_Y = Y.shape
    m = Y.shape[1]
    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))

def Exercise2():
    t_X, t_Y = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(t_X, t_Y)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

    layer_sizes_test(layer_sizes)

def Exercise3():
    np.random.seed(2)
    n_x, n_h, n_y = initialize_parameters_test_case()
    parameters = initialize_parameters(n_x, n_h, n_y)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    initialize_parameters_test(initialize_parameters)
def Exercise4():
    t_X, parameters = forward_propagation_test_case()
    A2, cache = forward_propagation(t_X, parameters)
    print("A2 = " + str(A2))

    forward_propagation_test(forward_propagation)
def Exercise5():
    A2, t_Y = compute_cost_test_case()
    cost = compute_cost(A2, t_Y)
    print("cost = " + str(compute_cost(A2, t_Y)))

    compute_cost_test(compute_cost)
def Exercise6():
    parameters, cache, t_X, t_Y = backward_propagation_test_case()

    grads = backward_propagation(parameters, cache, t_X, t_Y)

    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))

    backward_propagation_test(backward_propagation)
def Exercise7():
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    update_parameters_test(update_parameters)

def Exercise8():
    nn_model_test(nn_model)

def Exercise9():
    parameters, t_X = predict_test_case()

    predictions = predict(parameters, t_X)
    print("Predictions: " + str(predictions))

    predict_test(predict)

def printMenu():
    menu_data = {
        1: "Get Shape",
        2: "Layer size",
        3: "Initilize value",
        4: "The Loop",
        5: "Compute cost",
        6: "Backward Propagation",
        7: "Update parameters",
        8: "Neural network model",
        9: "Predict",
        0: "Exit"
    }
    print(f"{'Menu':-^50}")
    for nums, title in menu_data.items():
        print(f"{nums}: {title}")
    print('-' * 50)
    
if __name__ == '__main__':
    while True:
        printMenu()
        try:
            choice = int(input("Enter your choice: "))
        except:
            continue
        if choice == 1:
            Exercise1()
        elif choice == 2:
            Exercise2()
        elif choice == 3:
            Exercise3()
        elif choice == 4:
            Exercise4()
        elif choice == 5:
            Exercise5()
        elif choice == 6:
            Exercise6()
        elif choice == 7:
            Exercise7()
        elif choice == 8:
            Exercise8()
        elif choice == 9:
            Exercise9()
        elif choice == 0:
            break


        