from DNNs.W4A1 import *
from DNNs.source import *

def Exercise1():
    print("Test Case 1:\n")
    parameters = initialize_parameters(3,2,1)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    initialize_parameters_test_1(initialize_parameters)

    print("\033[90m\nTest Case 2:\n")
    parameters = initialize_parameters(4,3,2)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    initialize_parameters_test_2(initialize_parameters)

def Exercise2():
    print("Test Case 1:\n")
    parameters = initialize_parameters_deep([5,4,3])

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    initialize_parameters_deep_test_1(initialize_parameters_deep)

    print("\033[90m\nTest Case 2:\n")
    parameters = initialize_parameters_deep([4,3,2])

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    initialize_parameters_deep_test_2(initialize_parameters_deep)
    
def Exercise3():
    t_A, t_W, t_b = linear_forward_test_case()
    t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
    print("Z = " + str(t_Z))

    linear_forward_test(linear_forward)

def Exercise4():
    t_A_prev, t_W, t_b = linear_activation_forward_test_case()

    t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "sigmoid")
    print("With sigmoid: A = " + str(t_A))

    t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "relu")
    print("With ReLU: A = " + str(t_A))

    linear_activation_forward_test(linear_activation_forward)

def Exercise5():
    t_X, t_parameters = L_model_forward_test_case_2hidden()
    t_AL, t_caches = L_model_forward(t_X, t_parameters)

    print("AL = " + str(t_AL))

    L_model_forward_test(L_model_forward)

def Exercise6():
    t_Y, t_AL = compute_cost_test_case()
    t_cost = compute_cost(t_AL, t_Y)

    print("Cost: " + str(t_cost))

    compute_cost_test(compute_cost)

def Exercise7():
    t_dZ, t_linear_cache = linear_backward_test_case()
    t_dA_prev, t_dW, t_db = linear_backward(t_dZ, t_linear_cache)

    print("dA_prev: " + str(t_dA_prev))
    print("dW: " + str(t_dW))
    print("db: " + str(t_db))

    linear_backward_test(linear_backward)

def Exercise8():
    t_dAL, t_linear_activation_cache = linear_activation_backward_test_case()

    t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "sigmoid")
    print("With sigmoid: dA_prev = " + str(t_dA_prev))
    print("With sigmoid: dW = " + str(t_dW))
    print("With sigmoid: db = " + str(t_db))

    t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "relu")
    print("With relu: dA_prev = " + str(t_dA_prev))
    print("With relu: dW = " + str(t_dW))
    print("With relu: db = " + str(t_db))

    linear_activation_backward_test(linear_activation_backward)

def Exercise9():
    t_AL, t_Y_assess, t_caches = L_model_backward_test_case()
    grads = L_model_backward(t_AL, t_Y_assess, t_caches)

    print("dA0 = " + str(grads['dA0']))
    print("dA1 = " + str(grads['dA1']))
    print("dW1 = " + str(grads['dW1']))
    print("dW2 = " + str(grads['dW2']))
    print("db1 = " + str(grads['db1']))
    print("db2 = " + str(grads['db2']))

def Exercise10():
    t_parameters, grads = update_parameters_test_case()
    t_parameters = update_parameters(t_parameters, grads, 0.1)

    print ("W1 = "+ str(t_parameters["W1"]))
    print ("b1 = "+ str(t_parameters["b1"]))
    print ("W2 = "+ str(t_parameters["W2"]))
    print ("b2 = "+ str(t_parameters["b2"]))

    update_parameters_test(update_parameters)
    L_model_backward_test(L_model_backward)

def printMenu():
    menu_data = {
        1: "Initialize parameters",
        2: "Initialize parameters deep",
        3: "Linear forward",
        4: "Linear activation forward",
        5: "L_model forward",
        6: "Compute cost",
        7: "Linear backward",
        8: "Linear activation backward",
        9: "L_model backward",
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