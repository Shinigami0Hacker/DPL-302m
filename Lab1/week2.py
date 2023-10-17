from LogisticRegression.W2A2 import *
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

def Exercise1():
    m_train = train_set_x_orig.shape[0]
    m_test = np.squeeze(test_set_y.shape)
    num_px = train_set_x_orig[0].shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

def Exercise2():
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
    assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

def Exercise3():
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    sigmoid_test(sigmoid)
    x = np.array([0.5, 0, 2.0])
    output = sigmoid(x)
    print(f"Example output: {output}")

def Exercise4():
    dim = 2
    w, b = initialize_with_zeros(dim)

    assert type(b) == float
    print ("w = " + str(w))
    print ("b = " + str(b))

    initialize_with_zeros_test_1(initialize_with_zeros)
    initialize_with_zeros_test_2(initialize_with_zeros)

def Exercise5():
    w =  np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    grads, cost = propagate(w, b, X, Y)

    assert type(grads["dw"]) == np.ndarray
    assert grads["dw"].shape == (2, 1)
    assert type(grads["db"]) == np.float64


    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))

    propagate_test(propagate)

def Exercise6():
    w =  np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print("Costs = " + str(costs))

    optimize_test(optimize)

def Exercise7():
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
    print ("predictions = " + str(predict(w, b, X)))

    predict_test(predict)

def Exercise8():
    model_test(model)

def printMenu():
    menu_data = {
        1: "Get dataset info",
        2: "Get dataset info after preprocess",
        3: "Sigmoid",
        4: "Initialize_with_zeros",
        5: "Propagate",
        6: "Optimize",
        7: "Predict",
        8: "Model",
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
        elif choice == 0:
            break