from DNNs.source_v2 import *
from DNNs.W4A2 import *
import matplotlib.pyplot as plt
from PIL import Image

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075


def plot_image():
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    plt.show()

def Exercise1():
 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
def Exercise2():
    parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2, print_cost=False)

    print("Cost after first iteration: " + str(costs[0]))

    two_layer_model_test(two_layer_model)

def Exercise3():
    def plot_costs(costs, learning_rate=0.0075):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    global parameters
    parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)
    plot_costs(costs, learning_rate)

def Exercise4():
    layers_dims = [12288, 20, 7, 5, 1]
    global parameters
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

    print("Cost after first iteration: " + str(costs[0]))

    L_layer_model_test(L_layer_model)
def Exercise5():
    layers_dims = [12288, 20, 7, 5, 1]
    global parameters
    parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def real_test():
    my_image = r"C:\Users\skt1t\OneDrive\Máy tính\Lab_sumission\Lab1\DNNs\W4A2\images\my_image.jpg"
    my_label_y = [1]

    image = np.array(Image.open(my_image).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T

    my_predicted_image = predict(image, my_label_y, parameters)

    def backward_propagation(parameters, cache, X, Y):
        m = X.shape[1]

        W1 = parameters['W1']
        W2 = parameters['W2']

        A1 = cache['A1']
        A2 = cache['A2']

        dZ2= A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}

        return grads 

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()

def printMenu():
    menu_data = {
        1: "Plot image",
        2: "Data pre-processing",
        3: "Train model 2 layers model",
        4: "Train model L layers model and plot",
        5: "Experiment L with one layers",
        6: "Experiment test",
        7: "Real test",
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
            plot_image()
        elif choice == 2:
            Exercise1()
        elif choice == 3:
            Exercise2()
        elif choice == 4:
            Exercise3()
        elif choice == 5:
            Exercise4()
        elif choice == 6:
            Exercise5()
        elif choice == 7:
            real_test()
        elif choice == 0:
            break