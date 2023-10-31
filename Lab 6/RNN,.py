import numpy as np
from rnn_utils import *

class RNN:
    def rnn_cell_forward_test():
        np.random.seed(1)
        xt = np.random.randn(3,10)
        a_prev = np.random.randn(5,10)
        Waa = np.random.randn(5,5)
        Wax = np.random.randn(5,3)
        Wya = np.random.randn(2,5)
        ba = np.random.randn(5,1)
        by = np.random.randn(2,1)
        parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

        a_next, yt_pred, cache = RNN.rnn_cell_forward(xt, a_prev, parameters)
        print("a_next[4] = ", a_next[4])
        print("a_next.shape = ", a_next.shape)
        print("yt_pred[1] =", yt_pred[1])
        print("yt_pred.shape = ", yt_pred.shape)



    def rnn_cell_forward(xt, a_prev, parameters):
        """
        Implements a single forward step of the RNN-cell as described in Figure (2)

        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            ba --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
        """
        
        # Retrieve parameters from "parameters"
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]
        
        # compute next activation state using the formula given above
        a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
        # compute output of the current cell using the formula given above
        yt_pred = softmax(np.dot(Wya, a_next) + by)
        
        # store values you need for backward propagation in cache
        cache = (a_next, a_prev, xt, parameters)
        
        return a_next, yt_pred, cache
    
    def lstm_cell_forward_test():
        np.random.seed(1)
        xt = np.random.randn(3,10)
        a_prev = np.random.randn(5,10)
        c_prev = np.random.randn(5,10)
        Wf = np.random.randn(5, 5+3)
        bf = np.random.randn(5,1)
        Wi = np.random.randn(5, 5+3)
        bi = np.random.randn(5,1)
        Wo = np.random.randn(5, 5+3)
        bo = np.random.randn(5,1)
        Wc = np.random.randn(5, 5+3)
        bc = np.random.randn(5,1)
        Wy = np.random.randn(2,5)
        by = np.random.randn(2,1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
        print("a_next[4] = ", a_next[4])
        print("a_next.shape = ", c_next.shape)
        print("c_next[2] = ", c_next[2])
        print("c_next.shape = ", c_next.shape)
        print("yt[1] =", yt[1])
        print("yt.shape = ", yt.shape)
        print("cache[1][3] =", cache[1][3])
        print("len(cache) = ", len(cache))

    def lstm_cell_forward(xt, a_prev, c_prev, parameters):
        """
        Implement a single forward step of the LSTM-cell as described in Figure (4)

        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                            
        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        c_next -- next memory state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
        
        Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
            c stands for the memory value
        """

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        bf = parameters["bf"]
        Wi = parameters["Wi"]
        bi = parameters["bi"]
        Wc = parameters["Wc"]
        bc = parameters["bc"]
        Wo = parameters["Wo"]
        bo = parameters["bo"]
        Wy = parameters["Wy"]
        by = parameters["by"]
        
        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # Concatenate a_prev and xt (≈3 lines)
        concat = np.zeros((n_a + n_x, m))
        concat[: n_a, :] = a_prev
        concat[n_a :, :] = xt

        # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
        ft = sigmoid(np.dot(Wf, concat) + bf)
        it = sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = ft * c_prev + it * cct
        ot = sigmoid(np.dot(Wo, concat) + bo)
        a_next = ot * np.tanh(c_next)
        
        # Compute prediction of the LSTM cell (≈1 line)
        yt_pred = softmax(np.dot(Wy, a_next) + by)

        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

        return a_next, c_next, yt_pred, cache
    
    def lstm_forward_test():
        np.random.seed(1)
        x = np.random.randn(3,10,7)
        a0 = np.random.randn(5,10)
        Wf = np.random.randn(5, 5+3)
        bf = np.random.randn(5,1)
        Wi = np.random.randn(5, 5+3)
        bi = np.random.randn(5,1)
        Wo = np.random.randn(5, 5+3)
        bo = np.random.randn(5,1)
        Wc = np.random.randn(5, 5+3)
        bc = np.random.randn(5,1)
        Wy = np.random.randn(2,5)
        by = np.random.randn(2,1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

        a, y, c, caches = RNN.lstm_forward(x, a0, parameters)
        print("a[4][3][6] = ", a[4][3][6])
        print("a.shape = ", a.shape)
        print("y[1][4][3] =", y[1][4][3])
        print("y.shape = ", y.shape)
        print("caches[1][1[1]] =", caches[1][1][1])
        print("c[1][2][1]", c[1][2][1])
        print("len(caches) = ", len(caches))

    def lstm_forward(x, a0, parameters):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                            
        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
        """

        # Initialize "caches", which will track the list of all the caches
        caches = []
        
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wy"].shape
        
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))
        
        a_next = a0
        c_next = np.zeros(a_next.shape)
        
        # loop over all time-steps
        for t in range(T_x):
            a_next, c_next, yt, cache = RNN.lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
            a[:,:,t] = a_next
            y[:,:,t] = yt
            c[:,:,t]  = c_next
            caches.append(cache)
            
        # store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches
    
    def rnn_cell_backward_test():
        np.random.seed(1)
        xt = np.random.randn(3,10)
        a_prev = np.random.randn(5,10)
        Wax = np.random.randn(5,3)
        Waa = np.random.randn(5,5)
        Wya = np.random.randn(2,5)
        b = np.random.randn(5,1)
        by = np.random.randn(2,1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

        a_next, yt, cache = RNN.rnn_cell_forward(xt, a_prev, parameters)

        da_next = np.random.randn(5,10)
        gradients = RNN.rnn_cell_backward(da_next, cache)
        print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
        print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
        print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
        print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
        print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
        print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
        print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
        print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
        print("gradients[\"dba\"][4] =", gradients["dba"][4])
        print("gradients[\"dba\"].shape =", gradients["dba"].shape)


    def rnn_cell_backward(da_next, cache):
        """
        Implements the backward pass for the RNN-cell (single time-step).

        Arguments:
        da_next -- Gradient of loss with respect to next hidden state
        cache -- python dictionary containing useful values (output of rnn_cell_forward())

        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradients of input data, of shape (n_x, m)
                            da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dba -- Gradients of bias vector, of shape (n_a, 1)
        """
        
        # Retrieve values from cache
        (a_next, a_prev, xt, parameters) = cache
        
        # Retrieve values from parameters
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        dtanh = (1 - a_next ** 2) * da_next
        dxt = np.dot(Wax.T, dtanh) 
        dWax = np.dot(dtanh, xt.T)
        da_prev = np.dot(Waa.T, dtanh)
        dWaa = np.dot(dtanh, a_prev.T)
        dba = np.sum(dtanh, axis = 1,keepdims=1)

        # Store the gradients in a python dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
        
        return gradients

    def rnn_backward_test():
        np.random.seed(1)
        x = np.random.randn(3,10,4)
        a0 = np.random.randn(5,10)
        Wax = np.random.randn(5,3)
        Waa = np.random.randn(5,5)
        Wya = np.random.randn(2,5)
        ba = np.random.randn(5,1)
        by = np.random.randn(2,1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
        a, y, caches = RNN.rnn_forward(x, a0, parameters)
        da = np.random.randn(5, 10, 4)
        gradients = RNN.rnn_backward(da, caches)

        print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
        print("gradients[\"dx\"].shape =", gradients["dx"].shape)
        print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
        print("gradients[\"da0\"].shape =", gradients["da0"].shape)
        print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
        print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
        print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
        print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
        print("gradients[\"dba\"][4] =", gradients["dba"][4])
        print("gradients[\"dba\"].shape =", gradients["dba"].shape)

    def rnn_backward(da, caches):
        """
        Implement the backward pass for a RNN over an entire sequence of input data.

        Arguments:
        da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
        caches -- tuple containing information from the forward pass (rnn_forward)
        
        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                            da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                            dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                            dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                            dba -- Gradient w.r.t the bias, of shape (n_a, 1)
        """
        
        (caches, x) = caches
        (a1, a0, x1, parameters) = caches[0]
        
        n_a, m, T_x = da.shape
        n_x, m = x1.shape
        
        dx = np.zeros((n_x, m, T_x))
        dWax = np.zeros((n_a, n_x))
        dWaa = np.zeros((n_a, n_a))
        dba = np.zeros((n_a, 1))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))
            
        
        # Loop through all the time steps
        for t in reversed(range(T_x)):
            # Compute gradients at time step t. Choose wisely the "da_next" and the "cache" to use in the backward propagation step. (≈1 line)
            gradients = RNN.rnn_cell_backward(da[:,:,t] + da_prevt, caches[t])
            # Retrieve derivatives from gradients (≈ 1 line)
            dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
            # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
            dx[:, :, t] = dxt
            dWax += dWaxt
            dWaa += dWaat
            dba += dbat
            
        # Set da0 to the gradient of a which has been backpropagated through all time-steps (≈1 line) 
        da0 = da_prevt

        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
        
        return gradients

    def lstm_cell_backward_test():
        np.random.seed(1)
        xt = np.random.randn(3,10)
        a_prev = np.random.randn(5,10)
        c_prev = np.random.randn(5,10)
        Wf = np.random.randn(5, 5+3)
        bf = np.random.randn(5,1)
        Wi = np.random.randn(5, 5+3)
        bi = np.random.randn(5,1)
        Wo = np.random.randn(5, 5+3)
        bo = np.random.randn(5,1)
        Wc = np.random.randn(5, 5+3)
        bc = np.random.randn(5,1)
        Wy = np.random.randn(2,5)
        by = np.random.randn(2,1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

        da_next = np.random.randn(5,10)
        dc_next = np.random.randn(5,10)
        gradients = lstm_cell_backward(da_next, dc_next, cache)
        print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
        print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
        print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
        print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
        print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
        print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
        print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
        print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
        print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
        print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
        print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
        print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
        print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
        print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
        print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
        print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
        print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
        print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
        print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
        print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
        print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
        print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

    def lstm_cell_backward(da_next, dc_next, cache):
        """
        Implement the backward pass for the LSTM-cell (single time-step).

        Arguments:
        da_next -- Gradients of next hidden state, of shape (n_a, m)
        dc_next -- Gradients of next cell state, of shape (n_a, m)
        cache -- cache storing information from the forward pass

        Returns:
        gradients -- python dictionary containing:
                            dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                            da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                            dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                            dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                            dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
        """

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
        
        ### START CODE HERE ###
        # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
        n_x, m = xt.shape
        n_a, m = a_next.shape
        
        # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)
        dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
        dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)

        # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
        
        dWf = np.dot(dft, np.hstack([a_prev.T, xt.T]))
        dWi = np.dot(dit, np.hstack([a_prev.T, xt.T]))
        dWc = np.dot(dcct, np.hstack([a_prev.T, xt.T]))
        dWo = np.dot(dot, np.hstack([a_prev.T, xt.T]))
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)    
        da_prev = np.dot(Wf[:, :n_a].T, dft) + np.dot(Wc[:, :n_a].T, dcct) + np.dot(Wi[:, :n_a].T, dit) + np.dot(Wo[:, :n_a].T, dot)
        dc_prev = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * ft
        dxt = np.dot(Wf[:, n_a:].T, dft) + np.dot(Wc[:, n_a:].T, dcct) + np.dot(Wi[:, n_a:].T, dit) + np.dot(Wo[:, n_a:].T, dot)
        ### END CODE HERE ###
        
        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

        return gradients
    def lstm_backward_test():
        np.random.seed(1)
        x = np.random.randn(3,10,7)
        a0 = np.random.randn(5,10)
        Wf = np.random.randn(5, 5+3)
        bf = np.random.randn(5,1)
        Wi = np.random.randn(5, 5+3)
        bi = np.random.randn(5,1)
        Wo = np.random.randn(5, 5+3)
        bo = np.random.randn(5,1)
        Wc = np.random.randn(5, 5+3)
        bc = np.random.randn(5,1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

        a, y, c, caches = lstm_forward(x, a0, parameters)

        da = np.random.randn(5, 10, 4)
        gradients = lstm_backward(da, caches)

        print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
        print("gradients[\"dx\"].shape =", gradients["dx"].shape)
        print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
        print("gradients[\"da0\"].shape =", gradients["da0"].shape)
        print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
        print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
        print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
        print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
        print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
        print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
        print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
        print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
        print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
        print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
        print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
        print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
        print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
        print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
        print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
        print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

    def lstm_backward(da, caches):
        
        """
        Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

        Arguments:
        da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
        dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
        caches -- cache storing information from the forward pass (lstm_forward)

        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradient of inputs, of shape (n_x, m, T_x)
                            da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                            dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                            dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
        """

        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
        
        # Retrieve dimensions from da's and x1's shapes (≈2 lines)
        n_a, m, T_x = da.shape
        n_x, m = x1.shape
        
        # initialize the gradients with the right sizes (≈12 lines)
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))
        dc_prevt = np.zeros((n_a, m))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))
        
        # loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = lstm_cell_backward(da[:,:,t] + da_prevt, dc_prevt, caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            dx[:,:,t] = gradients["dxt"]
            dWf += gradients["dWf"]
            dWi += gradients["dWi"]
            dWc += gradients["dWc"]
            dWo += gradients["dWo"]
            dbf += gradients["dbf"]
            dbi += gradients["dbi"]
            dbc += gradients["dbc"]
            dbo += gradients["dbo"]
        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients["da_prev"]
        
        ### END CODE HERE ###
        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
        
        return gradients
    
def main():
    return

if __name__ == '__main__':
    main()