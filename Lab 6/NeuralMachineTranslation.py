from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import tensorflow as tf
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

from test_utils import *

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)


class NeuralMachineTranslation:
    
    @staticmethod
    def one_step_attention_test(target):
        m = 10
        Tx = 30
        n_a = 32
        n_s = 64
        #np.random.seed(10)
        a = np.random.uniform(1, 0, (m, Tx, 2 * n_a)).astype(np.float32)
        s_prev =np.random.uniform(1, 0, (m, n_s)).astype(np.float32) * 1
        context = target(a, s_prev)
        
        assert type(context) == tf.python.framework.ops.EagerTensor, "Unexpected type. It should be a Tensor"
        assert tuple(context.shape) == (m, 1, n_s), "Unexpected output shape"
        assert np.all(context.numpy() > 0), "All output values must be > 0 in this example"
        assert np.all(context.numpy() < 1), "All output values must be < 1 in this example"

        #assert np.allclose(context[0][0][0:5].numpy(), [0.50877404, 0.57160693, 0.45448175, 0.50074816, 0.53651875]), "Unexpected values in the result"
        print("\033[92mAll tests passed!")
        
    @staticmethod
    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        
        s_prev = repeator(s_prev)
        concat = concatenator([a,s_prev])
        e = densor1(concat)
        energies = densor2(e)
        alphas = activator(energies)
        context = dotor([alphas,a])
        
        return context
    @staticmethod
    def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """
        
        # Define the inputs of your model with a shape (Tx,)
        # Define s0 (initial hidden state) and c0 (initial cell state)
        # for the decoder LSTM with shape (n_s,)
        X = Input(shape=(Tx, human_vocab_size))
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0
        
        # Initialize empty list of outputs
        outputs = []
        
        a = Bidirectional(LSTM(n_a, return_sequences=True))(X)    
        
        # Step 2: Iterate for Ty steps
        for t in range(Ty):
        
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = one_step_attention(a, s)
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = post_activation_LSTM_cell(context, initial_state=[s,c])
            
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = output_layer(s)
            
            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append( out )
        
        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model( inputs=[X,s0,c0], outputs=outputs )
        
        
        return model

    def modelf_test(target):
        Tx = 30
        n_a = 32
        n_s = 64
        len_human_vocab = 37
        len_machine_vocab = 11
        
        
        model = target(Tx, Ty, n_a, n_s, len_human_vocab, len_machine_vocab)
        
        print(summary(model))

        expected_summary = [['InputLayer', [(None, 30, 37)], 0],
                            ['InputLayer', [(None, 64)], 0],
                            ['Bidirectional', (None, 30, 64), 17920],
                            ['RepeatVector', (None, 30, 64), 0, 30],
                            ['Concatenate', (None, 30, 128), 0],
                            ['Dense', (None, 30, 10), 1290, 'tanh'],
                            ['Dense', (None, 30, 1), 11, 'relu'],
                            ['Activation', (None, 30, 1), 0],
                            ['Dot', (None, 1, 64), 0],
                            ['InputLayer', [(None, 64)], 0],
                            ['LSTM',[(None, 64), (None, 64), (None, 64)], 33024,[(None, 1, 64), (None, 64), (None, 64)],'tanh'],
                            ['Dense', (None, 11), 715, 'softmax']]

        assert len(model.outputs) == 10, f"Wrong output shape. Expected 10 != {len(model.outputs)}"

        comparator(summary(model), expected_summary)
        

def final_test():
    model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    print(model.summary())
    opt = Adam(lr=.005, beta_1=.9, beta_2=.999, decay=.01)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    assert opt.lr == 0.005, "Set the lr parameter to 0.005"
    assert opt.beta_1 == 0.9, "Set the beta_1 parameter to 0.9"
    assert opt.beta_2 == 0.999, "Set the beta_2 parameter to 0.999"
    assert opt.decay == 0.01, "Set the decay parameter to 0.01"
    assert model.loss == "categorical_crossentropy", "Wrong loss. Use 'categorical_crossentropy'"
    assert model.optimizer == opt, "Use the optimizer that you have instantiated"
    assert model.compiled_metrics._user_metrics[0] == 'accuracy', "set metrics to ['accuracy']"

    print("\033[92mAll tests passed!")

if __name__ == '__main__':
    NeuralMachineTranslation.one_step_attention_test()
    NeuralMachineTranslation.modelf_test()
    final_test()