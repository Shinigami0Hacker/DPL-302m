import tensorflow as tf
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization
from transformers import DistilBertTokenizerFast #, TFDistilBertModel
from transformers import TFDistilBertForTokenClassification
from tqdm import tqdm_notebook as tqdm

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
])

class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network. 
    This archirecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)

        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            out2 -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        """
        # START CODE HERE
        # calculate self-attention using mha(~1 line)
        attn_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, fully_connected_dim)
        
        # apply dropout layer to the self-attention output (~1 line)
        attn_output = self.dropout1(attn_output, training=training)
        
        # apply layer normalization on sum of the input and the attention output to get the  
        # output of the multi-head attention layer (~1 line)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, fully_connected_dim)

        # pass the output of the multi-head attention layer through a ffn (~1 line)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, fully_connected_dim)
        
        # apply dropout layer to ffn output (~1 line)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        # output of the encoder layer (~1 line)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, fully_connected_dim)
        # END CODE HERE
        
        return out2
def EncoderLayer_test(target):
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    encoder_layer1 = EncoderLayer(4, 2, 8)
    tf.random.set_seed(10)
    encoded = encoder_layer1(q, True, np.array([[1, 0, 1]]))
    
    assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoded).numpy()) == (1, q.shape[1], q.shape[2]), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"

    assert np.allclose(encoded.numpy(), 
                       [[-0.5214877 , -1.001476  , -0.12321664,  1.6461804 ],
                       [-1.3114998 ,  1.2167752 , -0.5830886 ,  0.6778133 ],
                       [ 0.25485858,  0.3776546 , -1.6564771 ,  1.023964  ]],), "Wrong values"
    
    print("\033[92mAll tests passed")
    

class Transformer:
    def get_angles_test(target):
        position = 4
        d_model = 16
        pos_m = np.arange(position)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]

        result = target(pos_m, dims, d_model)

        assert type(result) == np.ndarray, "You must return a numpy ndarray"
        assert result.shape == (position, d_model), f"Wrong shape. We expected: ({position}, {d_model})"
        assert np.sum(result[0, :]) == 0
        assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
        even_cols =  result[:, 0::2]
        odd_cols = result[:,  1::2]
        assert np.all(even_cols == odd_cols), "Submatrices of odd and even columns must be equal"
        limit = (position - 1) / np.power(10000,14.0/16.0)
        assert np.isclose(result[position - 1, d_model -1], limit ), f"Last value must be {limit}"
        print("\033[92mAll tests passed")

    def get_angles(pos, i, d):
        """
        Get the angles for the positional encoding
        
        Arguments:
            pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
            i --   Row vector containing the dimension span [[0, 1, 2, ..., M-1]]
            d(integer) -- Encoding size
        
        Returns:
            angles -- (pos, d) numpy array 
        """
        # STATR CODE HERE
        angles = pos / np.power(10000, (2*(i//2)) / d)
        # END CODE HERE
        
        return angles

    def positional_encoding_test(target):
        position = 8
        d_model = 16

        pos_encoding = target(position, d_model)
        sin_part = pos_encoding[:, :, 0::2]
        cos_part = pos_encoding[:, :, 1::2]

        assert tf.is_tensor(pos_encoding), "Output is not a tensor"
        assert pos_encoding.shape == (1, position, d_model), f"Wrong shape. We expected: (1, {position}, {d_model})"

        ones = sin_part ** 2  +  cos_part ** 2
        assert np.allclose(ones, np.ones((1, position, d_model // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"
        
        angs = np.arctan(sin_part / cos_part)
        angs[angs < 0] += np.pi
        angs[sin_part.numpy() < 0] += np.pi
        angs = angs % (2 * np.pi)
        
        pos_m = np.arange(position)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]

        trueAngs = Transformer.get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)
        
        assert np.allclose(angs[0], trueAngs), "Did you apply sin and cos to even and odd parts respectively?"
    
        print("\033[92mAll tests passed")

    def positional_encoding(positions, d):
        """
        Precomputes a matrix with all the positional encodings 
        
        Arguments:
            positions (int) -- Maximum number of positions to be encoded 
            d (int) -- Encoding size 
        
        Returns:
            pos_encoding -- (1, position, d_model) A matrix with the positional encodings
        """
        # START CODE HERE
        # initialize a matrix angle_rads of all the angles 
        
        posm2 = np.arange(positions)[:, np.newaxis]
        dims2 = np.arange(d)[np.newaxis, :]
        angle_rads = Transformer.get_angles(posm2, dims2, d)
            
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        # END CODE HERE
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)

    def create_padding_mask(seq):
        """
        Creates a matrix mask for the padding cells
        
        Arguments:
            seq -- (n, m) matrix
        
        Returns:
            mask -- (n, 1, 1, m) binary tensor
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(size):
        """
        Returns an upper triangular matrix filled with ones
        
        Arguments:
            size -- matrix size
        
        Returns:
            mask -- (size, size) tensor
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def scaled_dot_product_attention_test(target):
        q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
        k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
        v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

        attention, weights = target(q, k, v, None)
        assert tf.is_tensor(weights), "Weights must be a tensor"
        assert tuple(tf.shape(weights).numpy()) == (q.shape[0], k.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
        assert np.allclose(weights, [[0.2589478,  0.42693272, 0.15705977, 0.15705977],
                                    [0.2772748,  0.2772748,  0.2772748,  0.16817567],
                                    [0.33620113, 0.33620113, 0.12368149, 0.2039163 ]])

        assert tf.is_tensor(attention), "Output must be a tensor"
        assert tuple(tf.shape(attention).numpy()) == (q.shape[0], v.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
        assert np.allclose(attention, [[0.74105227, 0.15705977],
                                    [0.7227253,  0.16817567],
                                    [0.6637989,  0.2039163 ]])

        mask = np.array([0, 0, 1, 0])
        attention, weights = target(q, k, v, mask)

        assert np.allclose(weights, [[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                    [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                    [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862]]), "Wrong masked weights"
        assert np.allclose(attention, [[0.6928040981292725, 0.18632373213768005],
                                      [0.6163482666015625, 0.2326965481042862], 
                                      [0.6163482666015625, 0.2326965481042862]]), "Wrong masked attention"
        
        print("\033[92mAll tests passed")

    def scaled_dot_product_attention(q, k, v, mask):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Arguments:
            q -- query shape == (..., seq_len_q, depth)
            k -- key shape == (..., seq_len_k, depth)
            v -- value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            output -- attention_weights
        """
        # Q*K'
        matmul_qk = tf.matmul(q, k.T)
        
        # scale matmul_qk
        #dk = len(k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1.0e9)

        attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        print('attention_weights: ', attention_weights)
        # attention_weights * V
        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
        print('output: ', output)

        return output, attention_weights

class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
        
    """   
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = Transformer.positional_encoding(maximum_position_encoding, 
                                                self.embedding_dim)


        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            out2 -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
        """

        seq_len = tf.shape(x)[1]
        
        # START CODE HERE
        # Pass input through the Embedding layer
        x = self.embedding(x)  # (batch_size, input_seq_len, fully_connected_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        x = self.dropout(x, training = training)
        # Pass the output through the stack of encoding layers 
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        # END CODE HERE

        return x  # (batch_size, input_seq_len, fully_connected_dim)

def Encoder_test(target):
    tf.random.set_seed(10)
    
    embedding_dim=4
    
    encoderq = target(num_layers=2,
                      embedding_dim=embedding_dim,
                      num_heads=2,
                      fully_connected_dim=8,
                      input_vocab_size=32,
                      maximum_position_encoding=5)
    
    x = np.array([[2, 1, 3], [1, 2, 0]])
    
    encoderq_output = encoderq(x, True, None)
    
    assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoderq_output).numpy()) == (x.shape[0], x.shape[1], embedding_dim), f"Wrong shape. We expected ({eshape[0]}, {eshape[1]}, {embedding_dim})"
    assert np.allclose(encoderq_output.numpy(), 
                       [[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                         [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                         [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                        [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                         [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                         [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]]), "Wrong values"
    
    print("\033[92mAll tests passed")

class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)

        self.mha2 = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)

        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            enc_output --  Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            out3 -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attn_weights_block1 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
            attn_weights_block2 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
        """
        
        # enc_output.shape == (batch_size, input_seq_len, fully_connected_dim)
        
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1 (~1 line)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        
        # apply dropout layer on the attention output (~1 line)
        attn1 = self.dropout1(attn1, training=training)
        
        # apply layer normalization to the sum of the attention output and the input (~1 line)
        
        out1 = self.layernorm1(attn1 + x)

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # Return attention scores as attn_weights_block2 (~1 line)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        
        # apply dropout layer on the attention output (~1 line)
        attn2 = self.dropout2(attn2, training=training)
        
        # apply layer normalization to the sum of the attention output and the output of the first block (~1 line)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, fully_connected_dim)
        
        #BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, fully_connected_dim)
        
        # apply a dropout layer to the ffn output
        ffn_output = self.dropout3(ffn_output, training=training)
        
        # apply layer normalization to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, fully_connected_dim)

        return out3, attn_weights_block1, attn_weights_block2
    
def DecoderLayer_test(target):
    
    num_heads=8
    tf.random.set_seed(10)
    
    decoderLayerq = target(
        embedding_dim=4, 
        num_heads=num_heads,
        fully_connected_dim=32, 
        dropout_rate=0.1, 
        layernorm_eps=1e-6)
    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                                   [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                                   [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]]])
    
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    
    look_ahead_mask = tf.constant([[0., 1., 1.],
                       [0., 0., 1.],
                       [0., 0., 0.]])
    
    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, True, look_ahead_mask, padding_mask)
    
    assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"
    
    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert tuple(tf.shape(attn_w_b1).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(attn_w_b2).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(out).numpy()) == q.shape, f"Wrong shape. We expected {q.shape}"

    assert np.allclose(attn_w_b1[0, 0, 0], [0, 0.5, 0.5], atol=1e-2), "Wrong values in attn_w_b1. Check the call to self.mha1"
    #assert np.allclose(attn_w_b2[0, 0, 1], [0.34485385, 0.33230072, 0.32284543]),  "Wrong values in attn_w_b2. Check the call to self.mha2"
    #assert np.allclose(out[0, 0], [0.64775777, -1.5134472,   1.1092964,  -0.24360693]), "Wrong values in out"
    

    # Now let's try a example with padding mask
    padding_mask = np.array([[0, 0, 1]])
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, True, look_ahead_mask, padding_mask)

    assert np.allclose(out[0, 0], [0.59296525, -1.4068702, 1.224841, -0.410936]), "Wrong values in out when we mask the last word. Are you passing the padding_mask to the inner functions?"

    print("\033[92mAll tests passed")

class Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder is starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
        
    """ 
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = Transformer.positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            enc_output --  Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            x -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attention_weights - Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # create word embeddings 
        x = self.embedding(x)  # (batch_size, target_seq_len, fully_connected_dim)
        
        # scale embeddings by multiplying by the square root of their dimension
        x *= tf.sqrt(tf.cast(self.embedding_dim, tf.float32))
        
        # calculate positional encodings and add to word embedding
        x += self.pos_encoding[:, :seq_len, :]
        
        # apply a dropout layer to x
        x = self.dropout(x, training=training)
        
        # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights
            # of block 1 and 2 (~1 line)
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
            #update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
                       
        # x.shape == (batch_size, target_seq_len, fully_connected_dim)
        return x, attention_weights
    
    def Decoder_test(target):
    
        tf.random.set_seed(10)
            
        num_layers=7
        embedding_dim=4 
        num_heads=3
        fully_connected_dim=8
        target_vocab_size=33
        maximum_position_encoding=6
        
        x = np.array([[3, 2, 1], [2, 1, 0]])

        
        encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                            [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                            [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                            [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                            [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                            [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]])
        
        look_ahead_mask = tf.constant([[0., 1., 1.],
                        [0., 0., 1.],
                        [0., 0., 0.]])
        
        decoderk = Decoder(num_layers,
                        embedding_dim, 
                        num_heads, 
                        fully_connected_dim,
                        target_vocab_size,
                        maximum_position_encoding)
        outd, att_weights = decoderk(x, encoderq_output, False, look_ahead_mask, None)
        
        assert tf.is_tensor(outd), "Wrong type for outd. It must be a dict"
        assert np.allclose(tf.shape(outd), tf.shape(encoderq_output)), f"Wrong shape. We expected { tf.shape(encoderq_output)}"
        assert np.allclose(outd[1, 1], [-0.34560338, -0.8762897,  -0.4767484,   1.6986415]), "Wrong values in outd"
        
        keys = list(att_weights.keys())
        assert type(att_weights) == dict, "Wrong type for att_weights[0]. Output must be a tensor"
        assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
        assert tf.is_tensor(att_weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
        shape1 = (x.shape[0], num_heads, x.shape[1], x.shape[1])
        assert tuple(tf.shape(att_weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
        assert np.allclose(att_weights[keys[0]][0, 0, 1], [0., 0., 1.]), f"Wrong values in att_weights[{keys[0]}]"
        
        print(outd)
    
        print("\033[92mAll tests passed")
    

class TransformerModel(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
            target_vocab_size, max_positional_encoding_input,
            max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            fully_connected_dim=fully_connected_dim,
                            input_vocab_size=input_vocab_size,
                            maximum_position_encoding=max_positional_encoding_input,
                            dropout_rate=dropout_rate,
                            layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            fully_connected_dim=fully_connected_dim,
                            target_vocab_size=target_vocab_size, 
                            maximum_position_encoding=max_positional_encoding_target,
                            dropout_rate=dropout_rate,
                            layernorm_eps=layernorm_eps)

        self.final_layer = Dense(target_vocab_size, activation='softmax')

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass for the entire Transformer
        Arguments:
            inp -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            tar -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            final_output -- Describe me
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        
        """
        # START CODE HERE
        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(inp, training, enc_padding_mask) # (batch_size, inp_seq_len, fully_connected_dim)
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, fully_connected_dim)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # pass decoder output through a linear layer and softmax (~2 lines)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # START CODE HERE

        return final_output, attention_weights
            
def Transformer_test(target):
    
    tf.random.set_seed(10)


    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = TransformerModel(num_layers, 
                        embedding_dim, 
                        num_heads, 
                        fully_connected_dim, 
                        input_vocab_size, 
                        target_vocab_size, 
                        max_positional_encoding_input,
                        max_positional_encoding_target)
    # 0 is the padding value
    sentence_lang_a = np.array([[2, 1, 4, 3, 0]])
    sentence_lang_b = np.array([[3, 2, 1, 0, 0]])

    enc_padding_mask = np.array([[0, 0, 0, 0, 1]])
    dec_padding_mask = np.array([[0, 0, 0, 1, 1]])

    look_ahead_mask = Transformer.create_look_ahead_mask(sentence_lang_a.shape[1])

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        True,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask
    )
    
    
    assert tf.is_tensor(translation), "Wrong type for translation. Output must be a tensor"
    shape1 = (sentence_lang_a.shape[0], max_positional_encoding_input, target_vocab_size)
    assert tuple(tf.shape(translation).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
        
    assert np.allclose(translation[0, 0, 0:8],
                       [[0.02664841, 0.02223665, 0.01641649, 0.02405972,
                         0.04249557, 0.02241551, 0.01557002, 0.03742596]]), "Wrong values in outd"
    
    keys = list(weights.keys())
    assert type(weights) == dict, "Wrong type for weights. It must be a dict"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
   
    shape1 = (sentence_lang_a.shape[0], num_heads, sentence_lang_a.shape[1], sentence_lang_a.shape[1])
    assert tuple(tf.shape(weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
    assert np.allclose(weights[keys[0]][0, 0, 1], [0., 0., 0.31332517, 0.33401096, 0.35266393]), f"Wrong values in weights[{keys[0]}]"
    
    print(translation)
    
    print("\033[92mAll tests passed")

    
Transformer_test(Transformer)
def main():    
    position = 4
    d_model = 8
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    Transformer.get_angles(pos_m, dims, d_model)
    return

if __name__ == '__main__':
    main()