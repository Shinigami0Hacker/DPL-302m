# print("Loading Environment")
import os

import keras
import numpy as np
from keras.layers import Input, Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pandas import read_pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '0' for all messages

from preprocessing import text_preprocess


class TextModule:
    """
    Important:
        Set max_length=65 and num_tokens=28815 when call this class. This is the default hyperparameter of the model
        which is trained

        Ensure that call two methods: `load_model` and `load_word_docs` before calling the `prediction` method
    """
    def __init__(self, max_length: int, num_tokens: int):
        """
        :param max_length: The maximun length of the input sentence
        :param num_tokens: The tokens of the vocabulary dictionary
        """
        self.model = None
        self.max_length = max_length
        self.num_tokens = num_tokens
        self.tokenizer = Tokenizer()

    def load_model(self, path: str) -> bool:
        """
        Load a pre-trained model from a file and set it as the current model.

        This function loads a pre-trained model from a specified file and sets it as the current model
        for sentiment analysis. The function initializes an Embedding layer with pre-trained word embeddings
        and then constructs a Bidirectional LSTM and Bidirectional GRU neural network. It takes a file path
        as input and attempts to load the model from that path. If successful, it sets the loaded model as
        the current model and returns True. If any exceptions occur during this process, an error message
        is printed to the console, and the function returns False.

        :param path: The file path to the pre-trained model's weights.
        :return: True if the model is successfully loaded, False if there is an error.
        """
        try:
            embedding_matrix_fasttext = np.load("Pretrained//embedding_matrix_fasttext.npy")
            embedding_dim = 300
            embedding_layer_fasttext = Embedding(
                input_dim=self.num_tokens,
                output_dim=embedding_dim,
                input_length=self.max_length,
                embeddings_initializer=keras.initializers.Constant(embedding_matrix_fasttext),
                trainable=False
            )

            activation = 'tanh'

            inputs = Input(shape=(self.max_length,))
            embedding = embedding_layer_fasttext(inputs)

            lstm_feats1 = Bidirectional(LSTM(16, return_sequences=True, activation=activation))(embedding)
            lstm_drop = Dropout(0.5)(lstm_feats1)
            lstm_feats2 = Bidirectional(LSTM(16, activation=activation))(lstm_drop)

            gru_feats1 = Bidirectional(GRU(16, return_sequences=True, activation=activation))(embedding)
            gru_drop = Dropout(0.5)(gru_feats1)
            gru_feats2 = Bidirectional(GRU(16, activation=activation))(gru_drop)

            merged = concatenate([lstm_feats2, gru_feats2])
            dens1 = Dense(8, activation='relu')(merged)
            drop1 = Dropout(0.6)(dens1)
            predict = Dense(2, activation='softmax')(drop1)
            model = Model(inputs, predict)
            self.model = model
            self.model.load_weights(path)
            return True
        except Exception as e:
            print(str(e))
            return False

    def load_word_docs(self, target_path):
        """
        Load and fit the Tokenizer on a dataset for word frequency analysis.

        This function attempts to load a dataset from the specified `target_path`, and then fits the
        Tokenizer on this dataset to analyze the frequency of words. The loaded dataset is expected to
        be in a pickle format. If the operation is successful, the function returns True, otherwise, it
        returns False. In case of an error, an exception message is printed to the console for further
        debugging.

        :param target_path: The path to the dataset in pickle format.
        :return: True if the operation is successful, False if there is an error.
        """
        try:
            Xtrain = read_pickle(target_path)
            self.tokenizer.fit_on_texts(Xtrain)
            return True
        except Exception as e:
            print(str(e))
            return False

    def prediction(self,
                   input_sentences: list[str],
                   is_print: bool = False,
                   preprocess: bool = False) -> dict[str, dict[str, str]]:
        """
        Predict sentiment scores for a list of input sentences.

        :param input_sentences: A list of input sentences to predict sentiment for.
        :type input_sentences: list of str
        :param is_print: If True, the results will be printed to the console. (Default: False)
        :type is_print: bool
        :param preprocess: If True, input sentences will be preprocessed. (Default: False)
        :type preprocess: bool
        :return: A dictionary with input sentences as keys and their respective sentiment scores as values.
        :rtype: dict of str, dict of str, str

        Example:
        >> input_sentences = ["Its ok but not greate"]
        >> sentiment_scores = prediction(input_sentences, is_print=True, preprocess=False)
        =======================================
        Input: Its ok but not greate
        Positive score: 34.36%
        Negative score: 65.64%
        Overall: Neutral
        =======================================
        """
        results = {}

        for input_sentence in input_sentences:
            # preprocess and vectorize
            if preprocess:
                input_sentence = text_preprocess(input_sentence)
            seq = self.tokenizer.texts_to_sequences([input_sentence])
            x = pad_sequences(sequences=seq, maxlen=self.max_length, padding='post')
            # call model to predict
            predictions = self.model.predict(x, verbose=0)
            # Get probability of positive
            positive_probabilities = predictions[:, 1]
            # convert to percentage
            positive_percentage = positive_probabilities * 100
            negative_percentage = (1 - positive_probabilities) * 100

            positive_score = positive_percentage[0]
            negative_score = negative_percentage[0]

            results[input_sentence] = (positive_score, negative_score)

            if positive_score >= 70:  # Positive: 70 ~ 100
                decision = "Positive"
            elif positive_score >= 30:  # Neutral: 30 ~ 69
                decision = "Neutral"
            else:
                decision = "Negative"  # Negative: 0 ~ 29

            results[input_sentence] = {"positive": positive_score,
                                       "negative": negative_score,
                                       "overall": decision}
        print("=======================================")
        if is_print:
            for sent in results:
                print(f"Input: {sent}")
                print(f"Positive score: {results[sent]['positive']:.2f}%")
                print(f"Negative score: {results[sent]['negative']:.2f}%")
                print(f"Overall: {results[sent]['overall']}")
                print("=======================================")

        return results


if __name__ == '__main__':
    # print("Loading Application")
    text_module = TextModule(max_length=65, num_tokens=28815)
    text_module.load_model(path="Model_checkpoint//sentiment_analysis_weights(2023-11-01).h5")
    text_module.load_word_docs(target_path="Preprocessed//Xtrain.pkl")
    # print("Application loaded successful")
    while True:
        input_text = input("[ENTER 0 TO EXIT] Text to sentiment: ")
        if input_text == "0":
            print("Application interrupted")
            print("Thanks for using the application!")
            break
        predicted = text_module.prediction(input_sentences=[input_text], is_print=True)
