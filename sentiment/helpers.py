import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

imdb = keras.datasets.imdb


class Word_Index():
    def __init__(self):
        self.word_dict = imdb.get_word_index()
        self.reverse_word_dict = dict(
            [(value, key) for (key, value) in self.word_dict.items()])

    def get_word(self, word_index, default="<unk>"):
        return self.reverse_word_dict.get(word_index, default)

    def get_index(self, word, default=0):
        return self.word_dict.get(word, default)


def one_hot(sequences, vocab_size=10000):
    """One Hot encode a sequence of words
    ---
    Expect to final result to be a list of 0 or 1 
    that represent the entire word sequence 
    with no spacial relationship

    sequences : two dimentional numpy array
        The list of word sequences
    vocab_size : int, optional
        How many possible words in dataset
    """

    # Create an all-zero matrix of shape (len(sequences), vocab_size)
    results = np.zeros((len(sequences), vocab_size))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


def save(model, filename='sentiment/saved_models/model.h5'):
    model.save(filename)


def load(filename='sentiment/saved_models/model.h5'):
    return tf.keras.models.load_model(filename)


def plot_history(history, filename='sentiment/history.png'):
    plt.close('all')
    fig = plt.figure()

    loss = fig.add_subplot(121)
    loss.set_title('loss')
    loss.plot(history['loss'], '.', label="training loss")
    loss.plot(history['val_loss'], 'b', label="calidation loss")

    accuracy = fig.add_subplot(122)
    accuracy.set_title('accuracy')
    accuracy.plot(history['binary_accuracy'], '.', label="training accuracy")
    accuracy.plot(history['val_binary_accuracy'],
                  'b', label="validation accuracy")

    plt.tight_layout()
    fig.savefig(filename)
