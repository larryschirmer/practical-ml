from helpers import Word_Index, one_hot, save, plot_history
import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
optimizers = keras.optimizers
losses = keras.losses
metrics = keras.metrics

num_words = 10000

word_index = Word_Index()

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=num_words)

decoded_review = ' '.join([word_index.get_word(encoded_word - 3, '?')
                           for encoded_word in train_data[0]])


x_train = one_hot(train_data, num_words)
y_train = np.asarray(train_labels).astype('float32')
x_test = one_hot(test_data, num_words)
y_test = np.asarray(test_labels).astype('float32')


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(num_words,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


partial_x_train = x_train[10000:]
x_val = x_train[:10000]

partial_y_train = y_train[10000:]
y_val = y_train[:10000]

model.fit(partial_x_train,
          partial_y_train,
          epochs=10,
          batch_size=512,
          validation_data=(x_val, y_val))

save(model)
plot_history(model.history.history)
