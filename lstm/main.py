import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from helpers import encodeDataset, generateTrainingSets
from helpers import makeLabledDatasets, processData, tokenizeDataset
from helpers import save, saveVocab, plotHistory

Sequential = tf.keras.Sequential
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Bidirectional = tf.keras.layers.Bidirectional
Dense = tf.keras.layers.Dense

file_names = ['cowper.txt', 'derby.txt', 'butler.txt', 'shakespeare.txt']
buffer_size = 50000
batch_size = 64
take_size = 5000
embedding_size = 64
lstm_size = 64

labeled_datasets = makeLabledDatasets(file_names)
labeled_datasets = processData(labeled_datasets, buffer_size)
vocab_size, vocab_set = tokenizeDataset(labeled_datasets)
encoded_data = encodeDataset(labeled_datasets, vocab_set)

opts = (take_size, buffer_size, batch_size)
train_data, test_data = generateTrainingSets(encoded_data, opts)


model = Sequential()
model.add(Embedding(vocab_size, embedding_size))
model.add(Bidirectional(LSTM(lstm_size)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=test_data)
save(model)
saveVocab(vocab_set)
plotHistory(model.history.history)
