import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from helpers import pad_sequences, plot_history, save, saveEmbedding

Sequential = keras.models.Sequential
Embedding = keras.layers.Embedding
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
Dense = keras.layers.Dense

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                          split=(tfds.Split.TRAIN,
                                                 tfds.Split.TEST),
                                          with_info=True,
                                          as_supervised=True)

encoder = info.features['text'].encoder
train_batches = pad_sequences(train_data)
test_batches = pad_sequences(test_data)


embedding_dim = 16

model = Sequential([
    Embedding(encoder.vocab_size, embedding_dim),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_batches,
          epochs=10,
          validation_data=test_batches,
          validation_steps=20)

save(model)
saveEmbedding(model.layers[0], info.features['text'].encoder)
plot_history(model.history.history)
