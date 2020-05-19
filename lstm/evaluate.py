import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from helpers import load, loadVocab

model = load()
vocab_set = loadVocab()

encoder = tfds.features.text.TokenTextEncoder(vocab_set)

example_text = b'to be or not to be'
encoded_example = encoder.encode(example_text)
print(encoded_example)

prediction = model.predict([encoded_example, ])
print(prediction)