import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from helpers import normalize, save, plot_pred_bars, load, plot_number, plot_history

Sequential = keras.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten

np.set_printoptions(linewidth=500)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = normalize(x_train, 255.0)
x_test = normalize(x_test, 255.0)


# model = Sequential(name='mnist')
# model.add(Flatten(input_shape=(28, 28)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer=tf.optimizers.Adam(),
#               loss=tf.losses.sparse_categorical_crossentropy,
#               metrics=['accuracy'])

# model.fit(x_train,
#           y_train,
#           validation_data=(x_test, y_test),
#           epochs=10,
#           shuffle=True)

# save(model)
# plot_history(model.history.history)
model = load()

test_img_index = 5
pred = model.predict([[x_test[test_img_index]]])
print(pred)
plot_pred_bars(pred[0])
plot_number(x_test[test_img_index])
