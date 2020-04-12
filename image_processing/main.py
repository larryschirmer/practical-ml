import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from helpers import normalize, save, plot_pred_bars, load, plot_number, plot_history, mnistTrainingCallback

Sequential = keras.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten

np.set_printoptions(linewidth=500)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_formatted = normalize(x_train, 255.0)
x_test_formatted = normalize(x_test, 255.0)

model = Sequential(name='mnist')
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train_formatted,
          y_train,
          validation_data=(x_test_formatted, y_test),
          callbacks=[mnistTrainingCallback(0.98)],
          epochs=100,
          shuffle=True)

save(model)
plot_history(model.history.history)
# model = load()

test_img_index = 6
pred = model.predict([[x_test_formatted[test_img_index]]])

plot_pred_bars(pred[0])
plot_number(x_test[test_img_index])
