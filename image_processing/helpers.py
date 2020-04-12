import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def normalize(data, divisor):
    data = data.reshape(len(data), 28, 28, 1)
    return data / divisor


def save(model, filename='image_processing/saved_models/process_mnist.h5'):
    model.save(filename)


def load(filename='image_processing/saved_models/process_mnist.h5'):
    return tf.keras.models.load_model(filename)


def plot_pred_bars(array, filename='image_processing/predicted_number.png'):
    plt.close('all')
    plt.bar(np.arange(len(array)), array)
    plt.savefig(filename)


def plot_number(data, filename='image_processing/mnist-img.png'):
    plt.close('all')
    plt.imshow(data)
    plt.savefig(filename)


def plot_history(history, filename='image_processing/history.png'):
    plt.close('all')
    fig = plt.figure()

    loss = fig.add_subplot(221)
    loss.set_title('loss')
    loss.plot(history['loss'])

    accuracy = fig.add_subplot(222)
    accuracy.set_title('accuracy')
    accuracy.plot(history['accuracy'])

    val_loss = fig.add_subplot(223)
    val_loss.set_title('validation loss')
    val_loss.plot(history['val_loss'])

    val_accuracy = fig.add_subplot(224)
    val_accuracy.set_title('validation accuracy')
    val_accuracy.plot(history['val_accuracy'])

    plt.tight_layout()
    fig.savefig(filename)


class mnistTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs={}):
        epoch_val_accuracy = logs.get('val_accuracy')

        if epoch_val_accuracy > self.target_accuracy:
            print("\n\nTarget accuracy achieved, '{}'\n".format(epoch_val_accuracy))
            self.model.stop_training = True

# model.summary()

# test = model.predict(tf.random.normal((1, 28, 28)))
