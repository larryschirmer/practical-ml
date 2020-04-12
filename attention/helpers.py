import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

K = keras.backend
Model = keras.models.Model


def overlay_attn(model, x, img_path, filepath='./attention/attn.jpg'):
    # https://github.com/Vadikus/practicalDL/blob/master/01%20-%2005%20-%20Attention%20of%20ConvNet%20(VGG16).ipynb
    last_vgg_conv_layer = model.get_layer('block5_conv3')
    heatmap_model = Model([model.inputs], [last_vgg_conv_layer.output, model.output])

    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    # https://stackoverflow.com/questions/58322147/how-to-generate-cnn-heatmaps-using-built-in-keras-in-tf2-0-tf-keras
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(x)
        loss = predictions[:, K.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = K.maximum(heatmap, 0)
    max_heat = K.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    heatmap = tf.reshape(heatmap, shape=(heatmap.shape[1], heatmap.shape[2], 1))
    heatmap = heatmap.numpy()

    img = cv2.imread(img_path)
    img_shape = (img.shape[1], img.shape[0])
    heatmap = cv2.resize(heatmap, img_shape)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(filepath, superimposed_img)
