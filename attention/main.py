import tensorflow as tf
from tensorflow import keras

from helpers import overlay_attn

K = keras.backend
VGG16 = keras.applications.VGG16
image = keras.preprocessing.image
preprocess_input = keras.applications.vgg16.preprocess_input
decode_predictions = keras.applications.vgg16.decode_predictions

model = VGG16(weights='imagenet')

img_path = './attention/goose.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = tf.reshape(x, shape=(1, 224, 224, 3))
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

pred_neuron = K.argmax(preds[0])

overlay_attn(model, x, img_path)