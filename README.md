# My own notes from "A Practical Guide to Machine Learning with TensorFlow 2.0 & Keras"

While taking the [Frontend Masters - Practical Machine Learning](https://frontendmasters.com/courses/practical-machine-learning/), I took my own notes and recoded the examples. Each section of the course is organized in its own folder

The project repository can be found here [https://github.com/Vadikus/practicalDL](https://github.com/Vadikus/practicalDL)

## Summary

#### Linear Regression

Estimate the values for `w` (slope) and `b` (bias) that best fit a linear dataset. After generating a set of noisy points along a known slope and bias, use the `GradientTape` TensorFlow method to estimate the values for `w` (slope) and `b` (bias).

This project reinforces the concepts behind the fitting process that TensorFlow uses.

<img src="https://github.com/larryschirmer/practical-ml/raw/master/assets/linear_regression.gif" alt="insert linear regression gif" width="600"/>

#### MNIST Classification

Predict the value of a written number in an image. Train a convolutional model using Keras' own mnist dataset to correctly output the value of a written number as an integer.

<img src="https://github.com/larryschirmer/practical-ml/raw/master/assets/mnist_example.png" alt="mnist example" width="600"/>

#### CNNs and Attention

Overlay onto the input image the parts where the model focused on to generate its output. Import the trained VGG16 model from Keras and generate a list of predictions for what is in the image. Then overlay the last convolutional layer on top of the image as a heat map.

<img src="https://github.com/larryschirmer/practical-ml/raw/master/assets/attention_example.png" alt="attention example" width="600"/>

#### Sentiment using One Hot Encoding

Predict the sentiment of a movie review using a one hot encoding of the Keras imdb dataset. Train a new model to predict either positive or negative classification from a binary two dimensional.

```python
review = "best movie best actors high praise"
tokenized_review = tokenizer.tokenize(review)
encoded_review = [word_index.get_index(word) for word in tokenized_review]
one_hot_review = one_hot([encoded_review])

prediction = model.predict(one_hot_review)
print(prediction) # [[0.592]]
```

#### Embedding and PCA Visualization

Train an embedding layer using imdb reviews from TensorFlow Datasets. Then visualize the inspect of the embedding in TensorFlow's Projector

[https://projector.tensorflow.org/](https://projector.tensorflow.org/)

<img src="https://github.com/larryschirmer/practical-ml/raw/master/assets/visualize_embedding.gif" alt="visualize enbedding" width="600"/>

#### Text Classification using LSTM

Predict the author of a line of input text. Train a new model using an embedding and an LSTM with batched data from provided files.

```python
example_text = b'to be or not to be'
encoded_example = encoder.encode(example_text)

prediction = model.predict([encoded_example, ])
print(prediction) # [0.1, 0.1, 0.1, 0.7]
```

## Install

This project was setup using virtual environments, to reinstall the dependencies I used in this project:

Create a new virtual environment

```bash
python3 -m venv ./venv
```

Activate the environment

```bash
source ./venv/bin/activate
```

Reinstall the recorded dependencies

```bash
pip install -r requirements.txt
```
