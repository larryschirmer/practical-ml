import time
import tensorflow as tf
import matplotlib.pyplot as plt


def random_float():
    return tf.random.uniform(shape=(1,)).numpy()[0]


def data_creation(w=0.1, b=0.5, n=100):
    X = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(n,), stddev=0.01)
    Y = X*w + b + noise
    return X, Y


def predict(x, w_guess, b_guess):
    y = x*w_guess + b_guess
    return y


def mean_square_error(y_pred, Y):
    return tf.reduce_mean(tf.square(y_pred-Y))


def fit(steps, X, Y, lr):
    w = tf.Variable(random_float())
    b = tf.Variable(random_float())
    for step in range(steps):
        with tf.GradientTape() as tape:
            predictions = predict(X, w, b)
            loss = mean_square_error(predictions, Y)

        [grad_w, grad_b] = tape.gradient(loss, [w, b])

        w.assign_sub(grad_w*lr)
        b.assign_sub(grad_b*lr)

        if step % 100 == 0:
            plot_data_with_trend(X, Y, w, b)
            time.sleep(0.25)


def plot_1D_tensor_hist(tensor, filename='1d-tensor-hist.png'):
    plt.hist(tensor.numpy())
    plt.savefig(filename)


def plot_data_with_trend(X, Y, w, b, plot_options=['r.', 'g:'], filename='linear_regression/plot-with-trend.png'):
    X = X.numpy()
    Y = Y.numpy()
    w = w.numpy()
    b = b.numpy()

    plt.plot(X, Y, plot_options[0])
    plt.plot([0, 1], [0*w+b, 1*w+b], plot_options[1])
    plt.savefig(filename)
    plt.clf()
