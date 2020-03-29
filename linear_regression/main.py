import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import data_creation, plot_1D_tensor_hist, plot_data_with_trend, predict, mean_square_error, fit

#hyperparams
lr = 0.01
steps = 5000

X, Y = data_creation()
fit(steps, X, Y, lr)