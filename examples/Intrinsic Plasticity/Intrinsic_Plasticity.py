# Author: Nathan Trouvain at 20/01/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from reservoirpy.activationsfunc import sigmoid, tanh
from reservoirpy.datasets import mackey_glass
from reservoirpy.mat_gen import generate_input_weights, generate_internal_weights

LR = 1.0
N = 100
DENSITY = 0.1
SR = 0.9
IS = 0.1
SIGMA = 0.1
MU = 0.3
TRANSIENT = 100
ETA = 5e-4
EPOCHS = 100

rng = default_rng(123456789)


def narma(steps, order=30):
    y = np.zeros(steps + order)
    noise = rng.uniform(0, 0.5, size=steps + order)
    for t in range(order, order + steps - 1):
        y[t + 1] = (
            0.2 * y[t]
            + 0.04 * y[t] * np.sum(y[t - order : t])
            + 1.5 * noise[t - order] * noise[t]
            + 0.001
        )
    return y[order:].reshape(-1, 1)


def gaussian_gradients(x, y, a, sigma, mu):
    sig2 = sigma**2
    delta_b = (mu / sig2) - (y / sig2) * (2 * sig2 + 1 - y**2 + mu * y)
    delta_a = (1 / a) + delta_b * x
    return delta_a, delta_b


def exp_gradients(x, y, a, mu):
    mu_inv = 1 / mu
    delta_b = 1 - (2 + mu_inv) * y + mu_inv * y**2
    delta_a = (1 / a) + delta_b * x
    return delta_a, delta_b


def apply_gradients(a, b, delta_a, delta_b, eta):
    a2 = a + eta * delta_a
    b2 = b + eta * delta_b
    return a2, b2


def res_states(u, x, W, Win):
    _u = np.c_[np.ones((1, 1)), u]
    _x = W @ x.T + Win @ _u.T
    return _x.T


def tanh_activation(s, a, b):
    return tanh(a * s + b)


def sigmoid_activation(s, a, b):
    return sigmoid(a * s + b)


def esn():
    W = generate_internal_weights(N, sr=SR, proba=1.0, dist="uniform", seed=rng)
    Win = generate_input_weights(
        N, 1, input_scaling=IS, proba=1, input_bias=True, seed=rng
    )

    a = np.ones((1, N))
    b = np.zeros((1, N))

    return W, Win, a, b


if __name__ == "__main__":

    steps = 1000
    W, Win, a, b = esn()
    X = narma(steps)
    X = (X - X.min()) / (X.ptp())

    s = np.zeros((1, N))
    pre_X = np.vstack([X] * EPOCHS)
    A, B = np.zeros((len(pre_X), N)), np.zeros((len(pre_X), N))
    for t, u in tqdm(enumerate(pre_X), total=len(pre_X)):
        pre_s = res_states(u.reshape(1, -1), s.reshape(1, -1), W, Win)
        s = sigmoid_activation(pre_s, a, b)
        delta_a, delta_b = exp_gradients(pre_s, s, a, MU)
        a, b = apply_gradients(a, b, delta_a, delta_b, ETA)
        A[t, :] = a
        B[t, :] = b

    states = np.zeros((steps - TRANSIENT, N))
    s = states[0, :]
    for t, u in enumerate(X):
        pre_s = res_states(u.reshape(1, -1), s.reshape(1, -1), W, Win)
        s = sigmoid_activation(pre_s, a, b)
        if t > TRANSIENT:
            states[t - TRANSIENT, :] = s

    plt.plot(states[:, :20])
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.hist(states.flatten(), bins=200)
    ax2.hist(states[:, 1].flatten(), bins=200)
    plt.show()

    plt.plot(A[:, :20], color="blue", label="a")
    plt.plot(B[:, :20], color="red", label="b")
    plt.show()
