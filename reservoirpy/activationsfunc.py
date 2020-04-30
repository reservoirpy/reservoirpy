import numpy as np


def get_function(name):
    index = {
        "softmax": softmax,
        "softplus": softplus,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "maxout": maxout,
        "identity": identity,
        "relu": relu,
        "smax": softmax,
        "sp": softplus,
        "sig": sigmoid,
        "max": maxout,
        "id": identity,
        "re": relu,
    }
    
    if index.get(name) is None:
        raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
    else:
        return index[name]

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def softplus(x):
    return np.log(1 + np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def maxout(x):
    y = np.zeros_like(x)
    y[x.argmax()] = x.max()
    return y

def identity(x):
    return x.copy()

def relu(x):
    y = x.copy()
    y[x < 0] = 0
    return y