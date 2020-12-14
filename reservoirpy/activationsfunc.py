import numpy as np

# TODO: major refactorization: module name, backend handler, doc


def get_function(name):  # pragma: no cover
    index = {
        "softmax": softmax,
        "softplus": softplus,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "identity": identity,
        "relu": relu,
        "smax": softmax,
        "sp": softplus,
        "sig": sigmoid,
        "id": identity,
        "re": relu,
    }

    if index.get(name) is None:
        raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
    else:
        return index[name]


def elementwise(func):
    vect = np.vectorize(func)

    def vect_wrapper(x):
        u = np.asanyarray(x)
        return vect(u)

    return vect_wrapper


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


@elementwise
def softplus(x):
    return np.log(1 + np.exp(x))


@elementwise
def sigmoid(x):
    if x < 0:
        u = np.exp(x)
        return u / (u + 1)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


@elementwise
def identity(x):
    return x


@elementwise
def relu(x):
    if x < 0:
        return 0
    return x
