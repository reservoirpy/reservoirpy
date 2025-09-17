# This is much more simple than reservoirpy.activationsfunc
# since those activation functions are defined in jax.nn

from typing import Callable, Union

from jax.nn import identity, relu, sigmoid, softmax, softplus, tanh


def get_function(name: Union[Callable, str]) -> Callable:
    """Return a Jax activation function from name.

    Parameters
    ----------
    name : str, Callable
        Name of the activation function.
        Can be one of {'softmax', 'softplus',
        'sigmoid', 'tanh', 'identity', 'relu'} or
        their respective short names {'smax', 'sp',
        'sig', 'id', 're'}. If `name` is a Callable,
        simply returns `name`.

    Returns
    -------
    callable
        An activation function.
    """
    if isinstance(name, Callable):
        return name

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

    if not name in index:
        raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
    else:
        return index[name]
