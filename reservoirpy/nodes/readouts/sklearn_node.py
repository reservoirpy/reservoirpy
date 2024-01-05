# Author: Deepayan Das at 16/08/2023 <deepayan.das@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from copy import deepcopy
from functools import partial

import numpy as np

from ...node import Node


def forward(readout: Node, X):
    instances = readout.params.get("instances")
    if type(instances) is not list:
        return instances.predict(X)
    else:
        return np.concatenate([instance.predict(X) for instance in instances], axis=-1)


def backward(readout: Node, X, Y):
    # Concatenate all the batches as one np.ndarray of shape (timeseries*timesteps, features)
    X_ = np.concatenate(X, axis=0)
    Y_ = np.concatenate(Y, axis=0)

    instances = readout.params.get("instances")
    if type(instances) is not list:
        instances.fit(X_, Y_)
    else:
        for i, instance in enumerate(instances):
            instance.fit(X_, Y_[..., i])


def initialize(readout: Node, x=None, y=None, *args, **kwargs):

    if x is not None:

        in_dim = x.shape[1]
        if readout.output_dim is not None:
            out_dim = readout.output_dim
        elif y is not None:
            out_dim = y.shape[1]
        else:
            raise RuntimeError(
                f"Impossible to initialize {readout.name}: "
                f"output dimension was not specified at "
                f"creation, and no teacher vector was given."
            )

        readout.set_input_dim(in_dim)
        readout.set_output_dim(out_dim)

        kwargs = {k: v for k, v in kwargs.items() if v}
        first_instance = readout.model(**deepcopy(kwargs))
        # If there are multiple output but the specified model doesn't support
        # multiple outputs, we create an instance of the model for each output.
        if out_dim > 1 and not first_instance._get_tags().get("multioutput"):
            instances = [readout.model(**deepcopy(kwargs)) for i in range(out_dim)]
            readout.set_param("instances", instances)
        else:
            readout.set_param("instances", first_instance)

        return


class ScikitLearnNode(Node):
    """
    A node interfacing a scikit-learn linear model that can be used as an offline
    readout node.

    The ScikitLearnNode takes a scikit-learn linear model as parameter and creates a
    node with the specified model.

    We currently support linear classifiers like `LogisticRegression`,
    `RidgeClassifier` and linear regressors like `Ridge`, `LinearRegression`
    Lasso and ElasticNet.

    For more information on the above-mentioned estimators,
    please visit scikit-learn linear model API reference <https://scikit-learn.org/
    stable/modules/classes.html#module-sklearn.linear_model>`_

    :py:attr:`ScikitLearnNode.hypers` **list**

    ================== =================================================================
    ``model``              Underlying scikit-learn model.
    ================== =================================================================

    Parameters
    ----------
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    name : str, optional
        Node name.
    **kwargs
        Additional keyword arguments for the scikit-learn model.

    Example
    -------
    >>> from reservoirpy import ScikitLearnNode
    >>> node = ScikitLearnNode(name="Ridge", alpha=0.5)
    """

    def __init__(self, output_dim=None, model=None, **kwargs):
        model_name = model.__name__
        if not hasattr(model, "fit"):
            raise AttributeError(
                f"Specified model {model_name} has no method called 'fit'."
            )
        if not hasattr(model, "predict"):
            raise AttributeError(
                f"Specified model {model_name} has no method called 'predict'."
            )

        super(ScikitLearnNode, self).__init__(
            hypers={"model": model, **kwargs},
            params={"instances": None},
            forward=forward,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize, **kwargs),
        )
