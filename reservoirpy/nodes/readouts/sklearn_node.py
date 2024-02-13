# Author: Deepayan Das at 16/08/2023 <deepayan.das@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from copy import deepcopy
from functools import partial

import numpy as np

from ...node import Node
from ...utils.random import rand_generator


def forward(readout: Node, X):
    instances = readout.params.get("instances")
    if type(instances) is not list:
        return instances.predict(X)
    else:
        return np.concatenate([instance.predict(X) for instance in instances], axis=-1)


def backward(readout: Node, X, Y):
    # Concatenate all the batches as one np.ndarray
    # of shape (timeseries*timesteps, features)
    X_ = np.concatenate(X, axis=0)
    Y_ = np.concatenate(Y, axis=0)

    instances = readout.params.get("instances")
    if type(instances) is not list:
        if readout.output_dim > 1:
            # Multi-output node and multi-output sklearn model
            instances.fit(X_, Y_)
        else:
            # Y_ should have 1 feature so we reshape to
            # (timeseries, ) to avoid scikit-learn's DataConversionWarning
            instances.fit(X_, Y_[..., 0])
    else:
        for i, instance in enumerate(instances):
            instance.fit(X_, Y_[..., i])


def initialize(readout: Node, x=None, y=None, model_hypers=None):
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

        first_instance = readout.model(**deepcopy(model_hypers))
        # If there are multiple output but the specified model doesn't support
        # multiple outputs, we create an instance of the model for each output.
        if out_dim > 1 and not first_instance._get_tags().get("multioutput"):
            instances = [
                readout.model(**deepcopy(model_hypers)) for i in range(out_dim)
            ]
            readout.set_param("instances", instances)
        else:
            readout.set_param("instances", first_instance)

        return


class ScikitLearnNode(Node):
    """
    A node interfacing a scikit-learn linear model that can be used as an offline
    readout node.

    The ScikitLearnNode takes a scikit-learn model as parameter and creates a
    node with the specified model.

    We currently support classifiers (like
    :py:class:`sklearn.linear_model.LogisticRegression` or
    :py:class:`sklearn.linear_model.RidgeClassifier`) and regressors (like
    :py:class:`sklearn.linear_model.Lasso` or
    :py:class:`sklearn.linear_model.ElasticNet`).

    For more information on the above-mentioned estimators,
    please visit scikit-learn linear model API reference
    <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_

    :py:attr:`ScikitLearnNode.params` **list**

    ================== =================================================================
    ``instances``      Instance(s) of the model class used to fit and predict. If
                       :py:attr:`ScikitLearnNode.output_dim` > 1 and the model doesn't
                       support multi-outputs, `instances` is a list of instances, one
                       for each output feature.
    ================== =================================================================

    :py:attr:`ScikitLearnNode.hypers` **list**

    ================== =================================================================
    ``model``          (class) Underlying scikit-learn model.
    ``model_hypers``   (dict) Keyword arguments for the scikit-learn model.
    ================== =================================================================

    Parameters
    ----------
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    model : str, optional
        Node name.
    model_hypers
        (dict) Additional keyword arguments for the scikit-learn model.

    Example
    -------
    >>> from reservoirpy.nodes import Reservoir, ScikitLearnNode
    >>> from sklearn.linear_model import Lasso
    >>> reservoir = Reservoir(units=100)
    >>> readout = ScikitLearnNode(model=Lasso, model_hypers={"alpha":1e-5})
    >>> model = reservoir >> readout
    """

    def __init__(self, model, model_hypers=None, output_dim=None, **kwargs):
        if model_hypers is None:
            model_hypers = {}

        if not hasattr(model, "fit"):
            model_name = model.__name__
            raise AttributeError(
                f"Specified model {model_name} has no method called 'fit'."
            )
        if not hasattr(model, "predict"):
            model_name = model.__name__
            raise AttributeError(
                f"Specified model {model_name} has no method called 'predict'."
            )

        # Ensure reproducibility
        # scikit-learn currently only supports RandomState
        if (
            model_hypers.get("random_state") is None
            and "random_state" in model.__init__.__kwdefaults__
        ):

            generator = rand_generator()
            model_hypers.update(
                {
                    "random_state": np.random.RandomState(
                        seed=generator.integers(1 << 32)
                    )
                }
            )

        super(ScikitLearnNode, self).__init__(
            hypers={"model": model, "model_hypers": model_hypers},
            params={"instances": None},
            forward=forward,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize, model_hypers=model_hypers),
            **kwargs,
        )
