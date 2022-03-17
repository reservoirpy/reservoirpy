# Author: Nathan Trouvain at 15/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

try:
    import sklearn
except ImportError:
    sklearn = None

from ..node import Node


def forward_predict(estimator, x):
    return np.atleast_2d(estimator.estimator.predict(x))


def forward_transform(estimator, x):
    return np.atleast_2d(estimator.estimator.transform(x))


def backward(estimator, X, Y=None):
    estimator.hypers["estimator"] = estimator.estimator.fit(X, Y)


def initialize(estimator, x, y=None):
    ...


class from_sklearn(Node):
    def __init__(self, estimator):
        if hasattr(estimator, "predict"):
            forward = forward_predict
        elif hasattr(estimator, "transform"):
            forward = forward_transform
        else:
            raise TypeError(
                f"Estimator {estimator} has no 'predict' or 'transform' attribute."
            )

        super(from_sklearn, self).__init__(
            hypers={"estimator": estimator}, forward=forward, backward=backward
        )
