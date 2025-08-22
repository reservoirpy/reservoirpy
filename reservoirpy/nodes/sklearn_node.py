# Author: Deepayan Das at 16/08/2023 <deepayan.das@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from copy import deepcopy
from typing import Any, Optional, Sequence, Union

import numpy as np

from reservoirpy.type import NodeInput, State, Timeseries, Timestep, is_array
from reservoirpy.utils.data_validation import check_node_input

from ..node import TrainableNode
from ..utils.random import rand_generator


class ScikitLearnNode(TrainableNode):
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


    Parameters
    ----------
    model : class, scikit-learn model
        scikit-learn class to be wrapped by the Node.
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    name : str, optional
        Node name.
    **kwargs
        Additional keyword arguments passed to the scikit-learn model.

    Example
    -------
    >>> from reservoirpy.nodes import Reservoir, ScikitLearnNode
    >>> from sklearn.linear_model import Lasso
    >>> reservoir = Reservoir(units=100)
    >>> readout = ScikitLearnNode(model=Lasso, model_hypers={"alpha":1e-5})
    >>> model = reservoir >> readout
    """

    #: scikit-learn class to be wrapped by the Node.
    model: type["sklearn.base.BaseEstimator"]
    #: Additional keyword arguments passed to the scikit-learn model.
    model_kwargs: dict[str, Any]
    #: Model instance or list of model instances if multiple output are expected
    #: and the scikit-learn model doesn't support it.
    instances: Union["sklearn.base.BaseEstimator", list["sklearn.base.BaseEstimator"]]

    def __init__(
        self,
        model,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if not hasattr(model, "fit"):
            model_name = model.__name__
            raise AttributeError(f"Specified model {model_name} has no method called 'fit'.")
        if not hasattr(model, "predict"):
            model_name = model.__name__
            raise AttributeError(f"Specified model {model_name} has no method called 'predict'.")

        # Ensure reproducibility
        # scikit-learn currently only supports RandomState
        if not "random_state" in kwargs and "random_state" in model.__init__.__kwdefaults__:
            generator = rand_generator()
            kwargs.update({"random_state": np.random.RandomState(seed=generator.integers(1 << 32))})

        self.model = model
        self.name = name
        self.model_kwargs = kwargs
        self.output_dim = output_dim
        self.state = {"out": None}
        self.initialized = False

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        self._set_input_dim(x)
        self._set_output_dim(y)

        # TODO: Use https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html instead
        first_instance = self.model(**deepcopy(self.model_kwargs))
        # If there are multiple output but the specified model doesn't support
        # multiple outputs, we create an instance of the model for each output.
        if self.output_dim > 1 and not first_instance._get_tags().get("multioutput"):
            self.instances = [self.model(**deepcopy(self.model_kwargs)) for i in range(self.output_dim)]
        else:
            self.instances = first_instance
        self.state = {"out": np.zeros((self.output_dim,))}
        self.initialized = True

    def _step(self, state: State, x: Timestep):
        instances = self.instances
        if not isinstance(instances, list):
            res = instances.predict(x.reshape(1, -1)).ravel()
        else:
            res = np.concatenate(
                [instance.predict(x.reshape(1, -1)).ravel() for instance in instances],
                axis=-1,
            )
        return {"out": res}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        instances = self.instances
        if not isinstance(instances, list):
            res = instances.predict(x)
            if res.ndim == 1:
                res = res[..., np.newaxis]
        else:
            res = np.column_stack([instance.predict(x) for instance in instances])
        return {"out": res[-1]}, res

    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0):
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y, expected_dim=self.output_dim)

        if not self.initialized:
            self.initialize(x, y)

        if isinstance(x, Sequence):
            # Concatenate all the batches as one np.ndarray
            # of shape (timeseries*timesteps, features)
            x = np.concatenate(x, axis=0)
        if is_array(x) and x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        if isinstance(y, Sequence):
            # Concatenate all the batches as one np.ndarray
            # of shape (timeseries*timesteps, features)
            y = np.concatenate(y, axis=0)
        if is_array(y) and y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])

        instances = self.instances
        if not isinstance(instances, list):
            if self.output_dim > 1:
                # Multi-output node and multi-output sklearn model
                instances.fit(x, y)
            else:
                # Y_ should have 1 feature so we reshape to
                # (timeseries, ) to avoid scikit-learn's DataConversionWarning
                instances.fit(x, y[..., [0]])
        else:
            for i, instance in enumerate(instances):
                instance.fit(x, y[..., [i]])

        return self
