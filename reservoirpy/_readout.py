from typing import Callable, Union, Optional

import numpy as np

from ._utils import _check_vector, _add_bias
from ._types import Weights, AnonymousReadout


class Readout:
    """Base tool for readout connections training
    and prediction.

    This class can be used as a callable able to
    "read in the internal states" the desired output.

    It can contain any
    """
    fitted: bool = False

    def __init__(self,
                 dim_output: int = None,
                 input_bias: bool = True,
                 Wout: Weights = None,
                 reg_model: Callable = None,
                 use_inputs: bool = False):
        self.dim_output = dim_output
        self.Wout = Wout
        self.input_bias = input_bias
        self.model = reg_model
        self.use_inputs = use_inputs
        self.dim_output = dim_output

        self._toggle_fitted()

    def __call__(self,
                 states: np.ndarray,
                 inputs: np.ndarray = None
                 ) -> np.ndarray:
        """Outputs prediction from a sequence of states.

        Parameters
        ----------
        states: numpy.ndarray
            A sequence of reservoir internal
            states to decode using the readout
            connections in `Wout`.
        inputs: numpy.ndarray, optional
            If `self.use_inputs` is `True`, will used the
            inputs passed along with the states to perform
            the readout. The readout must have been trained
            in this fashion before.

        Returns
        -------
            numpy.ndarray
                A sequence of predicted values.
        """
        if self.fitted:
            x = _check_vector(states)
            if self.use_inputs:
                if inputs is not None:
                    inputs = _check_vector(inputs)
                    x = np.c_[inputs, x]

            if self.input_bias:
                x = _add_bias(x, 1.0)

            outputs = np.zeros((x.shape[0], self.dim_output))
            for i, x in enumerate(x):
                y = self.Wout @ x.T
                outputs[i, :] = y

            return outputs
        else:
            raise RuntimeError("Impossible to compute outputs: "
                               "no readout matrix available.")

    def _toggle_fitted(self):
        """Change the `fitted` parameter
        value given the presence (or absence)
        of a readout matrix. Infer the output
        dimension.
        """
        if self.Wout is not None:
            self.fitted = True
            if self.dim_output is None:
                self.dim_output = self.Wout.shape[0]
        else:
            self.fitted = False

    def fit(self, states: np.ndarray,
            y: np.ndarray,
            inputs: np.ndarray = None
            ) -> Weights:
        """Fit the states obtained from a reservoir
        to some target values `y`, i.e. compute a
        readout matrix `Wout` such that:

        .. math::

            \\hat{y} = \\mathrm{Wout} \\cdot \\mathrm{states}^T

        where :math:`\\hat{y}` are the predicted values.

        Parameters
        ----------
        states: numpy.ndarray
            A sequence of reservoir internal
            states.
        y: numpy.ndarray
            A corresponding sequence of target
            values.
        inputs: numpy.ndarray, optional
            If `self.use_inputs` is `True`, will used the
            inputs passed along with the states to fit
            the readout. Passing the inputs to the readout
            along with the states will remain necessary
            during prediction.

        Returns
        -------
            numpy.ndarray
                A readout matrix `Wout`.
        """
        x = _check_vector(states)
        if self.use_inputs:
            if inputs is not None:
                inputs = _check_vector(inputs)
                x = np.c_[inputs, x]

        if self.input_bias:
            x = _add_bias(x, 1.0)

        self.Wout = self.model(x, y)

        self._toggle_fitted()

        return self.Wout


def teacher_forcing(teachers: np.ndarray = None,
                    *args) -> Optional[AnonymousReadout]:
    """Stud readout used for teacher forcing.

    When using feedback, it is useful during training
    to "force" the desired targets in the reservoir
    through the feedback connections. This readout
    will yield these values to the reservoir, given
    a sequence of targets `teachers`. If no feedback
    is needed, i.e. no teachers are passed as parameter,
    it will return `None`.

    Parameters
    ----------
    teachers: numpy.ndarray
        Sequence of targets to fed the reservoir with
        when training with feedback connections enabled.
    args: Any

    Returns
    -------
        Callable
            A stub readout function that :py:class:`Reservoir`
            will be able to call to obtain targets as feedback.
    """
    if teachers is not None:
        gen = (t for t in teachers)

        def _teacher_generator(*gen_args) -> np.ndarray:
            return next(gen)

        return _teacher_generator
    else:
        return None
