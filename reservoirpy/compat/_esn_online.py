"""Echo State Networks with online learning rule (FORCE [#]_ learning).

References
----------

    .. [#] D. Sussillo and L. F. Abbott, ‘Generating Coherent
           Patterns of Activity from Chaotic Neural Networks’, Neuron,
           vol. 63, no. 4, pp. 544–557, Aug. 2009,
           doi: 10.1016/j.neuron.2009.07.018.

"""
from typing import Sequence, Callable, Tuple, Union

import numpy as np

from .utils.save import _save


class ESNOnline:
    """Echo State Networks using FORCE algorithm as
    an online learning rule.

    Warning
    -------

    The v0.2 model :py:class:`compat.ESNOnline` is deprecated.
    Consider using the new Node API introduced in v0.3 (see
    :ref:`node`).

    The :py:class:`compat.ESNOnline` implements FORCE, an
    online learning method.
    Online Echo State Networks allow one to:
    - quickly build online ESNs, using the
    :py:mod:`reservoirpy.mat_gen` module to initialize weights,
    - train and test ESNs on the task of your choice in an online
    fashion, i.e. continuously in time.

    Parameters
    ----------
    lr: float
        Leaking rate
    W: np.ndarray
        Reservoir weights matrix
    Win: np.ndarray
        Input weights matrix
    Wout: np.ndarray
        Readout weights matrix
    alpha_coef : float, optional
        Coefficient to scale the inversed state correlation matrix
        used for FORCE learning. By defautl, equal to
        :math:`1e^{-6}`.
    use_raw_input : bool, optional
        If True, input is used directly when computing output.
        By default, is False.
    input_bias: bool, optional
        If True, will add a constant bias
        to the input vector. By default, True.
    Wfb: np.array, optional
        Feedback weights matrix.
    fbfunc: Callable, optional
        Feedback activation function.
    typefloat: numpy.dtype, optional

    Attributes
    ----------
    Wout: np.ndarray
        Readout matrix
    dim_out: int
        Output dimension
    dim_in: int
        Input dimension
    N: int
        Number of neuronal units
    state : numpy.ndarray
        Last internal state computed, used to
        store internal dynamics of the reservoir
        over time, to enable online learning.
    output_values : numpy.ndarray
        Last values predicted by the network,
        used to store the last response of the
        reservoir to enable online learning.
    state_corr_inv : numpy.ndarray
        Inverse correlation matrix used in FORCE learning
        algorithm.
    """

    def __init__(self,
                 lr: float,
                 W: np.ndarray,
                 Win: np.ndarray,
                 dim_out: int,
                 alpha_coef: float = 1e-6,
                 use_raw_input: bool = False,
                 input_bias: bool = True,
                 Wfb: np.ndarray = None,
                 fbfunc: Callable = None,
                 typefloat: np.dtype = np.float64):
        self.W = W
        self.Win = Win
        self.Wfb = Wfb

        self.use_raw_inp = use_raw_input

        # number of neurons
        self.N = self.W.shape[1]

        self.in_bias = input_bias
        # dimension of inputs (not including the bias at 1)
        self.dim_inp = self.Win.shape[1] - 1 if self.in_bias else self.Win.shape[1]

        self.dim_out = dim_out
        self.state_size = self.dim_inp + self.N + 1 if self.use_raw_inp else self.N + 1

        self.Wout = np.zeros((self.dim_out, self.state_size), dtype=typefloat)

        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        self._autocheck_nan()

        self.output_values = np.zeros((self.dim_out, 1)).astype(typefloat)

        self.typefloat = typefloat
        self.lr = lr # leaking rate

        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError(f"If a feedback matrix is provided, \
                fbfunc must be a callable object, not {self.fbfunc}.")

        # coef used to init state_corr_inv matrix
        self.alpha_coef = alpha_coef

        # Init internal state vector and state_corr_inv matrix
        # (useful if we want to freeze the online learning)
        self.state = None
        self.reset_reservoir()
        self.reset_correlation_matrix()

    def __repr__(self):
        trained = True
        if self.Wout is None:
            trained = False
        fb = True
        if self.Wfb is None:
            fb=False
        out = f"ESN(trained={trained}, feedback={fb}, N={self.N}, "
        out += f"lr={self.lr}, input_bias={self.in_bias}, input_dim={self.N})"
        return out

    def _autocheck_nan(self):
        """ Auto-check to see if some important variables do not have a problem (e.g. NAN values). """
        # assert np.isnan(self.W).any() == False, "W matrix should not contain NaN values."
        assert np.isnan(self.Win).any() == False, "Win matrix should not contain NaN values."
        if self.Wfb is not None:
            assert np.isnan(self.Wfb).any() == False, "Wfb matrix should not contain NaN values."

    def _autocheck_dimensions(self):
        """ Auto-check to see if ESN matrices have correct dimensions."""
        # W dimensions check list
        assert len(self.W.shape) == 2, f"W shape should be (N, N) but is {self.W.shape}."
        assert self.W.shape[0] == self.W.shape[1], f"W shape should be (N, N) but is {self.W.shape}."

        # Win dimensions check list
        assert len(self.Win.shape) == 2, f"Win shape should be (N, input) but is {self.Win.shape}."
        err = f"Win shape should be ({self.W.shape[1]}, input) but is {self.Win.shape}."
        assert self.Win.shape[0] == self.W.shape[0], err

        # Wout dimensions check list
        assert len(self.Wout.shape) == 2, f"Wout shape should be (output, nb_states) but is {self.Wout.shape}."
        err = f"Wout shape should be (output, {self.state_size}) but is {self.Wout.shape}."
        assert self.Wout.shape[1] == self.state_size, err
        # Wfb dimensions check list
        if self.Wfb is not None:
            assert len(self.Wfb.shape) == 2, f"Wfb shape should be (input, output) but is {self.Wfb.shape}."
            err = f"Wfb shape should be ({self.Win.shape[0]}, {self.Wout.shape[0]}) but is {self.Wfb.shape}."
            assert (self.Win.shape[0],self.Wout.shape[0]) == self.Wfb.shape, err

    def _autocheck_io(self,
                      inputs,
                      outputs=None):

        # Check if inputs and outputs are lists
        assert type(inputs) is list, "Inputs should be a list of numpy arrays"
        if outputs is not None:
            assert type(outputs) is list, "Outputs should be a list of numpy arrays"

        # check if Win matrix has coherent dimensions with input dimensions
        inputs_0 = inputs[0]
        if self.in_bias:
            err = f"With bias, Win matrix should be of shape ({self.N}, "
            err += f"{inputs_0.shape[0] + 1}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs_0.shape[1] + 1, err
        else:
            err = f"Win matrix should be of shape ({self.N}, "
            err += f"{inputs_0.shape[1]}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs_0.shape[1], err

        if outputs is not None:
            # check feedback matrix
            if self.Wfb is not None:
                outputs_0 = outputs[0]
                err = f"With feedback, Wfb matrix should be of shape ({self.N}, "
                err += f"{outputs_0.shape[1]}) but is {self.Wfb.shape}."
                assert outputs_0.shape[1] == self.Wfb.shape[1], err
            # check output weights matrix
            if self.Wout is not None:
                outputs_0 = outputs[0]
                err = f"Wout matrix should be of shape ({outputs_0.shape[0]}, "
                err += f"{self.state_size}) but is {self.Wout.shape}."
                assert (outputs_0.shape[1], self.state_size) == self.Wout.shape, err

    def _get_next_state(self,
                        single_input: np.ndarray) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t), compute the state vector x(t+1).

        Parameters:
            single_input {np.ndarray} -- Input vector u(t).

        Raises:
            RuntimeError: feedback is enabled but no feedback vector is available.

        Returns:
            np.ndarray -- Next state s(t+1).
        """

        # check if feedback weights matrix is not None but empty feedback
        if self.Wfb is not None and self.output_values is None:
            raise RuntimeError("Missing a feedback vector.")

        x = self.state[1:self.N+1]

        # add bias
        if self.in_bias:
            u = np.hstack((1, single_input.flatten())).astype(self.typefloat)
        else:
            u = np.asarray(single_input)

        # linear transformation
        x1 = self.Win @ u.reshape(-1, 1) + self.W @ x.reshape(-1, 1)

        # add feedback if requested
        if self.Wfb is not None:
            x1 += self.Wfb @ self.fbfunc(self.output_values)

        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*np.tanh(x1)

        # return the next state computed
        if self.use_raw_inp:
            self.state = np.vstack((1.0, x1, single_input.reshape(-1, 1)))
        else:
            self.state = np.vstack((1.0, x1))

        return self.state.copy()

    def compute_output_from_current_state(self):
        """ Compute output from current state s(t) of the reservoir.

        Returns
        -------
            np.ndarray
                Output at time t.
        """

        assert self.Wout is not None, "Matrix Wout is not initialized/trained yet"

        self.output_values = (self.Wout @ self.state).astype(self.typefloat)
        return self.output_values.copy().ravel()

    def compute_output(self,
                       single_input: np.ndarray,
                       wash_nr_time_step: int = 0):
        """ Compute output from input to the reservoir.

        Parameters
        ---------
            single_input: np.ndarray
                Input vector u(t)
            wash_nr_time_step : int, optional
                Time for reservoir to run without collecting output
                or fitting ``Wout``. (default, 0)

        Returns
        -------
            np.ndarray, np.ndarray
                New state after input u(t) is passed and
                output after input u(t) is passed
        """

        state = self._get_next_state(single_input)
        output = self.compute_output_from_current_state()

        return output, state

    def reset_state(self):
        """Reset reservoir by setting internal values to zero.

        """
        self.state = np.zeros((self.state_size, 1), dtype=self.typefloat)

    def reset_reservoir(self):
        """Reset reservoir by setting internal values to zero.

        """
        self.reset_state()
        self.Wout = np.zeros((self.dim_out, self.state_size), dtype=self.typefloat)
        self.reset_correlation_matrix()

    def reset_correlation_matrix(self):
        """Reset inverse correlation state matrix to the initial value :

        .. math::

            Corr^{inv} = Id_{N} * \\alpha

        where :math:`\\alpha` is the ``alpha_coef`` and :math:`N` is the
        number of units in the reservoir.
        """
        self.state_corr_inv = np.asmatrix(np.eye(self.state_size)) / self.alpha_coef

    def train_from_current_state(self,
                                 targeted_output: np.ndarray,
                                 indexes: Union[int, list] = None):
        """ Train Wout from current internal state.

        Parameters
        ---------
            targeted_output : np.ndarray
                Expected output of the ESN.
            indexes : int or list, optional
                If indexes is not None, only the provided output
                indexes are learned.
        """

        error = self.output_values - targeted_output.reshape(-1, 1)

        self.state_corr_inv = _new_correlation_matrix_inverse(self.state,
                                                              self.state_corr_inv)

        if indexes is None:
            self.Wout -= error @ (self.state_corr_inv @ self.state).T
        else:
            self.Wout[indexes] -= error[indexes] * (self.state_corr_inv @ self.state).T

    def train(self,
              inputs: Sequence[np.ndarray],
              teachers: Sequence[np.ndarray],
              wash_nr_time_step: int = 0,
              verbose: bool = False) -> Sequence[np.ndarray]:
        """Train the ESN model on a sequence of inputs.

        Parameters
        ---------
            inputs: list of numpy.ndarray
                List of inputs.
                Note that it should always be a list of sequences, i.e. if
                only one sequence (array with rows representing time axis)
                of inputs is used, it should be alone in a list.
            teachers: list of numpy.ndarray
                List of ground truths.
                Note that is should always be a list of
                sequences of the same length than the `inputs`, i.e. if
                only one sequence of inputs is used, it should be alone in a
                list.
            wash_nr_time_step: int, optional
                Number of states to considered as transient when training. Transient
                states will be discarded when computing readout matrix. By default,
                no states are removes.
            verbose : bool

        Returns
        -------
            list of np.ndarray
                All states computed, for all inputs.
        """
        inputs_concat = [inp[t, :].reshape(-1, self.dim_inp)
                         for inp in inputs for t in range(inp.shape[0])]
        teachers_concat = [tea[t, :].reshape(-1, self.dim_out)
                           for tea in teachers for t in range(tea.shape[0])]

        ## Autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs_concat, outputs=teachers_concat)

        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Training on {len(inputs)} inputs ({steps} steps) "
                  f"-- wash: {wash_nr_time_step} steps")

        # List of all internal states when training
        all_states = []
        start = 1 if self.in_bias else 0
        end = self.N + start

        for i in range(len(inputs)):

            t = 0
            all_states_inp_i = []

            # First 'warm up' the network
            while t < wash_nr_time_step:
                self.compute_output(inputs_concat[i+t])
                t += 1

            # Train Wout on each input
            while t < inputs[i].shape[0]:
                _, state = self.compute_output(inputs_concat[i+t])
                self.train_from_current_state(teachers_concat[i+t])
                all_states_inp_i.append(state[start:end])
                t += 1

            # Pack in all_states
            all_states.append(np.hstack(all_states_inp_i))

        # return all internal states
        return [st.T for st in all_states]

    def run(self,
            inputs: Sequence[np.ndarray],
            verbose: bool = False) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Run the model on a sequence of inputs, and returned the states and
           readouts vectors.

        Parameters
        ----------
            inputs: list of numpy.ndarray
                List of inputs.
                Note that it should always be a list of sequences, i.e. if
                only one sequence (array with rows representing time axis)
                of inputs is used, it should be alone in a list.
            verbose : bool

        Returns
        -------
            list of numpy.ndarray, list of numpy.ndarray
                All outputs computed from readout and all corresponding internal states,
                for all inputs.
        """

        inputs_concat = [inp[t,:].reshape(-1, self.dim_inp) for inp in inputs for t in range(inp.shape[0])]

        steps = np.sum([i.shape[0] for i in inputs])
        if verbose:
            print(f"Running on {len(inputs)} inputs ({steps} steps)")

        # autochecks of inputs
        self._autocheck_io(inputs=inputs_concat)

        all_outputs = []
        all_states = []
        for i in range(len(inputs)):
            internal_pred = []; output_pred = []
            for t in range(inputs[i].shape[0]):
                output, state = self.compute_output(inputs_concat[i+t])
                internal_pred.append(state)
                output_pred.append(output)
            all_states.append(np.asarray(internal_pred))
            all_outputs.append(np.asarray(output_pred))

        # return all_outputs, all_int_states
        return all_outputs, all_states

    def save(self, directory: str):
        """Save the ESN to disk.

        Parameters
        ----------
            directory: str or Path
                Directory where to save the model..
        """
        _save(self, directory)


def _new_correlation_matrix_inverse(new_data, old_corr_mat_inv):
    """
        If old_corr_mat_inv is an approximation for the correlation
        matrix inverse of a dataset (p1, ..., pn), then the function
        returns an approximatrion for the correlation matrix inverse
        of dataset (p1, ..., pn, new_data)

        TODO : add forgetting parameter lbda
    """

    P = old_corr_mat_inv
    x = new_data

    # TODO : numerical instabilities if xTP is not computed first
    # (order of multiplications)
    xTP = x.T @ P
    P = P - (P @ x @ xTP)/(1. + np.dot(xTP, x))

    return P
