"""Simple, fast, parallelizable and object-oriented
implementation of Echo State Networks [#]_ [#]_, using offline
learning methods.

References
----------

    .. [#] H. Jaeger, ‘The “echo state” approach to analysing
           and training recurrent neural networks – with an
           Erratum note’, p. 48.
    .. [#] M. Lukoševičius, ‘A Practical Guide to Applying Echo
           State Networks’, Jan. 2012, doi: 10.1007/978-3-642-35289-8_36.

"""
# @author: Xavier HINAUT
# xavier.hinaut@inria.fr
# Copyright Xavier Hinaut 2018
# We would like to thank Mantas Lukosevicius for his code that
# was used as inspiration for this code:
# # http://minds.jacobs-university.de/mantas/code

import time
import warnings

from typing import Sequence, Callable, Tuple, Union
from tempfile import mkdtemp
from pathlib import Path

import joblib
import numpy as np

from tqdm import tqdm
from numpy.random import default_rng, SeedSequence, Generator

from ._utils import _check_values, _save
from .regression_models import sklearn_linear_model
from .regression_models import ridge_linear_model
from .regression_models import pseudo_inverse_linear_model


class ESN:
    """Base class of Echo State Networks.

    The :py:class:`reservoirpy.ESN` class is the angular stone of ReservoirPy
    offline learning methods using reservoir computing.
    Echo State Network allows one to:
        - quickly build ESNs, using the :py:mod:`reservoirpy.mat_gen` module
          to initialize weights,
        - train and test ESNs on the task of your choice,
        - use the trained ESNs on the task of your choice, either in
          predictive mode or generative mode.

    Parameters
    ----------
        lr: float
            Leaking rate

        W: np.ndarray
            Reservoir weights matrix

        Win: np.ndarray
            Input weights matrix

        input_bias: bool, optional
            If True, will add a constant bias
            to the input vector. By default, True.

        reg_model: Callable, optional
            A scikit-learn linear model function to use for
            regression. Should be None if ridge is used.

        ridge: float, optional
            Ridge regularization coefficient for Tikonov regression.
            Should be None if reg_model is used. By default, pseudo-inversion
            of internal states and teacher signals is used.

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
        dim_inp: int
            Input dimension
        N: int
            Number of neuronal units

    See also
    --------
        reservoirpy.ESNOnline for ESN with online learning using FORCE.

    """
    # memmap temporay storage
    # is cleaned after training/running
    _tempstates = Path(mkdtemp(), "states.dat")
    _tempteach = Path(mkdtemp(), "teachers.dat")

    def __init__(self,
                 lr: float,
                 W: np.ndarray,
                 Win: np.ndarray,
                 input_bias: bool = True,
                 reg_model: Callable = None,
                 ridge: float = None,
                 Wfb: np.ndarray = None,
                 fbfunc: Callable = None,
                 noise_in: float = 0.0,
                 noise_rc: float = 0.0,
                 noise_out: float = 0.0,
                 seed: int = None,
                 typefloat: np.dtype = np.float64):

        self.W = W
        self.Win = Win
        # output weights matrix. must be learnt through training.
        self.Wout = None
        self.Wfb = Wfb

        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        self._autocheck_nan()

        # number of neurons
        self.N = self.W.shape[1]
        self.in_bias = input_bias
        # dimension of inputs (including the bias at 1)
        self.dim_inp = self.Win.shape[1]
        self.dim_out = None
        if self.Wfb is not None:
            # dimension of outputs
            self.dim_out = self.Wfb.shape[1]

        self.typefloat = typefloat
        self.lr = lr  # leaking rate

        self.noise_in = noise_in
        self.noise_rc = noise_rc
        self.noise_out = noise_out

        self.seed = seed

        self.ridge = None
        self.sklearn_model = None
        self.reg_model = self._get_regression_model(ridge, reg_model)
        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError("If a feedback matrix is provided, fbfunc must"
                             f"be a callable object, not {self.fbfunc}.")

    def __repr__(self):
        trained = True
        if self.Wout is None:
            trained = False
        fb = self.Wfb is not None

        out = f"ESN(trained={trained}, feedback={fb}, N={self.N}, "
        out += f"lr={self.lr}, input_bias={self.in_bias}, input_dim={self.N})"
        return out

    def _get_regression_model(self,
                              ridge: float = None,
                              sklearn_model: Callable = None):
        """Set the type of regression used in the model. All regression models available
        for now are described in reservoipy.regression_models:
            - any scikit-learn linear regression model (like Lasso or Ridge)
            - Tikhonov linear regression (l1 regularization)
            - Solving system with pseudo-inverse matrix
        """
        if ridge is not None and sklearn_model is not None:
            raise ValueError("ridge and sklearn_model can not be"
                             "defined at the same time.")

        elif ridge is not None:
            self.ridge = ridge
            return ridge_linear_model(self.ridge)

        elif sklearn_model is not None:
            self.sklearn_model = sklearn_model
            return sklearn_linear_model(self.sklearn_model)

        else:
            return pseudo_inverse_linear_model()

    def _autocheck_nan(self):
        """ Auto-check to see if some important variables do not have
        a problem (e.g. NAN values).
        """
        assert np.isnan(self.Win).any() == False, \
            "Win matrix should not contain NaN values."
        if self.Wfb is not None:
            assert np.isnan(self.Wfb).any() == False, \
                "Wfb matrix should not contain NaN values."

    def _autocheck_dimensions(self):
        # W dimensions check list
        assert len(self.W.shape) == 2, ("W shape should be 2-dimensional "
                                        f"but is {len(self.W.shape)}-dimensional "
                                        f"({self.W.shape}).")

        assert self.W.shape[0] == self.W.shape[1], f"W shape should be (N, N) but is {self.W.shape}."

        # Win dimensions check list
        assert len(self.Win.shape) == 2, f"Win shape should be (N, input) but is {self.Win.shape}."
        err = f"Win shape should be ({self.W.shape[1]}, input) but is {self.Win.shape}."
        assert self.Win.shape[0] == self.W.shape[0], err

    def _autocheck_io(self, inputs, outputs=None):
        # Check if inputs and outputs are lists
        assert type(inputs) is list, "Inputs should be a list of numpy arrays"
        if outputs is not None:
            assert type(outputs) is list, "Outputs should be a list of numpy arrays"

        # check if Win matrix has coherent dimensions with input dimensions
        if self.in_bias:
            err = f"With bias, Win matrix should be of shape ({self.N}, "
            err += f"{inputs[0].shape[1] + 1}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs[0].shape[1] + 1, err
        else:
            err = f"Win matrix should be of shape ({self.N}, "
            err += f"{self.dim_inp}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs[0].shape[1], err

        if outputs is not None:
            # check feedback matrix
            if self.Wfb is not None:
                err = f"With feedback, Wfb matrix should be of shape ({self.N}, "
                err += f"{outputs[0].shape[1]}) but is {self.Wfb.shape}."
                assert outputs[0].shape[1] == self.Wfb.shape[1], err

    def _get_next_state(self,
                        single_input: np.ndarray,
                        feedback: np.ndarray = None,
                        last_state: np.ndarray = None,
                        noise_generator: Generator = None) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t),
        compute the state vector x(t+1).

        Parameters
        ----------
            single_input: np.ndarray
                Input vector u(t)

            feedback: numpy.ndarray, optional
                Feedback vector if enabled.
            last_state: numpy.ndarray, optional
                Current state to update x(t). Default to 0 vector.

        Raises
        ------
            RuntimeError: feedback is enabled but no feedback vector is available.

        Returns
        -------
            numpy.ndarray
                Next state x(t+1)
        """

        # check if the user is trying to add empty feedback
        if self.Wfb is not None and feedback is None:
            raise RuntimeError("Missing a feedback vector.")

        if noise_generator is None:
            noise_generator = default_rng()

        # first initialize the current state of the ESN
        if last_state is None:
            x = np.zeros((self.N, 1), dtype=self.typefloat)
        else:
            x = np.asarray(last_state)

        # add bias
        if self.in_bias:
            u = np.hstack((1, single_input.flatten())).astype(self.typefloat)
        else:
            u = np.asarray(single_input)

        # prepare noise sequence
        noise_in = self.noise_in * noise_generator.uniform(-1, 1, size=(self.dim_inp, 1))
        noise_rc = self.noise_rc * noise_generator.uniform(-1, 1, size=(self.N, 1))

        # linear transformation
        x1 = np.dot(self.Win, u.reshape(self.dim_inp, 1) + noise_in) \
            + self.W.dot(x)

        # add feedback if requested
        if self.Wfb is not None:
            noise_out = self.noise_out * noise_generator.uniform(-1, 1,
                                                                 size=(self.dim_out, 1))
            fb = self.fbfunc(np.asarray(feedback)).reshape(self.dim_out, 1)
            x1 += np.dot(self.Wfb, fb + noise_out)

        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*(np.tanh(x1) + noise_rc)

        # return the next state computed
        return x1

    def _compute_states(self,
                        input: np.ndarray,
                        forced_teacher: np.ndarray = None,
                        init_state: np.ndarray = None,
                        init_fb: np.ndarray = None,
                        wash_nr_time_step: int = 0,
                        seed: int = None,
                        input_id: int = None,
                        memmap: np.memmap = None,
                        input_pos: int = None,
                        verbose: bool = False
                        ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Compute all states generated from a single sequence of inputs.

        Parameters
            input {np.ndarray} -- Sequence of inputs.

        Keyword Arguments:
            forced_teacher {np.ndarray} -- Ground truth vectors to use as feedback
                                           during training, if feedback is enabled.
                                           (default: {None})
            init_state {np.ndarray} -- Initialization vector for states.
            (default: {None})
            init_fb {np.ndarray} -- Initialization vector for feedback.
            (default: {None})
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            input_id {int} -- Index of the input in the queue. Used for parallelization
                              of computations. (default: {None})

        Raises:
            RuntimeError: raised if no teachers are specifiyed for training
            with feedback.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray] -- Index of the
            input in queue
            and computed states, or just states if no index is provided.
        """
        if self.Wfb is not None and forced_teacher is None and self.Wout is None:
            raise RuntimeError("Impossible to use feedback without readout"
                               "matrix or teacher forcing.")

        # to track successives internal states of the reservoir
        states = np.zeros((self.N, len(input)-wash_nr_time_step), dtype=self.typefloat)

        # if a feedback matrix is available, feedback will be set to 0 or to
        # a specific value.
        if self.Wfb is not None:
            if init_fb is None:
                last_feedback = np.zeros((self.dim_out, 1), dtype=self.typefloat)
            else:
                last_feedback = init_fb.copy()
        else:
            last_feedback = None

        # State is initialized to 0 or to a specific value.
        if init_state is None:
            current_state = np.zeros((self.N, 1),dtype=self.typefloat)
        else:
            current_state = init_state.copy().reshape(-1, 1)

        # random generator for training/running with additive noise
        rg = default_rng(seed)

        # for each time step in the input
        for t in tqdm(range(input.shape[0])):
            # compute next state from current state
            current_state = self._get_next_state(input[t, :],
                                                 feedback=last_feedback,
                                                 last_state=current_state,
                                                 noise_generator=rg)

            # compute last feedback
            if self.Wfb is not None:
                # during training outputs are equal to teachers for feedback
                if forced_teacher is not None:
                    last_feedback = forced_teacher[t,:].reshape(
                        forced_teacher.shape[1], 1).astype(self.typefloat)
                # feedback of outputs, computed with Wout
                else:
                    last_feedback = np.dot(self.Wout,
                                           np.vstack((1, current_state))).astype(self.typefloat)
                last_feedback = last_feedback.reshape(self.dim_out, 1)

            # will track all internal states during inference, and only the
            # states after wash_nr_time_step during training.
            if t >= wash_nr_time_step:
                states[:, t-wash_nr_time_step] = current_state.reshape(-1,).astype(self.typefloat)

        if input_id is None:
            return 0, states

        if memmap is not None:
            memmap[:, input_pos[0]: input_pos[1]] = np.vstack(
                (np.ones((1, states.shape[1]), dtype=self.typefloat),
                 states))

        return input_id, states

    def compute_all_states(self,
                           inputs: Sequence[np.ndarray],
                           forced_teachers: Sequence[np.ndarray] = None,
                           init_state: np.ndarray = None,
                           init_fb: np.ndarray = None,
                           wash_nr_time_step: int = 0,
                           workers: int = -1,
                           backend: str = "threading",
                           seed: int = None,
                           verbose: bool = True,
                           memmap: np.memmap = None,
                           ) -> Sequence[np.ndarray]:
        """Compute all states generated from sequences of inputs.

        Parameters
        ----------
            inputs: list or array of numpy.array
                All sequences of inputs used for internal state computation.
                Note that it should always be a list of sequences, i.e. if
                only one sequence of inputs is used, it should be alone in a
                list

            forced_teachers: list or array of numpy.array, optional
                Sequence of ground truths, for computation with feedback without
                any trained readout. Note that is should always be a list of
                sequences of the same length than the `inputs`, i.e. if
                only one sequence of inputs is used, it should be alone in a
                list.

            init_state: np.ndarray, optional
                State initialization vector for all inputs. By default, state
                is initialized at 0.

            init_fb: np.ndarray, optional
                Feedback initialization vector for all inputs, if feedback is
                enabled. By default, feedback is initialized at 0.

            wash_nr_time_step: int, optional
                Number of states to consider as transient when training, and to
                remove when computing the readout weights. By default, no states are
                removed.

            workers: int, optional
                If n >= 1, will enable parallelization of states computation with
                n threads/processes, if possible. If n = -1, will use all available
                resources for parallelization. By default, -1.

            backend: {"threadings", "multiprocessing", "loky"}, optional
                Backend used for parallelization of states computations.
                By default, "threading".

            verbose: bool, optional

        Returns:
            list of np.ndarray
                All computed states.
        """

        # initialization of workers
        loop = joblib.Parallel(n_jobs=workers, backend=backend)
        delayed_states = joblib.delayed(self._compute_states)

        # generation of seed sequence
        # each seed in the sequence is independant from the others
        # i.e. this is thread safe random generation.
        # Used for noisy training and running of reservoirs
        ss = SeedSequence(seed)
        # one independant seed per sequence
        child_seeds = ss.spawn(len(inputs))

        # progress bar if needed
        if verbose:
            track = tqdm
        else:
            def track(x, text):
                return x

        inputs_ends = np.cumsum([i.shape[0] for i in inputs])
        inputs_starts = [end - i.shape[0] for i, end in zip(inputs, inputs_ends)]

        # no feedback training or running
        if forced_teachers is None:
            all_states = loop(delayed_states(inputs[i],
                                             wash_nr_time_step=wash_nr_time_step,
                                             input_id=i,
                                             init_state=init_state,
                                             init_fb=init_fb,
                                             memmap=memmap,
                                             input_pos=(
                                                 inputs_starts[i],
                                                 inputs_ends[i]),
                                             seed=child_seeds[i],
                                             verbose=verbose)
                              for i in track(range(len(inputs)), "Computing states"))
        # feedback training
        else:
            all_states = loop(delayed_states(inputs[i],
                                             forced_teachers[i],
                                             wash_nr_time_step=wash_nr_time_step,
                                             input_id=i,
                                             init_state=init_state,
                                             init_fb=init_fb,
                                             memmap=memmap,
                                             input_pos=(
                                                 inputs_starts[i],
                                                 inputs_ends[i]),
                                             seed=child_seeds[i],
                                             verbose=verbose)
                              for i in track(range(len(inputs)), "Computing states"))

        # input ids are used to make sure that the returned states are in the same order
        # as inputs, because parallelization can change this order.
        return [s[1] for s in sorted(all_states, key=lambda x: x[0])]

    def compute_outputs(self,
                        states: Sequence[np.ndarray],
                        verbose: bool = False
                        ) -> Sequence[np.ndarray]:
        """Compute all readouts of a given sequence of states,
        when a readout matrix is available (i.e. after training).

        Parameters
        ----------
            states: list of numpy.array
                All sequences of states used for readout.

            verbose: bool, optional

        Raises
        ------
            RuntimeError: no readout matrix Wout is available.
            Consider training model first, or load an existing matrix.

        Returns
        -------
            list of numpy.arrays
                All outputs of readout matrix.
        """
        # because all states and readouts will be concatenated,
        # first save the indexes of each inputs states in the concatenated vector.
        if self.Wout is not None:

            if verbose:
                print("Computing outputs...")
                tic = time.time()

            outputs = [None] * len(states)
            for i, s in enumerate(states):
                x = np.vstack((np.ones((s.shape[1],), dtype=self.typefloat), s))
                y = np.dot(self.Wout, x).astype(self.typefloat)
                outputs[i] = y

            if verbose:
                toc = time.time()
                print(f"Outputs computed! (in {toc - tic}sec)")

            return outputs

        else:
            raise RuntimeError("Impossible to compute outputs: "
                               "no readout matrix available.")

    def fit_readout(self,
                    states: Sequence,
                    teachers: Sequence,
                    reg_model: Callable = None,
                    ridge: float = None,
                    force_pinv: bool = False,
                    verbose: bool = False,
                    use_memmap: bool = False) -> np.ndarray:
        """Compute a readout matrix by fitting the states computed by the ESN
        to the expected values, using the regression model defined
        in the ESN.

        Parameters
        ----------
            states: list of numpy.ndarray
                All states computed.

            teachers: list of numpy.ndarray
                All ground truth vectors.

            reg_model: scikit-learn regression model, optional
                A scikit-learn regression model to use for readout
                weights computation.

            ridge: float, optional
                Use Tikhonov regression for readout weights computation
                and set regularization parameter to the parameter value.

            force_pinv: bool, optional
                Overwrite all previous parameters and
                force computation of readout using pseudo-inversion.

            verbose: bool, optional

        Returns
        -------
            numpy.ndarray
                Readout matrix.
        """
        # switch the regression model used at instanciation if needed.
        # WARNING: this change won't be saved by the save function.
        if (ridge is not None) or (reg_model is not None):
            reg_model = self._get_regression_model(ridge, reg_model)
        elif force_pinv:
            reg_model = self._get_regression_model(None, None)
        else:
            reg_model = self.reg_model

        # check if network responses are valid
        _check_values(array_or_list=states, value=None)

        if verbose:
            tic = time.time()
            print("Linear regression...")
        # concatenate the lists (along timestep axis)
        if not use_memmap:
            X = np.hstack(states).astype(self.typefloat)
            Y = np.hstack(teachers).astype(self.typefloat)

            # Adding ones for regression with bias b in (y = a*x + b)
            X = np.vstack((np.ones((1, X.shape[1]), dtype=self.typefloat), X))

            # Building Wout with a linear regression model.
            # saving the output matrix in the ESN object for later use
            Wout = reg_model(X, Y)

        else:
            Wout = reg_model(states, teachers)
            del states
            del teachers

        if verbose:
            toc = time.time()
            print(f"Linear regression done! (in {toc - tic} sec)")

        # return readout matrix
        return Wout

    def train(self,
              inputs: Sequence[np.ndarray],
              teachers: Sequence[np.ndarray],
              wash_nr_time_step: int = 0,
              workers: int = -1,
              backend: str = "threading",
              seed: int = None,
              verbose: bool = False,
              use_memmap: bool = False) -> Sequence[np.ndarray]:
        """Train the ESN model on set of input sequences.

        Parameters
        ----------
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
            wash_nr_time_step: int
                Number of states to considered as transient when training. Transient
                states will be discarded when computing readout matrix. By default,
                no states are removes.
            workers: int, optional
                If n >= 1, will enable parallelization of states computation with
                n threads/processes, if possible. If n = -1, will use all available
                resources for parallelization. By default, -1.
            backend: {"threadings", "multiprocessing", "loky"}, optional
                Backend used for parallelization of states computations.
                By default, "threading".
            verbose: bool, optional

        Returns
        -------
            list of numpy.ndarray
                All states computed, for all inputs.

        Note
        ----
            If only one input sequence is provided ("continuous time" inputs),
            workers should be 1, because parallelization is impossible. In other
            cases, if using large NumPy arrays during computation (which is often
            the case), prefer using `threading` backend to avoid huge overhead.
            Multiprocess is a good idea only in very specific cases, and this code
            is not (yet) well suited for this.
        """
        # autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs, outputs=teachers)

        steps = np.sum([i.shape[0] for i in inputs])
        if verbose:
            print(f"Training on {len(inputs)} inputs ({steps} steps) "
                  f"-- wash: {wash_nr_time_step} steps")

        memstates = None
        if use_memmap:
            memstates = np.memmap(self._tempstates, dtype=self.typefloat, mode="w+",
                                  shape=(self.N + 1, steps))

        seed = seed if seed is not None else self.seed

        # compute all states
        all_states = self.compute_all_states(inputs,
                                             forced_teachers=teachers,
                                             wash_nr_time_step=wash_nr_time_step,
                                             workers=workers,
                                             backend=backend,
                                             verbose=verbose,
                                             memmap=memstates,
                                             seed=seed)

        all_teachers = [t[wash_nr_time_step:].T for t in teachers]

        # compute readout matrix
        if use_memmap:
            memteachers = np.memmap(self._tempteach,
                                    dtype=self.typefloat,
                                    mode="w+",
                                    shape=(
                                        all_teachers[0].shape[0],
                                        steps))

            memteachers[:] = np.hstack(all_teachers)

            self.Wout = self.fit_readout(memstates,
                                         memteachers,
                                         use_memmap=True,
                                         verbose=verbose)
        else:
            self.Wout = self.fit_readout(all_states,
                                         all_teachers,
                                         verbose=verbose)

        # save the expected dimension of outputs
        self.dim_out = self.Wout.shape[0]

        # return all internal states
        return [st.T for st in all_states]

    def run(self,
            inputs: Sequence[np.ndarray],
            init_state: np.ndarray = None,
            init_fb: np.ndarray = None,
            workers: int = -1,
            backend: str = "threading",
            seed: int = None,
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

            init_state: numpy.ndarray
                State initialization vector for all inputs. By default, internal
                state of the reservoir is initialized to 0.

            init_fb: numpy.ndarray
                Feedback initialization vector for all inputs, if feedback is
                enabled. By default, feedback is initialized to 0.

           workers: int, optional
                If n >= 1, will enable parallelization of states computation with
                n threads/processes, if possible. If n = -1, will use all available
                resources for parallelization. By default, -1.

            backend: {"threadings", "multiprocessing", "loky"}, optional
                Backend used for parallelization of states computations.
                By default, "threading".

            verbose: bool, optional

        Returns
        -------
            list of numpy.ndarray, list of numpy.ndarray
                All outputs computed from readout and all corresponding internal states,
                for all inputs.

        Note
        ----
            If only one input sequence is provided ("continuous time" inputs),
            workers should be 1, because parallelization is impossible. In other
            cases, if using large NumPy arrays during computation (which is often
            the case), prefer using `threading` backend to avoid huge overhead.
            Multiprocess is a good idea only in very specific cases, and this code
            is not (yet) well suited for this.
        """

        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Running on {len(inputs)} inputs ({steps} steps)")

        # autochecks of inputs
        self._autocheck_io(inputs=inputs)

        seed = seed if seed is not None else self.seed

        all_states = self.compute_all_states(inputs,
                                             init_state=init_state,
                                             init_fb=init_fb,
                                             workers=workers,
                                             backend=backend,
                                             verbose=verbose,
                                             seed=seed)

        all_outputs = self.compute_outputs(all_states)
        # return all_outputs, all_int_states
        return [st.T for st in all_outputs], [st.T for st in all_states]

    def generate(self,
                 nb_timesteps: int,
                 warming_inputs: np.ndarray = None,
                 init_state: np.ndarray = None,
                 init_fb: np.ndarray = None,
                 verbose: bool = False,
                 init_inputs: np.ndarray = None,
                 seed: int = None,
                 return_init: bool = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the ESN on generative mode.

        After the `̀warming_inputs` are consumed, new outputs are
        used as inputs for the next nb_timesteps, i.e. the
        ESN is feeding himself with its own outputs.

        Note that this mode can only work if the ESN is trained
        on a regression task. The outputs of the ESN must be
        the same kind of data as its input.

        To train an ESN on generative mode, use the :py:func:`ESN.train`
        method to train the ESN on a regression task (for
        instance, predict the future data point t+1 of a timeseries
        give the data at time t).

        Parameters
        ----------
            nb_timesteps: int
                Number of timesteps of data to generate
                from the intial input.
            warming_inputs: numpy.ndarray
                Input data used to initiate generative mode.
                This data is meant to "seed" the ESN internal
                states with some real information, before it runs
                on its own created outputs.
            init_state: numpy.ndarray, optional:
                State initialization vector for the reservoir.
                By default, internal state of the reservoir is initialized to 0.
            init_fb: numpy.ndarray, optional
                Feedback initialization vector for the reservoir, if feedback is
                enabled. By default, feedback is initialized to 0.
            verbose: bool, optional
            init_intputs: list of numpy.ndarray, optional
                Same as ``warming_inputs̀``.
                Kept for compatibility with previous version. Deprecated
                since 0.2.2, will be removed soon.
            return_init: bool, optional
                Kept for compatibility with previous version. Deprecated
                since 0.2.2, will be removed soon.

        Returns
        -------
            tuple of numpy.ndarray
                Generated outputs, generated states, warming outputs, warming states

                Generated outputs are the timeseries predicted by the ESN from
                its own predictions over time. Generated states are the
                corresponding internal states.

                Warming outputs are the predictions made by the ESN based on the
                warming inputs passed as parameters. These predictions are prior
                to the generated outputs. Warming states are the corresponding
                internal states. In the case no warming inputs are provided, warming
                outputs and warming states are None.

        """
        if warming_inputs is None and init_state is None and init_inputs is None:
            raise ValueError("at least one of the parameter 'warming_input' "
                             "or 'init_state' must not be None. Impossible "
                             "to generate from scratch.")

        if return_init is not None:
            warnings.warn("Deprecation warning : return_init parameter "
                          "is deprecated since 0.2.2 and will be removed.")

        # for additive noise in the reservoir
        # 2 separate seeds made from one: one for the warming
        # (if needed), one for the generation
        seed = seed if seed is not None else self.seed
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(2)

        if warming_inputs is not None or init_inputs is not None:
            if init_inputs is not None:
                warnings.warn("Deprecation warning : init_inputs parameter "
                              "is deprecated since 0.2.2 and will be removed. "
                              "Please use warming_inputs instead.")
                warming_inputs = init_inputs

            if verbose:
                print(f"Generating {nb_timesteps} timesteps from "
                      f"{warming_inputs.shape[0]} inputs.")
                print("Computing initial states...")

            _, warming_states = self._compute_states(warming_inputs,
                                                     init_state=init_state,
                                                     init_fb=init_fb,
                                                     seed=child_seeds[0])

            # initial state (at begining of generation)
            s0 = warming_states[:, -1].reshape(-1, 1)
            warming_outputs = self.compute_outputs([warming_states])[0]
            # intial input (at begining of generation)
            u1 = warming_outputs[:, -1].reshape(1, -1)

            if init_fb is not None:
                # initial feedback (at begining of generation)
                fb0 = warming_outputs[:, -2].reshape(1, -1)
            else:
                fb0 = None
            warming_outputs = warming_outputs.T
            warming_states = warming_states.T
        else:
            warming_outputs, warming_states = None, None
            # time is often first axis but compute_outputs await
            # for time in second axis, so the reshape :
            s0 = init_state.reshape(-1, 1)

            if init_fb is not None:
                fb0 = init_fb.reshape(-1, 1)
            else:
                fb0 = None

            u1 = self.compute_outputs([s0])[0][:, -1].reshape(1, -1)

        states = np.zeros((nb_timesteps, self.N))
        outputs = np.zeros((nb_timesteps, self.dim_out))

        if verbose:
            track = tqdm
        else:
            def track(x, text): return x

        # for additive noise in the reservoir
        rg = default_rng(child_seeds[1])

        for i in track(range(nb_timesteps), "Generating"):
            # from new input u1 and previous state s0
            # compute next state s1 -> s0
            s1 = self._get_next_state(single_input=u1,
                                      feedback=fb0,
                                      last_state=s0,
                                      noise_generator=rg)

            s0 = s1[:, -1].reshape(-1, 1)
            states[i, :] = s0.flatten()

            if fb0 is not None:
                fb0 = u1.copy()

            # from new state s1 compute next input u2 -> u1
            u1 = self.compute_outputs([s0])[0][:, -1].reshape(1, -1)
            outputs[i, :] = u1.flatten()

        return outputs, states, warming_outputs, warming_states

    def save(self, directory: str):
        """Save the ESN to disk.

        Parameters
        ----------
            directory: str or Path
                Directory where to save the model.
        """
        _save(self, directory)
