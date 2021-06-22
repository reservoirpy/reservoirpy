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
from pathlib import Path
from functools import partial

import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.random import default_rng, SeedSequence, Generator

from .utils.parallel import ParallelProgressQueue, get_joblib_backend, memmap, clean_tempfile
from .utils.validation import _check_values, add_bias, check_input_lists
from .utils.save import _save
from .regression_models import RidgeRegression, SklearnLinearModel


def _get_offline_model(ridge: float = 0.0,
                       sklearn_model: Callable = None,
                       dtype=np.float64):
    if ridge > 0.0 and sklearn_model is not None:
        raise ValueError("Parameters 'ridge' and 'sklearn_model' can not be "
                         "defined at the same time.")
    elif sklearn_model is not None:
        return SklearnLinearModel(sklearn_model, dtype=dtype)
    else:
        return RidgeRegression(ridge, dtype=dtype)


def _parallelize(esn,
                 func,
                 workers,
                 lengths,
                 return_states,
                 pbar_text=None,
                 verbose=False,
                 **func_kwargs):

    workers = min(len(lengths), workers)
    backend = get_joblib_backend() if workers > 1 or workers == -1 else "sequential"

    steps = np.sum(lengths)
    ends = np.cumsum(lengths)
    starts = ends - np.asarray(lengths)

    fn_kwargs = ({k: func_kwargs[k][i] for k in func_kwargs.keys()}
                 for i in range(len(lengths)))

    states = None
    if return_states:
        shape = (steps, esn.N)
        states = memmap(shape, dtype=esn.typefloat, caller=esn)

    with ParallelProgressQueue(total=steps, text=pbar_text, verbose=verbose) as pbar:

        func = partial(func, pbar=pbar)

        with Parallel(backend=backend, n_jobs=workers) as parallel:

            def func_wrapper(states, start_pos, end_pos, *args, **kwargs):
                s = func(*args, **kwargs)

                out = None
                # if function returns states and outputs
                if hasattr(s, "__len__") and len(s) == 2:
                    out = s[0]  # outputs are always returned first
                    s = s[1]

                if return_states:
                    states[start_pos:end_pos] = s[:]

                return out

            outputs = parallel(delayed(func_wrapper)(states, start, end, **kwargs)
                               for start, end, kwargs in zip(starts, ends, fn_kwargs))

    if return_states:
        states = [np.array(states[start:end])
                  for start, end in zip(starts, ends)]

    clean_tempfile(esn)

    return outputs, states


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
        dim_in: int
            Input dimension
        N: int
            Number of neuronal units

    See also
    --------
        reservoirpy.ESNOnline for ESN with online learning using FORCE.

    """
    def __init__(self,
                 lr: float,
                 W: np.ndarray,
                 Win: np.ndarray,
                 input_bias: bool = True,
                 reg_model: Callable = None,
                 ridge: float = 0.0,
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
        self.dim_in = self.Win.shape[1]
        if self.in_bias:
            self.dim_in = self.dim_in - 1

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

        self.model = _get_offline_model(ridge, reg_model, dtype=typefloat)

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

    @property
    def ridge(self):
        return getattr(self.model, "ridge", None)

    @ridge.setter
    def ridge(self, value):
        if hasattr(self.model, "ridge"):
            self.model.ridge = value

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
            x = self.zero_state()
        else:
            x = np.asarray(last_state, dtype=self.typefloat).reshape(1, -1)

        u = np.asarray(single_input, dtype=self.typefloat).reshape(1, -1)
        # add bias
        if self.in_bias:
            u = add_bias(u)

        # prepare noise sequence
        noise_in = self.noise_in * noise_generator.uniform(-1, 1, size=u.shape)
        noise_rc = self.noise_rc * noise_generator.uniform(-1, 1, size=x.shape)

        # linear transformation
        x1 = (u + noise_in) @ self.Win.T + x @ self.W

        # add feedback if requested
        if self.Wfb is not None:
            noise_out = self.noise_out * noise_generator.uniform(-1, 1,
                                                                 size=feedback.shape)
            fb = self.fbfunc(np.asarray(feedback)).reshape(1, -1)
            x1 += (fb + noise_out) @ self.Wfb.T

        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr) * x + self.lr * (np.tanh(x1)+noise_rc)

        # return the next state computed
        return x1

    def compute_states(self,
                       input: np.ndarray,
                       forced_teacher: np.ndarray = None,
                       init_state: np.ndarray = None,
                       init_fb: np.ndarray = None,
                       seed: int = None,
                       verbose: bool = False,
                       **kwargs
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

        check_input_lists(input, self.dim_in, forced_teacher, self.dim_out)

        # to track successives internal states of the reservoir
        states = np.zeros((len(input), self.N), dtype=self.typefloat)

        # if a feedback matrix is available, feedback will be set to 0 or to
        # a specific value.
        if init_fb is not None:
            last_feedback = init_fb.copy().reshape(1, -1)
        else:
            last_feedback = self.zero_feedback()

        # State is initialized to 0 or to a specific value.
        if init_state is None:
            current_state = self.zero_state()
        else:
            current_state = init_state.copy().reshape(1, -1)

        # random generator for training/running with additive noise
        rng = default_rng(seed)

        pbar = None
        if kwargs.get("pbar") is not None:
            pbar = kwargs.get("pbar")
        elif verbose is True:
            pbar = tqdm(total=input.shape[0])

        # for each time step in the input
        for t in range(input.shape[0]):
            # compute next state from current state
            current_state = self._get_next_state(input[t, :],
                                                 feedback=last_feedback,
                                                 last_state=current_state,
                                                 noise_generator=rng)

            # compute last feedback
            if self.Wfb is not None:
                # during training outputs are equal to teachers for feedback
                if forced_teacher is not None:
                    last_feedback = forced_teacher[t, :]
                # feedback of outputs, computed with Wout
                else:
                    last_feedback = (add_bias(current_state) @ self.Wout.T)

            states[t, :] = current_state

            if pbar is not None:
                pbar.update(1)

        return states

    def compute_all_states(self,
                           inputs: Sequence[np.ndarray],
                           forced_teachers: Sequence[np.ndarray] = None,
                           init_state: np.ndarray = None,
                           init_fb: np.ndarray = None,
                           workers: int = -1,
                           seed: int = None,
                           verbose: bool = False,
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
        inputs, forced_teachers = check_input_lists(inputs, self.dim_in,
                                                    forced_teachers, self.dim_out)

        workers = min(workers, len(inputs))

        backend = "sequential"
        if workers > 1 or workers == -1:
            backend = get_joblib_backend()

        if forced_teachers is None:
            forced_teachers = [None] * len(inputs)

        compute_states = partial(self.compute_states,
                                 init_state=init_state,
                                 init_fb=init_fb)

        steps = np.sum([i.shape[0] for i in inputs])

        progress = ParallelProgressQueue(total=steps,
                                         text="States",
                                         verbose=verbose)
        with progress as pbar:
            with Parallel(backend=backend, n_jobs=workers) as parallel:
                states = parallel(delayed(
                    partial(compute_states, pbar=pbar, seed=seed))(x, t)
                                  for x, t in zip(inputs, forced_teachers))

        return states

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
        states, _ = check_input_lists(states, self.N)
        # because all states and readouts will be concatenated,
        # first save the indexes of each inputs states in the concatenated vector.
        if self.Wout is not None:

            if verbose:
                print("Computing outputs...")
                tic = time.time()

            outputs = [None] * len(states)
            for i, s in enumerate(states):
                x = add_bias(s)
                y = (x @ self.Wout.T).astype(self.typefloat)
                outputs[i] = y

            if verbose:
                toc = time.time()
                print(f"Outputs computed! (in {toc - tic}sec)")

            return outputs

        else:
            raise RuntimeError("Impossible to compute outputs: "
                               "no readout matrix available. "
                               "Train the network first.")

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
        states, teachers = check_input_lists(states, self.N, teachers, self.dim_out)

        # switch the regression model used at instanciation if needed.
        # WARNING: this change won't be saved by the save function.
        if (ridge is not None) or (reg_model is not None):
            offline_model = _get_offline_model(ridge, reg_model, dtype=self.typefloat)
        elif force_pinv:
            offline_model = _get_offline_model(ridge=0.0)
        else:
            offline_model = self.model

        # check if network responses are valid
        _check_values(array_or_list=states, value=None)

        if verbose:
            tic = time.time()
            print("Linear regression...")

        self.Wout = offline_model.fit(X=states, Y=teachers)

        if verbose:
            toc = time.time()
            print(f"Linear regression done! (in {toc - tic} sec)")

        return self.Wout

    def train(self,
              inputs: Union[Sequence[np.ndarray], np.ndarray],
              teachers: Union[Sequence[np.ndarray], np.ndarray],
              washout: int = 0,
              workers: int = -1,
              seed: int = None,
              verbose: bool = False,
              return_states: bool = False) -> Sequence[np.ndarray]:
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
        inputs, teachers = check_input_lists(inputs, self.dim_in,
                                             teachers, self.dim_out)

        self.dim_out = teachers[0].shape[1]
        self.model.initialize(self.N, self.dim_out)

        lengths = [i.shape[0] for i in inputs]
        steps = sum(lengths)

        if verbose:
            print(f"Training on {len(inputs)} inputs ({steps} steps) "
                  f"-- wash: {washout} steps")

        def train_fn(*, x, y, pbar):
            s = self.compute_states(x, y, seed=seed, pbar=pbar)
            self.model.partial_fit(s[washout:], y)  # increment X.X^T and Y.X^T
                                                    # or save data for sklearn fit
            return s

        _, states = _parallelize(self, train_fn, workers, lengths, return_states,
                                 pbar_text="Train", verbose=verbose,
                                 x=inputs, y=teachers)

        self.Wout = self.model.fit()  # perform Y.X^T.(X.X^T + ridge)^-1
                                              # or sklearn fit

        return states

    def run(self,
            inputs: Sequence[np.ndarray],
            init_state: np.ndarray = None,
            init_fb: np.ndarray = None,
            workers: int = -1,
            return_states=False,
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
        # autochecks of inputs and outputs
        inputs, _ = check_input_lists(inputs, self.dim_in)

        lengths = [i.shape[0] for i in inputs]
        steps = sum(lengths)

        if verbose:
            print(f"Running on {len(inputs)} inputs ({steps} steps)")

        def run_fn(*, x, pbar):
            s = self.compute_states(x,
                                    init_state=init_state,
                                    init_fb=init_fb,
                                    seed=seed,
                                    pbar=pbar)

            out = self.compute_outputs([s])[0]

            return out, s

        outputs, states = _parallelize(self, run_fn, workers, lengths, return_states,
                                       pbar_text="Run", verbose=verbose,
                                       x=inputs)

        return outputs, states

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

            warming_states = self.compute_states(warming_inputs,
                                                 init_state=init_state,
                                                 init_fb=init_fb,
                                                 seed=child_seeds[0])

            # initial state (at begining of generation)
            s0 = warming_states[-1, :].reshape(1, -1)
            warming_outputs = self.compute_outputs([warming_states])[0]
            # intial input (at begining of generation)
            u1 = warming_outputs[-1, :].reshape(1, -1)

            if init_fb is not None:
                # initial feedback (at begining of generation)
                fb0 = warming_outputs[-2, :].reshape(1, -1)
            else:
                fb0 = None

        else:
            warming_outputs, warming_states = None, None
            # time is often first axis but compute_outputs await
            # for time in second axis, so the reshape :
            s0 = init_state.reshape(1, -1)

            if init_fb is not None:
                fb0 = init_fb.reshape(1, -1)
            else:
                fb0 = None

            u1 = self.compute_outputs([s0])[0][-1, :].reshape(1, -1)

        states = np.zeros((nb_timesteps, self.N))
        outputs = np.zeros((nb_timesteps, self.dim_out))

        if verbose:
            track = tqdm
        else:
            def track(x, text): return x

        # for additive noise in the reservoir
        rg = default_rng(child_seeds[1])

        for i in track(range(nb_timesteps), "Generate"):
            # from new input u1 and previous state s0
            # compute next state s1 -> s0
            s1 = self._get_next_state(single_input=u1,
                                      feedback=fb0,
                                      last_state=s0,
                                      noise_generator=rg)

            s0 = s1[-1, :].reshape(1, -1)
            states[i, :] = s0

            if fb0 is not None:
                fb0 = u1.copy()

            # from new state s1 compute next input u2 -> u1
            u1 = self.compute_outputs([s0])[0][-1, :].reshape(1, -1)
            outputs[i, :] = u1

        return outputs, states, warming_outputs, warming_states

    def save(self, directory: str):
        """Save the ESN to disk.

        Parameters
        ----------
            directory: str or Path
                Directory where to save the model.
        """
        _save(self, directory)

    def zero_state(self):
        return np.zeros((1, self.N), dtype=self.typefloat)

    def zero_feedback(self):
        if self.Wfb is not None:
            return np.zeros((1, self.dim_out), dtype=self.typefloat)
        else:
            return None
