import time
import warnings

from typing import Sequence, Tuple, Union
from pathlib import Path
from functools import partial
from abc import ABCMeta

import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.random import default_rng, SeedSequence, Generator

from ..utils.parallel import ParallelProgressQueue, get_joblib_backend, parallelize
from ..utils.validation import add_bias, check_input_lists, check_reservoir_matrices
from .utils.save import _save
from ..types import Weights, Data, Activation


class _ESNBase(metaclass=ABCMeta):

    def __init__(self,
                 W: Weights,
                 Win: Weights,
                 lr: float = 1.0,
                 input_bias: bool = True,
                 activation: Activation = np.tanh,
                 Wfb: Weights = None,
                 fbfunc: Activation = lambda x: x,
                 Wout: Weights = None,
                 noise_in: float = 0.0,
                 noise_rc: float = 0.0,
                 noise_out: float = 0.0,
                 seed: int = None,
                 typefloat: np.dtype = np.float64):

        W, Win, Wout, Wfb = check_reservoir_matrices(W, Win, Wout=Wout,
                                                     Wfb=Wfb, caller=self)

        self._W = W
        self._Win = Win
        self._Wout = Wout
        self._Wfb = Wfb

        self.lr = lr
        self.activation = activation
        self.noise_in = noise_in
        self.noise_rc = noise_rc
        self.noise_out = noise_out
        self.typefloat = typefloat
        self.seed = seed
        self.fbfunc = fbfunc
        self._input_bias = input_bias

        self._N = self._W.shape[0]
        self._dim_in = self._Win.shape[1]
        if self._input_bias:
            self._dim_in = self._dim_in - 1

        if self._Wout is not None:
            self._dim_out = self._Wout.shape[0]
        elif self._Wfb is not None:
            self._dim_out = self._Wfb.shape[1]
        else:
            self._dim_out = None

    def __repr__(self):
        trained = True
        if self.Wout is None:
            trained = False
        fb = self.Wfb is not None
        name = self.__class__.__name__
        out = f"{name}(trained={trained}, feedback={fb}, N={self.N}, " \
              f"lr={self.lr}, input_bias={self.input_bias}, input_dim={self.dim_in})"
        return out

    @property
    def N(self):
        """Number of units."""
        return self._N

    @property
    def dim_in(self):
        """Input dimension."""
        return self._dim_in

    @property
    def dim_out(self):
        """Output (readout) dimension."""
        return self._dim_out

    @property
    def input_bias(self):
        """If True, constant bias is added to inputs."""
        return self._input_bias

    @property
    def use_raw_input(self):
        """If True, raw inputs are concatenated to states before readout."""
        return self._use_raw_input

    @property
    def Win(self):
        """Input weight matrix."""
        return self._Win

    @Win.setter
    def Win(self, matrix):
        _, Win, _, _ = check_reservoir_matrices(self._W, matrix,
                                                self._Wout,
                                                self._Wfb,
                                                caller=self)
        self._Win = Win
        self._dim_in = self._Win.shape[1]
        if self._input_bias:
            self._dim_in = self._dim_in - 1

    @property
    def W(self):
        """Recurrent weight matrix."""
        return self._W

    @W.setter
    def W(self, matrix):
        W, _, _, _ = check_reservoir_matrices(matrix,
                                              self._Win,
                                              self._Wout,
                                              self._Wfb,
                                              caller=self)
        self._W = W
        self._N = self._W.shape[0]

    @property
    def Wfb(self):
        """Feedback weight matrix."""
        return self._Wfb

    @Wfb.setter
    def Wfb(self, matrix):
        _, _, _, Wfb = check_reservoir_matrices(self._W,
                                                self._Win,
                                                self._Wout,
                                                matrix,
                                                caller=self)
        self._Wfb = Wfb
        self._dim_out = self._Wfb.shape[1]

    @property
    def Wout(self):
        """Readout weight matrix."""
        return self._Wout

    @Wout.setter
    def Wout(self, matrix):
        _, _, Wout, _ = check_reservoir_matrices(self._W,
                                                 self._Win,
                                                 matrix,
                                                 self._Wfb,
                                                 caller=self)
        self._Wout = Wout
        self._dim_out = self._Wout.shape[0]

    def zero_state(self):
        """Returns zero state vector."""
        return np.zeros((1, self.N), dtype=self.typefloat)

    def zero_feedback(self):
        """Returns a zero feedabck vector."""
        if self.Wfb is not None:
            return np.zeros((1, self.dim_out), dtype=self.typefloat)
        else:
            return None

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
        if noise_generator is None:
            noise_generator = default_rng()

        # first initialize the current state of the ESN
        if last_state is None:
            x = self.zero_state()
        else:
            x = np.asarray(last_state, dtype=self.typefloat).reshape(1, -1)

        u = np.asarray(single_input, dtype=self.typefloat).reshape(1, -1)
        # add bias
        if self._input_bias:
            u = add_bias(u)

        # prepare noise sequence
        noise_in = self.noise_in * noise_generator.uniform(-1, 1, size=u.shape)
        noise_rc = self.noise_rc * noise_generator.uniform(-1, 1, size=x.shape)

        # linear transformation
        x1 = (u + noise_in) @ self.Win.T + x @ self.W

        # add feedback if requested
        if self.Wfb is not None:
            if feedback is None:
                feedback = self.zero_feedback()

            noise_out = self.noise_out * noise_generator.uniform(-1, 1,
                                                                 size=feedback.shape)
            fb = self.fbfunc(np.asarray(feedback)).reshape(1, -1)
            x1 += (fb + noise_out) @ self.Wfb.T

        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr) * x + self.lr * (self.activation(x1)+noise_rc)

        # return the next state computed
        return x1

    def _compute_states(self,
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
            raise RuntimeError("Impossible to use feedback without readout "
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

            workers: int, optional
                If n >= 1, will enable parallelization of states computation with
                n threads/processes, if possible. If n = -1, will use all available
                resources for parallelization. By default, -1.

            verbose: bool, optional

        Returns
        -------
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

        compute_states = partial(self._compute_states,
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
                        states: Data,
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

    def train(self,
              inputs: Data,
              teachers: Data,
              washout: int = 0,
              workers: int = -1,
              return_states: bool = False,
              seed: int = None,
              verbose: bool = False) -> Sequence[np.ndarray]:
        raise NotImplementedError

    def run(self,
            inputs: Data,
            init_state: np.ndarray = None,
            init_fb: np.ndarray = None,
            workers: int = -1,
            return_states=False,
            backend=None,
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

            return_states: bool, False by default
                If `True`, the function will return all the internal states computed
                during the training. Be warned that this may be too heavy for the
                memory of your computer.

            verbose: bool, optional
            backend:
                kept for compatibility with previous versions.

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
            s = self._compute_states(x,
                                     init_state=init_state,
                                     init_fb=init_fb,
                                     seed=seed,
                                     pbar=pbar)

            out = self.compute_outputs([s])[0]

            return out, s

        outputs, states = parallelize(self, run_fn, workers, lengths, return_states,
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

            warming_states = self._compute_states(warming_inputs,
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
