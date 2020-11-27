# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""reservoirpy/ESN

Simple, parallelizable implementation of Echo State Networks.

@author: Xavier HINAUT
xavier.hinaut@inria.fr
Copyright Xavier Hinaut 2018

I would like to thank Mantas Lukosevicius for his code that was used as inspiration for this code:
http://minds.jacobs-university.de/mantas/code
"""
import time
import warnings
from typing import Sequence, Callable, Tuple, Union, Dict

import joblib
import numpy as np
from scipy import linalg
from tqdm import tqdm

from .utils import check_values, _save
from .regression_models import sklearn_linear_model
from .regression_models import ridge_linear_model
from .regression_models import pseudo_inverse_linear_model


# TODO: base class for ESN objects
# TODO: add logging function, delete some prints
class ESN(object):

    # TODO: instanciation of matrices within the ESN object
    def __init__(self,
                 lr: float,
                 W: np.ndarray,
                 Win: np.ndarray,
                 input_bias: bool = True,
                 reg_model: Callable = None,
                 ridge: float = None,
                 Wfb: np.ndarray = None,
                 fbfunc: Callable = None,
                 typefloat: np.dtype = np.float64):
        """Base class of Echo State Networks

        Arguments:
            lr {float} -- Leaking rate
            W {np.ndarray} -- Reservoir weights matrix
            Win {np.ndarray} -- Input weights matrix

        Keyword Arguments:
            input_bias {bool} -- If True, will add a constant bias
                                 to the input vector (default: {True})
            reg_model {Callable} -- A scikit-learn linear model function to use for regression. Should
                                   be None if ridge is used. (default: {None})
            ridge {float} -- Ridge regularization coefficient for Tikonov regression. Should be None
                             if reg_model is used. (default: {None})
            Wfb {np.array} -- Feedback weights matrix. (default: {None})
            fbfunc {Callable} -- Feedback activation function. (default: {None})
            typefloat {np.dtype} -- Float precision to use (default: {np.float32})

        Raises:
            ValueError: If a feedback matrix is passed without activation function.
            NotImplementedError: If trying to set input_bias to False. This is not
                                 implemented yet.
        """

        self.W = W
        self.Win = Win
        # output weights matrix. must be learnt through training.
        self.Wout = None
        self.Wfb = Wfb

        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        self._autocheck_nan()

        self.N = self.W.shape[1]  # number of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1]  # dimension of inputs (including the bias at 1)
        self.dim_out = None
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1]  # dimension of outputs

        self.typefloat = typefloat
        self.lr = lr  # leaking rate

        self.reg_model = self._get_regression_model(ridge, reg_model)
        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError(f"If a feedback matrix is provided, \
                fbfunc must be a callable object, not {self.fbfunc}.")

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

    # TODO: simplify this with a dict
    def _get_regression_model(self,
                              ridge: float = None,
                              sklearn_model: Callable=None):
        """Set the type of regression used in the model. All regression models available
        for now are described in reservoipy.regression_models:
            - any scikit-learn linear regression model (like Lasso or Ridge)
            - Tikhonov linear regression (l1 regularization)
            - Solving system with pseudo-inverse matrix
        Keyword Arguments:
            ridge {[float]} -- Ridge regularization coefficient. (default: {None})
            sklearn_model {[Callable]} -- scikit-learn regression model to use. (default: {None})

        Raises:
            ValueError: if ridge and scikit-learn models are requested at the same time.

        Returns:
            [Callable] -- A linear regression function.
        """
        if ridge is not None and sklearn_model is not None:
            raise ValueError("ridge and sklearn_model can't be defined at the same time.")

        elif ridge is not None:
            self.ridge = ridge
            return ridge_linear_model(self.ridge)

        elif sklearn_model is not None:
            self.sklearn_model = sklearn_model
            return sklearn_linear_model(self.sklearn_model)

        else:
            return pseudo_inverse_linear_model()

    def _autocheck_nan(self):
        """ Auto-check to see if some important variables do not have a problem (e.g. NAN values). """
        #assert np.isnan(self.W).any() == False, "W matrix should not contain NaN values."
        assert np.isnan(self.Win).any() == False, \
            "Win matrix should not contain NaN values."
        if self.Wfb is not None:
            assert np.isnan(self.Wfb).any() == False, \
                "Wfb matrix should not contain NaN values."

    # TODO: better check, clearer infos
    def _autocheck_dimensions(self):
        """ Auto-check to see if ESN matrices have correct dimensions."""
        # W dimensions check list
        assert len(self.W.shape) == 2, f"W shape should be (N, N) but is {self.W.shape}."
        assert self.W.shape[0] == self.W.shape[1], f"W shape should be (N, N) but is {self.W.shape}."

        # Win dimensions check list
        assert len(self.Win.shape) == 2, f"Win shape should be (N, input) but is {self.Win.shape}."
        err = f"Win shape should be ({self.W.shape[1]}, input) but is {self.Win.shape}."
        assert self.Win.shape[0] == self.W.shape[0], err

    # TODO: better check, clearer infos
    def _autocheck_io(self,
                      inputs,
                      outputs = None):

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
                        last_state: np.ndarray = None) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t), compute the state vector x(t+1).

        Arguments:
            single_input {np.ndarray} -- Input vector u(t).

        Keyword Arguments:
            feedback {np.ndarray} -- Feedback vector if enabled. (default: {None})
            last_state {np.ndarray} -- Current state to update x(t). If None,
                                       state is initialized to 0. (default: {None})

        Raises:
            RuntimeError: feedback is enabled but no feedback vector is available.

        Returns:
            np.ndarray -- Next state x(t+1).
        """

        # check if the user is trying to add empty feedback
        if self.Wfb is not None and feedback is None:
            raise RuntimeError("Missing a feedback vector.")

        # warn if the user is adding a feedback vector when feedback is not available
        # (might have forgotten the feedback weights matrix)
        if self.Wfb is None and feedback is not None:
            warnings.warn("Feedback vector should not be passed to update_state if no feedback matrix is provided.", UserWarning)

        # first initialize the current state of the ESN
        if last_state is None:
            x = np.zeros((self.N,1),dtype=self.typefloat)
            warnings.warn("No previous state was passed for computation of next state. Will assume a 0 vector as initial state to compute the update.", UserWarning)
        else:
            x = last_state

        # add bias
        if self.in_bias:
            u = np.hstack((1, single_input)).astype(self.typefloat)
        else:
            u = single_input

        # linear transformation
        x1 = np.dot(self.Win, u.reshape(self.dim_inp, 1)) \
            + self.W.dot(x)

        # add feedback if requested
        if self.Wfb is not None:
            x1 += np.dot(self.Wfb, self.fbfunc(feedback))

        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*np.tanh(x1)

        # return the next state computed
        return x1

    def _compute_states(self,
                        input: np.ndarray,
                        forced_teacher: np.ndarray = None,
                        init_state: np.ndarray = None,
                        init_fb: np.ndarray = None,
                        wash_nr_time_step: int = 0,
                        input_id: int = None
                        )-> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Compute all states generated from a single sequence of inputs.

        Arguments:
            input {np.ndarray} -- Sequence of inputs.

        Keyword Arguments:
            forced_teacher {np.ndarray} -- Ground truth vectors to use as feedback
                                           during training, if feedback is enabled.
                                           (default: {None})
            init_state {np.ndarray} -- Initialization vector for states. (default: {None})
            init_fb {np.ndarray} -- Initialization vector for feedback. (default: {None})
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            input_id {int} -- Index of the input in the queue. Used for parallelization
                              of computations. (default: {None})

        Raises:
            RuntimeError: raised if no teachers are specifiyed for training with feedback.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray] -- Index of the input in queue
            and computed states, or just states if no index is provided.
        """

        if self.Wfb is not None and forced_teacher is None and self.Wout is None:
            raise RuntimeError("Impossible to use feedback without readout matrix or teacher forcing.")

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

        # for each time step in the input
        for t in range(input.shape[0]):
            # compute next state from current state
            current_state = self._get_next_state(input[t, :], feedback=last_feedback, last_state=current_state)

            # compute last feedback
            if self.Wfb is not None:
                # during training outputs are equal to teachers for feedback
                if forced_teacher is not None:
                    last_feedback = forced_teacher[t,:].reshape(forced_teacher.shape[1], 1).astype(self.typefloat)
                # feedback of outputs, computed with Wout
                else:
                    last_feedback = np.dot(self.Wout, np.vstack((1,current_state))).astype(self.typefloat)
                last_feedback = last_feedback.reshape(self.dim_out, 1)

            # will track all internal states during inference, and only the
            # states after wash_nr_time_step during training.
            if t >= wash_nr_time_step:
                states[:, t-wash_nr_time_step] = current_state.reshape(-1,).astype(self.typefloat)

        if input_id is None:
            return 0, states
        return input_id, states

    def compute_all_states(self,
                           inputs: Sequence[np.ndarray],
                           forced_teachers: Sequence[np.ndarray] = None,
                           init_state: np.ndarray = None,
                           init_fb: np.ndarray = None,
                           wash_nr_time_step: int = 0,
                           workers: int = -1,
                           backend: str = "threading",
                           verbose: bool = True
                           ) -> Sequence[np.ndarray]:
        """Compute all states generated from sequences of inputs.

        Arguments:
            inputs {Sequence[np.ndarray]} -- Sequence of input sequences.

        Keyword Arguments:
            forced_teachers {Sequence[np.ndarray]} -- Sequence of ground truth
                                                      sequences, for training with
                                                      feedback. (default: {None})
            init_state {np.ndarray} -- State initialization vector
                                       for all inputs. (default: {None})
            init_fb {np.ndarray} -- Feedback initialization vector
                                    for all inputs, if feedback is
                                    enabled. (default: {None})
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            workers {int} -- if n >= 1, will enable parallelization of
                             states computation with n threads/processes, if possible.
                             if n = -1, will use all available resources for
                             parallelization. (default: {-1})
            backend {str} -- Backend used for parallelization of
                             states computations. Available backends are
                             `threadings`(recommended, see `train` Note), `multiprocessing`,
                             `loky` (default: {"threading"}).
            verbose {bool} -- if `True`, display progress in stdout.

        Returns:
            Sequence[np.ndarray] -- All computed states.
        """

        # initialization of workers
        loop = joblib.Parallel(n_jobs=workers, backend=backend)
        delayed_states = joblib.delayed(self._compute_states)

        # progress bar if needed
        if verbose:
            track = tqdm
        else:
            def track(x, text):
                return x

        # no feedback training or running
        if forced_teachers is None:
            all_states = loop(delayed_states(inputs[i], wash_nr_time_step=wash_nr_time_step, input_id=i,
                                             init_state=init_state, init_fb=init_fb)
                              for i in track(range(len(inputs)), "Computing states"))
        # feedback training
        else:
            all_states = loop(delayed_states(inputs[i], forced_teachers[i], wash_nr_time_step=wash_nr_time_step,
                                             input_id=i, init_state=init_state, init_fb=init_fb)
                              for i in track(range(len(inputs)), "Computing states"))

        # input ids are used to make sure that the returned states are in the same order
        # as inputs, because parallelization can change this order.
        return [s[1] for s in sorted(all_states, key=lambda x: x[0])]

    def compute_outputs(self,
                        states: Sequence[np.ndarray],
                        verbose: bool = False
                        ) -> Sequence[np.ndarray]:
        """Compute all readouts of a given sequence of states.

        Arguments:
            states {Sequence[np.ndarray]} -- Sequence of states.

        Keyword Arguments:
            verbose {bool} -- Set verbosity.

        Raises:
            RuntimeError: no readout matrix Wout is available.
            Consider training model first, or load an existing matrix.

        Returns:
            Sequence[np.ndarray] -- All outputs of readout matrix.
        """
        # because all states and readouts will be concatenated,
        # first save the indexes of each inputs states in the concatenated vector.
        if self.Wout is not None:
            # idx = [None] * len(states)
            # c = 0
            # for i, s in enumerate(states):
            #     idx[i] = [j for j in range(c, c+s.shape[1])]
            #     c += s.shape[1]

            if verbose:
                print("Computing outputs...")
                tic = time.time()

            # x = np.hstack(states)
            # x = np.vstack((np.ones((x.shape[1],), dtype=self.typefloat), x))

            outputs = [None] * len(states)
            for i, s in enumerate(states):
                x = np.vstack((np.ones((s.shape[1],), dtype=self.typefloat), s))
                y = np.dot(self.Wout, x).astype(self.typefloat)
                outputs[i] = y

            if verbose:
                toc = time.time()
                print(f"Outputs computed! (in {toc - tic}sec)")

            # return separated readouts vectors corresponding to the saved
            # indexes built with input states.
           # return [y[:, i] for i in idx]
            return outputs

        else:
            raise RuntimeError("Impossible to compute outputs: no readout matrix available.")

    def fit_readout(self,
                    states: Sequence,
                    teachers: Sequence,
                    reg_model: Callable=None,
                    ridge: float=None,
                    force_pinv: bool=False,
                    verbose: bool=False) -> np.ndarray:
        """Compute a readout matrix by fitting the states computed by the ESN
        to the ground truth expected values, using the regression model defined
        in the ESN.

        Arguments:
            states {Sequence} -- All states computed.
            teachers {Sequence} -- All ground truth vectors.

        Keyword Arguments:
            reg_model {scikit-learn regression model} -- Use a scikit-learn regression model. (default: {None})
            ridge {float} -- Use Tikhonov regression and set regularization parameter to ridge. (default: {None})
            force_pinv -- Overwrite all previous parameters and force pseudo-inverse resolution. (default: {False})
            verbose {bool} -- (default: {False})

        Returns:
            np.ndarray -- Readout matrix.
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
        check_values(array_or_list=states, value=None)

        if verbose:
            tic = time.time()
            print("Linear regression...")
        # concatenate the lists (along timestep axis)
        X = np.hstack(states).astype(self.typefloat)
        Y = np.hstack(teachers).astype(self.typefloat)

        # Adding ones for regression with bias b in (y = a*x + b)
        X = np.vstack((np.ones((1, X.shape[1]),dtype=self.typefloat), X))

        # Building Wout with a linear regression model.
        # saving the output matrix in the ESN object for later use
        Wout = reg_model(X, Y)

        if verbose:
            toc = time.time()
            print(f"Linear regression done! (in {toc - tic} sec)")

        # return readout matrix
        return Wout

    def train(self,
              inputs: Sequence[np.ndarray],
              teachers: Sequence[np.ndarray],
              wash_nr_time_step: int=0,
              workers: int=-1,
              backend: str="threading",
              verbose: bool=False) -> Sequence[np.ndarray]:
        """Train the ESN model on a sequence of inputs.

        Arguments:
            inputs {Sequence[np.ndarray]} -- Training set of inputs.
            teachers {Sequence[np.ndarray]} -- Training set of ground truth.

        Keyword Arguments:
            wash_nr_time_step {int} -- Number of states to considered as transitory
                            when training. (default: {0})
            workers {int} -- if n >= 1, will enable parallelization of
                             states computation with n threads/processes, if possible.
                             if n = -1, will use all available resources for
                             parallelization.
            backend {str} -- Backend used for parallelization of
                             states computations. Available backends are
                             `threadings`(recommended, see Note), `multiprocess`,
                             `loky` (default: {"threading"}).
            verbose {bool} -- if `True`, display progress in stdout.

        Returns:
            Sequence[np.ndarray] -- All states computed, for all inputs.

        Note:
            If only one input sequence is provided ("continuous time" inputs), workers should be 1,
            because parallelization is impossible. In other cases, if using large NumPy arrays during
            computation (which is often the case), prefer using `threading` backend to avoid huge
            overhead. Multiprocess is a good idea only in very specific cases, and this code is not
            (yet) well suited for this.
        """
        ## Autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs, outputs=teachers)

        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Training on {len(inputs)} inputs ({steps} steps)-- wash: {wash_nr_time_step} steps")

        # compute all states
        all_states = self.compute_all_states(inputs,
                                             forced_teachers=teachers,
                                             wash_nr_time_step=wash_nr_time_step,
                                             workers=workers,
                                             backend=backend,
                                             verbose=verbose)

        all_teachers = [t[wash_nr_time_step:].T for t in teachers]

        # compute readout matrix
        self.Wout = self.fit_readout(all_states, all_teachers, verbose=verbose)

        # save the expected dimension of outputs
        self.dim_out = self.Wout.shape[0]

        # return all internal states
        return [st.T for st in all_states]

    def run(self,
            inputs: Sequence[np.ndarray],
            init_state: np.ndarray=None,
            init_fb: np.ndarray=None,
            workers: int=-1,
            backend: str="threading",
            verbose: bool=False) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Run the model on a sequence of inputs, and returned the states and
           readouts vectors.

        Arguments:
            inputs {Sequence[np.ndarray]} -- Sequence of inputs.

        Keyword Arguments:
            init_state {np.ndarray} -- State initialization vector
                                       for all inputs. (default: {None})
            init_fb {np.ndarray} -- Feedback initialization vector
                                    for all inputs, if feedback is
                                    enabled. (default: {None})
            workers {int} -- if n >= 1, will enable parallelization of
                             states computation with n threads/processes, if possible.
                             if n = -1, will use all available resources for
                             parallelization.
            backend {str} -- Backend used for parallelization of
                             states computations. Available backends are
                             `threadings`(recommended, see Note), `multiprocess`,
                             `loky` (default: {"threading"}).
            verbose {bool} -- if `True`, display progress in stdout.

        Returns:
            Tuple[Sequence[np.ndarray], Sequence[np.ndarray]] -- All states and readouts,
                                                                 for all inputs.

        Note:
            If only one input sequence is provided ("continuous time" inputs), workers should be 1,
            because parallelization is impossible. In other cases, if using large NumPy arrays during
            computation (which is often the case), prefer using `threading` backend to avoid huge
            overhead. Multiprocess is a good idea only in very specific cases, and this code is not
            (yet) well suited for this.
        """

        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Running on {len(inputs)} inputs ({steps} steps)")

        ## Autochecks of inputs
        self._autocheck_io(inputs=inputs)

        all_states = self.compute_all_states(inputs,
                                             init_state=init_state,
                                             init_fb=init_fb,
                                             workers=workers,
                                             backend=backend,
                                             verbose=verbose)

        all_outputs = self.compute_outputs(all_states)
        # return all_outputs, all_int_states
        return [st.T for st in all_outputs], [st.T for st in all_states]

    def generate(self,
                 nb_timesteps: int,
                 init_inputs: np.ndarray,
                 init_state: np.ndarray = None,
                 init_fb: np.ndarray = None,
                 return_init: bool = False,
                 verbose: bool = False
                 ) -> np.ndarray:
        """Run the ESN on a generative mode.
        After the init_inputs are consumed, new outputs are
        used as inputs for the next nb_timesteps.

        Args:
            nb_timesteps (int): [description]
            init_inputs (np.ndarray): [description]
            init_state (np.ndarray, optional): [description]. Defaults to None.
            init_fb (np.ndarray, optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            np.ndarray: [description]
        """

        if verbose:
            print(f"Generating {nb_timesteps} timesteps from {init_inputs.shape[0]} inputs.")
            print("Computing initial states...")

        _, init_states = self._compute_states(init_inputs, init_state=init_state,
                                              init_fb=init_fb)

        s0 = init_states[:, -1].reshape(-1, 1)
        init_outputs = self.compute_outputs([init_states])[0]
        u0 = init_outputs[:, -1].reshape(1, -1)

        if init_fb is not None:
            fb0 = self.compute_outputs([init_states[:, -2]])[0]
        else:
            fb0 = None

        if verbose:
            track = tqdm
        else:
            track = lambda x, text: x

        states = [None] * nb_timesteps
        outputs = [None] * nb_timesteps
        for i in track(range(nb_timesteps), "Generating"):
            _, s = self._compute_states(u0, init_state=s0, init_fb=fb0)
            s0 = s[:, -1].reshape(-1, 1)
            states[i] = s0.reshape(self.N)

            if fb0 is not None:
                fb0 = u0.copy()

            u = self.compute_outputs([s0])
            u0 = u[0].reshape(1, -1)
            outputs[i] = u0.reshape(self.dim_inp - self.in_bias)

        outputs = np.array(outputs)
        states = np.array(states)

        if return_init:
            outputs = np.vstack([init_outputs.T, outputs])
            states = np.vstack([init_states.T, states])

        return outputs, states

    def save(self, directory: str):
        """Save the ESN to disk.

        Arguments:
            directory {str or Path} -- Directory of the saved model.
        """
        _save(self, directory)

    #? maybe remove this in the future: unused
    def describe(self) -> Dict:
        """
        Provide descriptive stats about ESN matrices.

        Returns:
            Dict -- Descriptive data.
        """

        desc = {
            "Win": {
                "max": np.max(self.Win),
                "min": np.min(self.Win),
                "mean": np.mean(self.Win),
                "median": np.median(self.Win),
                "std": np.std(self.Win)
            },
            "W": {
                "max": np.max(self.W),
                "min": np.min(self.W),
                "mean": np.mean(self.W),
                "median": np.median(self.W),
                "std": np.std(self.W),
                "sr": max(abs(linalg.eig(self.W)[0]))
            }
        }
        if self.Wfb is not None:
            desc["Wfb"] = {
                "max": np.max(self.Wfb),
                "min": np.min(self.Wfb),
                "mean": np.mean(self.Wfb),
                "median": np.median(self.Wfb),
                "std": np.std(self.Wfb)
            }
        if self.Wout is not None:
            desc["Wout"] = {
                "max": np.max(self.Wout),
                "min": np.min(self.Wout),
                "mean": np.mean(self.Wout),
                "median": np.median(self.Wout),
                "std": np.std(self.Wout)
            }
        return desc
