# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""reservoirpy/esn_online

Echo State Networks with online learning (FORCE learning)
"""
import os
import time
import warnings
import pickle
from typing import Sequence, Callable, Tuple, Union, Dict

import numpy as np
from scipy import linalg
from tqdm import tqdm

from .utils import check_values, _save




class ESNOnline(object):
    
    def __init__(self,
                 lr: float,
                 W: np.ndarray,
                 Win: np.ndarray,
                 Wout: np.ndarray,
                 alpha_coef: float=1e-6,
                 use_raw_input: bool=False,
                 input_bias: bool=True,
                 Wfb: np.ndarray=None,
                 fbfunc: Callable=None,
                 typefloat: np.dtype=np.float64):
        """Base class of Echo State Networks
        
        Arguments:
            lr {float} -- Leaking rate 
            W {np.ndarray} -- Reservoir weights matrix
            Win {np.ndarray} -- Input weights matrix
            Wout {np.ndarray} -- Output weights matrix
        
        Keyword Arguments:
            alpha_coef {float} -- Coefficient to scale the inversed state correlation matrix
                                  used for FORCE learning
            use_raw_input {bool} -- If True, input is used directly when computing output
                                    (default: {False})
            input_bias {bool} -- If True, will add a constant bias
                                 to the input vector (default: {True})
            Wfb {np.ndarray} -- Feedback weights matrix
            fbfunc {Callable} -- Feedback activation function. (default: {None})
            typefloat {np.dtype} -- Float precision to use (default: {np.float32})
        
        Raises:
            ValueError: If a feedback matrix is passed without activation function.
            NotImplementedError: If trying to set input_bias to False. This is not
                                 implemented yet.
        """
        
        self.W = W 
        self.Win = Win 
        self.Wfb = Wfb    
        self.Wout = Wout
        
        self.use_raw_inp = use_raw_input

        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        self._autocheck_nan() 

        self.N = self.W.shape[1] # number of neurons
        
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] - 1 if self.in_bias else self.Win.shape[1] # dimension of inputs (not including the bias at 1)

        self.dim_out = Wout.shape[0]
        self.state_size = Wout.shape[1]
        self.output_values = np.zeros((self.dim_out,1)).astype(typefloat)
            
        self.typefloat = typefloat
        self.lr = lr # leaking rate
        
        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError(f"If a feedback matrix is provided, \
                fbfunc must be a callable object, not {self.fbfunc}.")

        self.alpha_coef = alpha_coef # coef used to init state_corr_inv matrix
        
        # Init internal state vector and state_corr_inv matrix
        # (useful if we want to freeze the online learning)
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
        nb_states = self.Win.shape[1] + self.W.shape[0] + 1 if self.use_raw_inp else self.W.shape[0] + 1
        err = f"Wout shape should be (output, {nb_states}) but is {self.Wout.shape}."
        assert self.Wout.shape[1] == nb_states, err

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
            assert self.Win.shape[1] == inputs_0.shape[0] + 1, err
        else:
            err = f"Win matrix should be of shape ({self.N}, "
            err += f"{inputs_0.shape[0]}) but is {self.Win.shape}."
            assert self.Win.shape[1] == inputs_0.shape[0], err
                
        if outputs is not None:
            # check feedback matrix
            if self.Wfb is not None:
                outputs_0 = outputs[0]
                err = f"With feedback, Wfb matrix should be of shape ({self.N}, "
                err += f"{outputs_0.shape[0]}) but is {self.Wfb.shape}."
                assert outputs_0.shape[0] == self.Wfb.shape[1], err
            # check output weights matrix
            if self.Wout is not None:
                outputs_0 = outputs[0]
                err = f"Wout matrix should be of shape ({outputs_0.shape[0]}, "
                err += f"{self.state_size}) but is {self.Wout.shape}."
                assert (outputs_0.shape[0], self.state_size) == self.Wout.shape, err

        
    def _get_next_state(self,
                        single_input: np.ndarray) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t), compute the state vector x(t+1).
        
        Arguments:
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
            u = np.hstack((1, single_input)).astype(self.typefloat)
        else:
            u = single_input
        
        # linear transformation
        x1 = self.Win @ u.reshape(-1, 1) + self.W @ x
        
        # add feedback if requested
        if self.Wfb is not None:
            x1 += self.Wfb @ self.fbfunc(self.output_values)
        
        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*np.tanh(x1)

        # return the next state computed
        if self.use_raw_inp:
            self.state = np.vstack((1.0, x1, u.reshape(-1, 1)))
        else:
            self.state = np.vstack((1.0, x1))         
        
        return self.state.copy()
    
    
    def compute_output_from_current_state(self):
        """ Compute output from current state s(t) of the reservoir
        
        Returns:
            np.ndarray -- Output at time t
        """
        
        assert self.Wout is not None, "Matrix Wout is not initialized/trained yet"
        
        self.output_values = (self.Wout @ self.state).astype(self.typefloat)
        return self.output_values.copy().ravel()


    def compute_output(self,
                       single_input: np.ndarray,
                       wash_nr_time_step: int=0):
        """ Compute output from input to the reservoir

        Arguments:
            single_input {np.ndarray} -- Input vector u(t).
    
        Keyword Arguments:
            wash_nr_time_step {int} -- Time for reservoir to run in free (without collecting output 
                                       or fitting Wout). (default: {0})

        Returns:
            state {np.ndarray} -- New state after input u(t) is passed
            output {np.ndarray} -- Output after input u(t) is passed
        """
        
        state = self._get_next_state(single_input)
        output = self.compute_output_from_current_state()
        
        return output, state
    
    
    def reset_reservoir(self):
        """
            Reset reservoir by setting internal values to zero
        """
        self.state = np.zeros((self.state_size,1),dtype=self.typefloat)


    def reset_correlation_matrix(self):
        self.state_corr_inv = np.asmatrix(np.eye(self.state_size)) / self.alpha_coef


    def train_from_current_state(self, targeted_output, indexes = None):
        """ Train Wout from current internal state.
        
        Arguments:
            error {np.ndarray} -- Error when using current Wout to compute output
        
        Keyword Arguments:
            indexes {int | list} -- If indexes is not None, only the provided output 
                                    indexes are learned (default: {None})            
        """

        error = self.output_values - targeted_output.reshape(-1,1)

        self.state_corr_inv = _new_correlation_matrix_inverse(self.state, self.state_corr_inv)
        
        if indexes == None:
            self.Wout -= error @ (self.state_corr_inv @ self.state).T
        else:
            self.Wout[indexes] -= error[indexes] * (self.state_corr_inv @ self.state).T


    def train(self, 
              inputs: Sequence[np.ndarray], 
              teachers: Sequence[np.ndarray],
              wash_nr_time_step: int=0,
              verbose: bool=False) -> Sequence[np.ndarray]:
        """Train the ESN model on a sequence of inputs.
        
        Arguments:
            inputs {Sequence[np.ndarray]} -- Training set of inputs.
            teachers {Sequence[np.ndarray]} -- Training set of ground truth.
        
        Keyword Arguments:
            wash_nr_time_step {int} -- Number of states to considered as transitory 
                            when training. (default: {0})
            verbose {bool} -- if `True`, display progress in stdout.
        
        Returns:
            Sequence[np.ndarray] -- All states computed, for all inputs.
        """
        inputs_concat = [inp[t,:] for inp in inputs for t in range(inp.shape[0])]
        teachers_concat = [tea[t,:] for tea in teachers for t in range(tea.shape[0])]

        ## Autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs_concat, outputs=teachers_concat)
        
        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Training on {len(inputs)} inputs ({steps} steps)-- wash: {wash_nr_time_step} steps")

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
            verbose: bool=False) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Run the model on a sequence of inputs, and returned the states and 
           readouts vectors.
        
        Arguments:
            inputs {Sequence[np.ndarray]} -- Sequence of inputs.
        
        Keyword Arguments:
            verbose {bool} -- if `True`, display progress in stdout.
        
        Returns:
            Tuple[Sequence[np.ndarray], Sequence[np.ndarray]] -- All states and readouts, 
                                                                 for all inputs.

        """
        
        inputs_concat = [inp[t,:] for inp in inputs for t in range(inp.shape[0])]

        steps = np.sum([i.shape[0] for i in inputs])
        if verbose:
            print(f"Running on {len(inputs)} inputs ({steps} steps)")
        
        ## Autochecks of inputs
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
        
        Arguments:
            directory {str or Path} -- Directory of the saved model.
        """
        _save(self, directory)
        
        
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
    
    # TODO : numerical instabilities if xTP is not computed first (order of multiplications)  
    xTP = x.T @ P
    P = P - (P @ x @ xTP)/(1. + np.dot(xTP, x))
    
    return P
