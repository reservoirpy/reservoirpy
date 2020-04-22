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
        
        Keyword Arguments:
            input_bias {bool} -- If True, will add a constant bias
                                 to the input vector (default: {True})
            use_raw_input {bool} -- If True, input is used directly when computing output
                                    (default: {False})
            fbfunc {Callable} -- Feedback activation function. (default: {None})
            typefloat {np.dtype} -- Float precision to use (default: {np.float32})
        
        Raises:
            ValueError: If a feedback matrix is passed without activation function.
            NotImplementedError: If trying to set input_bias to False. This is not
                                 implemented yet.
        """
        
        self.W = W 
        self.Win = Win 
        self.Wout = None # output weights matrix. In case of FORCE learning, it is initialized before training.
        self.Wfb = Wfb    
        
        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        self._autocheck_nan() 

        self.N = self.W.shape[1] # number of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] # dimension of inputs (including the bias at 1)
        self.dim_out = None
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1] # dimension of outputs
            
        self.typefloat = typefloat
        self.lr = lr # leaking rate
        
        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError(f"If a feedback matrix is provided, \
                fbfunc must be a callable object, not {self.fbfunc}.")

        self.use_raw_inp = use_raw_input
        
        # Set the size of state vector containing the value of neurons, bias, (may/maynot) input
        if self.use_raw_inp:
            self.state_size = self.N + 1 + self.Win.shape[1]
        else:
            self.state_size = self.N + 1
            
        self.alpha_coef = alpha_coef # coef used to init state_corr_inv matrix
        
        # Init internal state vector and state_corr_inv matrix
        # (useful if we want to freeze the online learning)
        self.reset_reservoir()


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
        assert np.isnan(self.W).any() == False, "W matrix should not contain NaN values."
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

        
    def _get_next_state(self,
                        single_input: np.ndarray,   
                        feedback: np.ndarray=None) -> np.ndarray:
        """Given a state vector x(t) and an input vector u(t), compute the state vector x(t+1).
        
        Arguments:
            single_input {np.ndarray} -- Input vector u(t).
        
        Keyword Arguments:
            feedback {np.ndarray} -- Feedback vector if enabled. (default: {None})
            
        Raises:
            RuntimeError: feedback is enabled but no feedback vector is available.
        
        Returns:
            np.ndarray -- Next state s(t+1).
        """
        
        # check if the user is trying to add empty feedback
        if self.Wfb is not None and feedback is None:
            raise RuntimeError("Missing a feedback vector.")
        
        # warn if the user is adding a feedback vector when feedback is not available
        # (might have forgotten the feedback weights matrix)
        if self.Wfb is None and feedback is not None:
            warnings.warn("Feedback vector should not be passed to update_state if no feedback matrix is provided.", UserWarning)
        
        x = self.state[1:self.N+1]

        # add bias
        if self.in_bias:
            u = np.vstack((1, single_input)).astype(self.typefloat)
        else:
            u = single_input
        
        # linear transformation
        x1 = np.dot(self.Win, u.reshape(self.dim_inp, 1)) \
            + np.dot(self.W, x)
        
        # add feedback if requested
        if self.Wfb is not None:
            x1 += np.dot(self.Wfb, self.fbfunc(feedback))
        
        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*np.tanh(x1)

        # return the next state computed
        if self.use_raw_inp:
            self.state = np.vstack((1.0, x1, u))
        else:
            self.state = np.vstack((1.0, x1))         
        
        return self.state.copy()
    
    
    def compute_output(self,
                       single_input: np.ndarray,
                       last_feedback: np.ndarray=None,
                       wash_nr_time_step: int=0):        
        # if a feedback matrix is available, feedback will be set to 0 or to 
        # a specific value.
        if self.Wfb is not None:
            if last_feedback is None:
                last_feedback = np.zeros((self.dim_out, 1), dtype=self.typefloat)
        else:
            last_feedback = None
        
        state = self._get_next_state(single_input, feedback=last_feedback)
        output = np.dot(self.Wout, state).astype(self.typefloat)
        
        return state, output
    
    
    def reset_reservoir(self):
        """
            Reset reservoir by setting internal values to zero
        """
        self.state = np.zeros((self.state_size,1),dtype=self.typefloat)
        self.state_corr_inv = np.asmatrix(np.eye(self.state_size)) / self.alpha_coef


    def prepare_for_training(self, output_dim, alpha_coef, random_Wout=None):
        
        # Reset reservoir
        self.reset_reservoir()
        
        # Set the output dimension
        self.dim_out = output_dim
        
        # Init Wout(0)
        Wout_shape = (self.dim_out, self.state_size)
        if random_Wout is None:
            self.Wout = np.zeros(Wout_shape)
        elif random_Wout == 'gaussian':
            self.Wout = np.random.normal(0, 1, Wout_shape)
        else:
            self.Wout = np.random.randint(0, 2, Wout_shape) * 2 - 1


    def train_from_current_state(self, error, indexes = None):
        """
            Fit Wout to achieve the target output from current internal state.
            If indexes is not None, only the provided output indexes are learned
        """

        self.state_corr_inv = _new_correlation_matrix_inverse(self.state, self.state_corr_inv)
        
        if indexes == None:
            self.Wout -= np.dot(error, np.dot(self.state_corr_inv, self.state).T)
        else:
            self.Wout[indexes] -= np.dot(error[indexes], np.dot(self.state_corr_inv, self.state).T)


    def train(self, 
              inputs: Sequence[np.ndarray], 
              teachers: Sequence[np.ndarray],
              alpha_coef: float=1e-6,
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
        ## Autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs, outputs=teachers)
        
        # Prepare the network for training
        self.prepare_for_training(teachers[0].shape[0], alpha_coef)
        
        if verbose:
            steps = np.sum([i.shape[0] for i in inputs])
            print(f"Training on {len(inputs)} inputs ({steps} steps)-- wash: {wash_nr_time_step} steps")

        i = 0; nb_inputs = len(inputs)

        # First 'warm up' the network
        while i < wash_nr_time_step:
            self.compute_output(inputs[i])
            i += 1

        # Train Wout on each input
        while i < nb_inputs:
            _, output = self.compute_output(inputs[i])
            self.train_from_current_state(output - teachers[i])
            i += 1


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
    xTP = np.dot(x.T, P)
    Px = np.dot(P, x)
    P = P - np.dot(Px, xTP)/(1. + np.dot(xTP, x))
    
    return P
