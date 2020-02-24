# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
@author: Xavier HINAUT
xavier.hinaut@inria.fr
Copyright Xavier Hinaut 2018

I would like to thank Mantas Lukosevicius for his code that was used as inspiration for this code:
http://minds.jacobs-university.de/mantas/code
"""
import os
import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from .utils import check_values
from .regression_models import sklearn_linear_model
from .regression_models import ridge_linear_model
from .regression_models import pseudo_inverse_linear_model


class ESN():
        
    
    def __init__(self,
                 lr: float,
                 W: np.array,
                 Win: np.array,
                 input_bias: bool=True,
                 reg_model: Callable=None,
                 ridge: float=None,
                 Wfb: np.array=None,
                 fbfunc: Callable=None,
                 typefloat: np.dtype=np.float64):
        """Base class of Echo State Networks
        
        Arguments:
            lr {float} -- Leaking rate 
            W {np.array} -- Reservoir weights matrix
            Win {np.array} -- Input weights matrix
        
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
        self.Wout = None # output weights matrix. must be learnt through training.
        
        # check if dimensions of matrices are coherent
        self._autocheck_dimensions()
        
        self._state = None
        
        self.N = self.W.shape[1] # number of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] # dimension of inputs (including the bias at 1)
        
        self.typefloat = typefloat
        self.lr = lr # leaking rate

        self.Wfb = Wfb 
        self.fbfunc = fbfunc
        if self.Wfb is not None and self.fbfunc is None:
            raise ValueError(f"If a feedback matrix is provided, \
                fbfunc must be a callable object, not {self.fbfunc}.")
        
        self.reg_model = self._set_regression_model(ridge, reg_model)

        if not self.in_bias:
            str_err = "TODO: the ESN class is uncomplete for the case you try to use "
            str_err += "-> the ESN without input bias is not implemented yet."
            raise NotImplementedError(str_err)
        
        self.dim_out = None
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1] # dimension of outputs
            
        self._state = None # state of the reservoir
        self._last_feedback = None # memory of the last feedback vector received.  
        
        self._autocheck_nan()


    def _set_regression_model(self, ridge: float=None, sklearn_model: Callable=None):
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
        assert self.Win.shape[0] == self.W.shape[0], f"Win shape should be ({self.W.shape[1]}, input) but is {self.Win.shape}."


    def _autocheck_io(self,
                      inputs,
                      outputs=None):
        
        # Check if inputs and outputs are lists
        assert type(inputs) is list, "Inputs should be a list of numpy arrays"
        if outputs is not None:
            assert type(outputs) is list, "Outputs should be a list of numpy arrays"
        
        # check if Win matrix has coherent dimensions with input dimensions
        if self.in_bias:
            assert self.Win.shape[1] == inputs[0].shape[1] + 1, f"With bias, Win matrix should be of shape \
                ({self.N}, {inputs[0].shape[1] + 1}) but is {self.Win.shape}."
        else:
            assert self.Win.shape[1] == inputs[0].shape[1], f"Win matrix should be of shape ({self.N}, {self.dim_inp}) \
                but is {self.Win.shape}."
                
        if outputs is not None:
            # check feedback matrix
            if self.Wfb is not None:
                assert outputs[0].shape[1] == self.Wfb.shape[1], f"With feedback, Wfb matrix should be of shape \
                    ({self.N}, {outputs[0].shape[1]}) but is {self.Wfb.shape}."
    
    
    def reset_reservoir(self):
        """
        Reset ESN state and saved last feedback to 0 vectors.
        """
        self._state = np.zeros((self.N,1),dtype=self.typefloat)
        if self.Wfb is not None:
            self._last_feedback = np.zeros((self.dim_out,1),dtype=self.typefloat)
        
    
    def update_state(self,
                     single_input: np.array,
                     feedback: None=np.array, 
                     inplace=True) -> np.array:
        """
        Update the current reservoir state.
        
        Arguments:
            single_input {np.array} -- Input vector (represents only one timestep.)
        
        Keyword Arguments:
            feedback {np.array} -- Feedback vector, either ground truth vector during
                                   training or output vector during inference. Is 
                                   needed if the feedback matrix (Wfb) is not None.
                                   (default: {None})
            inplace {bool} -- if True, will change the current state stored in the 
                              object. Otherwise, just return the new state without
                              modifiying the previous stored state. 
                              (default: {True})
        

        
        Raises:
            RuntimeError: if a feedback matrix exists but no feedback vector is passed as argument.
            UserWarning: if a feedback vector is passed but their is no feedback matrix in the ESN.
            UserWarning: no current state is available in the ESN, so update will occur on a null vector.
        
        Returns:
            np.array -- New state computed from the input.
        """
        # check if the user is trying to add empty feedback
        if self.Wfb is not None and feedback is None:
            raise RuntimeError("Missing a feedback vector.")
        
        # warn if the user is adding a feedback vector when feedback is not available
        # (might have forgotten the feedback weights matrix)
        if self.Wfb is None and feedback is not None:
            warnings.warn("Feedback vector should not be passed to update_state if no \
                                  feedback matrix is provided.", UserWarning)
        
        # first retrieve the current state of the ESN
        # (or initialize it)
        if self._state is not None:
            x = self._state
        else:
            x = np.zeros((self.N,1),dtype=self.typefloat)
            warnings.warn("State was not initialized neither by training nor by running. \
                           Will assume a 0 vector as initial state to compute the update.", UserWarning)
            
        # linear transformation
        x1 = np.dot(self.Win, single_input.reshape(self.dim_inp, 1)) \
            + np.dot(self.W, x)
        
        # add feedback if requested
        if self.Wfb is not None:
            x1 += np.dot(self.Wfb, self.fbfunc(feedback))
        
        # previous states memory leak and non-linear transformation
        x1 = (1-self.lr)*x + self.lr*np.tanh(x1)
        
        # update the current state if requested
        if inplace:
            self._state = x1

        # return the next state computed
        return x1  
    
    
    def activate_on_input(self,
                          input: np.array,
                          teacher: np.array=None,
                          wash_nr_time_step=0,
                          tqdm_object=None):
        """Run the network over an input time sequence, for training or inference.
        
        Arguments:
            input {np.array} -- Input time sequence.
        
        Keyword Arguments:
            teacher {np.array} -- If training, sequence of ground truths to predict. (default: {None})
            wash_nr_time_step {int} -- If training, number of warmup steps. (default: {0})
            tqdm_object {Callable} -- object used to track progress if needed (default: {False})
        
        Raises:
            RuntimeError: if trying to run an untrained model.
            ValueError: if trying to set warmup above 0 for inference.
        
        Returns:
            [np.array, np.array] -- internal states and outputs. If training,
                                    outputs will be equal to teacher.
        """
        
        # check if the user is trying to run the model without having it trained.
        if self.Wout is None and teacher is None:
            raise RuntimeError("Impossible to run without output matrix. First train the model.")
        
        # check if the user is trying to run the model with warmup.
        if teacher is None and wash_nr_time_step > 0:
            raise ValueError("Warmup should be 0 if not in training phase.")
        
        # if running
        if teacher is None:
            outputs = np.zeros((self.dim_out, len(input)-wash_nr_time_step), dtype=self.typefloat)
        # if training
        else:
            outputs = teacher[wash_nr_time_step:, :].T

        # to track successives internal states of the reservoir
        internal_states = np.zeros((self.N, len(input)-wash_nr_time_step), dtype=self.typefloat)
                
        # If a feedback matrix is available, feedback will be set to 0 or to 
        # a saved previous output value.
        if self.Wfb is not None and self._last_feedback is None:
            self._last_feedback = np.zeros((self.dim_out, 1), dtype=self.typefloat)
        
        # for each time step in the input
        for t in range(input.shape[0]):
            # compute next state. By default, self._state is also updated.
            x = self.update_state(input[t, :], feedback=self._last_feedback)

            if teacher is not None: # during training outputs are equal to teachers for feedback
                y = teacher[t,:].reshape(teacher.shape[1], 1).astype(self.typefloat)
            else: # outputs for inference, computed with Wout
                y = np.dot(self.Wout, np.vstack((1,x))).astype(self.typefloat)
                
            # save the last output for next feedback
            if self.Wfb is not None:
                self._last_feedback = y.reshape(self.dim_out, 1)
                
            # will track all internal states during inference, and only the 
            # states after wash_nr_time_step during training.
            if t >= wash_nr_time_step:
                internal_states[:, t-wash_nr_time_step] = x.reshape(-1,).astype(self.typefloat)
                
                # if training
                if teacher is None:
                    outputs[:, t-wash_nr_time_step] = y.reshape(-1,)
                    
            if tqdm_object is not None:
                tqdm_object.update(1)
        
        return internal_states, outputs
    
      
    def train(self, 
              inputs: List[np.array], 
              teachers: List[np.array], 
              wash_nr_time_step: int, 
              reset_state=True,
              verbose=False) -> np.array:
        """ Train ESN on a serie of inputs, using a regression model.
        
        Arguments:
            inputs {np.array} -- List of input time sequences.
            teachers {np.array} -- List of ground truths sequences.
            wash_nr_time_step {int} -- Number of warmup steps.
            
        Keyword Arguments:
            reset_state {bool} -- if True, will reset internal state and 
                                  potential feedback vector to 0 after
                                  each input.
            verbose {bool} -- (default: {False})
        
        Returns:
            np.array -- All internal states computed.
        """        
        ## Autochecks of inputs and outputs
        self._autocheck_io(inputs=inputs, outputs=teachers)

        # 'pre-allocated' memory for the list that will collect the states
        all_int_states = [None]*len(inputs)
        # 'pre-allocated' memory for the list that will collect the teachers (desired outputs)
        all_teachers = [None]*len(teachers)

        # reset state and feedback memory of the reservoir to 0
        self.reset_reservoir()

        # tqdm progress bar to monitor training
        prog_bar = None
        if verbose:
            total_timesteps = np.array([i.shape[0] for i in inputs]).sum()
            prog_bar = tqdm(total=total_timesteps)
        
        # run reservoir over the inputs, to save the internal states and the teachers (desired outputs) in lists
        for (j, (inp, tea)) in enumerate(zip(inputs, teachers)):

            if verbose:
                prog_bar.set_description(f"Training input {j + 1} of {len(inputs)}")
                
            # add bias to the input (b = 1)
            if self.in_bias:
                u = np.column_stack((np.ones((inp.shape[0],1)),inp)).astype(self.typefloat)
            else:
                u = inp.astype(self.typefloat)
            
            # reset the states or not, that is the question
            if reset_state:
                self.reset_reservoir()
            else:
                # keep the previously runned state
                pass
            
            # collect the network responses to the input
            internal_states, output_tea = self.activate_on_input(u, 
                                                                 teacher=tea.astype(self.typefloat),
                                                                 wash_nr_time_step=wash_nr_time_step,
                                                                 tqdm_object=prog_bar)

            all_int_states[j] = internal_states.astype(self.typefloat)
            all_teachers[j] = output_tea.astype(self.typefloat)
        
        
        if verbose:
            prog_bar.close()
        
        # check if network responses are valid
        check_values(array_or_list=all_int_states, value=None)
        check_values(array_or_list=all_teachers, value=None)

        
        if verbose:
            print("Linear regression...")
        # concatenate the lists (along timestep axis)
        X = np.hstack(all_int_states).astype(self.typefloat)
        Y = np.hstack(all_teachers).astype(self.typefloat)
        
        # Adding ones for regression with biais b in (y = a*x + b)
        X = np.vstack((np.ones((1, X.shape[1]),dtype=self.typefloat), X))

        # Building Wout with a linear regression model.
        # saving the output matrix in the ESN object for later use
        self.Wout = self.reg_model(X, Y)
        
        # save the expected dimension of outputs
        self.dim_out = self.Wout.shape[0]
        
        if verbose:
            print("Done !")
        
        # return all internal states
        return [st.T for st in all_int_states]


    def run(self, 
            inputs: np.array, 
            reset_state=True, 
            verbose=False) -> Tuple[np.array, np.array]:
        """ Run ESN on a serie of inputs, and return the inferred values.
        
        Arguments:
            inputs {np.array} -- List of input time sequences.
            
        Keyword Arguments:
            reset_state {bool} -- if True, will reset internal state and 
                                  potential feedback vector to 0 after
                                  each input. 
            verbose {bool} -- (default: {False})
        
        Returns:
            np.array -- All internal states computed.
        """ 
        
        ## Autochecks of inputs
        self._autocheck_io(inputs=inputs)

        # run the trained ESN without 
        if reset_state or self._state is None:
            self.reset_reservoir()
        else:
            # run the trained ESN in a generative mode. no need to initialize here,
            # because x is initialized with training data and we continue from there
            pass

        all_int_states = [None]*len(inputs)
        all_outputs = [None]*len(inputs)

        # tqdm progress bar to monitor runs
        prog_bar = None
        if verbose:
            total_timesteps = np.array([i.shape[0] for i in inputs]).sum()
            prog_bar = tqdm(total=total_timesteps)

        # run reservoir over the inputs, to save the internal states and the teachers (desired outputs) in lists
        for (j, inp) in enumerate(inputs):
            
            if verbose:
                prog_bar.set_description(f"Running input {j + 1} of {len(inputs)}")
            
            if self.in_bias:
                u = np.column_stack((np.ones((inp.shape[0],1)),inp)).astype(self.typefloat)
            else:
                u = inp.astype(self.typefloat)

            # reset the states or not, that is the question
            if reset_state:
                self.reset_reservoir()
            
            internal_states, outputs = self.activate_on_input(u, tqdm_object=prog_bar)
            
            all_int_states[j] = internal_states.astype(self.typefloat)
            all_outputs[j] = outputs.astype(self.typefloat)
        
        if verbose:
            prog_bar.close()

        # return all_outputs, all_int_states
        return [st.T for st in all_outputs], [st.T for st in all_int_states]


    def print_trained_esn_info(self):
        print("esn.Win", self.Win)
        print("esn.Win max", np.max(self.Win))
        print("esn.Win min", np.min(self.Win))
        print("esn.Win mean", np.mean(self.Win))
        print("esn.Win median", np.median(self.Win))
        print("esn.Win std", np.std(self.Win))
        print("esn.W", self.W)
        print("esn.W max", np.max(self.W))
        print("esn.W min", np.min(self.W))
        print("esn.W mean", np.mean(self.W))
        print("esn.W median", np.median(self.W))
        print("esn.W std", np.std(self.W))
        if self.Wout is not None:
            print("esn.Wout", self.Wout)
            print("esn.Wout max", np.max(self.Wout))
            print("esn.Wout min", np.min(self.Wout))
            print("esn.Wout mean", np.mean(self.Wout))
            print("esn.Wout median", np.median(self.Wout))
            print("esn.Wout std", np.std(self.Wout))
