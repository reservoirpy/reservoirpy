# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
@author: Xavier HINAUT
xavier.hinaut #/at\# inria.fr
Copyright Xavier Hinaut 2018

I would like to thank Mantas Lukosevicius for his code that was used as inspiration for this code:
http://minds.jacobs-university.de/mantas/code
"""

import numpy as np
from scipy import linalg
import sklearn.linear_model as sklm

class ESN():
    def __init__(self, lr, W, Win, input_bias=True, ridge=None, Wfb=None, fbfunc=None,
                typefloat=np.float64, reg_model=None):
        #TODO : add check if fbfunc is not set when Wfb is not None
        """
        Dimensions of matrices:
            - W : (nr_neurons, nr_neurons)
            - Win : (nr_neurons, dim_input)
            - Wout : (dim_output, nr_neurons)

        Inputs:
            - reg_model: sklearn.linear_model regression object.
                If it is defined, the provided model is used to learn output weights.
                ridge coefficient must not be defined at the same time
                Examples : reg_model = sklearn.linear_model.Lasso(1e-8)
                            reg_model = sklearn.linear_model.Ridge(1e-8)

        """
        self.W = W # reservoir matrix
        self.Win = Win # input matrix

        self.typefloat = typefloat
        self.lr = lr # leaking rate

        self.Wout = None
        self.Wfb = Wfb
        self.fbfunc = fbfunc
        self.N = self.W.shape[1] # nr of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] # dimension of inputs (including the bias at 1)

        self.ridge = None
        self.reg_model = None
        self.update_regression_model(ridge, reg_model)

        if self.in_bias:
            pass
        else:
            str_err = "TODO: the ESN class is uncomplete for the case you try to use "
            str_err += "-> the ESN without input bias is not implemented yet."
            raise ValueError(str_err)
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1] # dimension of outputs
        else:
            self.dim_out = None
        self.autocheck_nan()

    def update_regression_model(self, ridge = None, reg_model = None):

        if ridge is not None and reg_model is not None:
            raise Exception("ridge and reg_model can't be defined at the same time")

        if ridge is not None:
            self.ridge = ridge
            self.reg_model = None

        if reg_model is not None:
            self.reg_model = reg_model
            self.ridge = None

    def autocheck_nan(self):
        """ Auto-check to see if some important variables do not have a problem (e.g. NAN values). """
        assert np.isnan(self.W).any() == False # W matrix should not contain NAN values
        assert np.isnan(self.Win).any() == False # W matrix should not contain NAN values
        if self.Wfb is not None:
            assert np.isnan(self.Wfb).any() == False # W matrix should not contain NAN values

    def check_values(self, array_or_list, value):
        """ Check if the given array or list contains the given value. """
        if value == np.nan:
            assert np.isnan(array_or_list).any() == False # array should not contain NAN values
        if value == None:
            if type(array_or_list) is list:
                assert np.count_nonzero(array_or_list == None) == 0
                #assert array_or_list.count(None) == 0 # check if there is a None value
                #TODO: this one does not seem to work: assert any([x is None for x in array_or_list])  # check if there is a None value
            elif type(array_or_list) is np.array:
                # None is transformed to np.nan when it is in an array
                assert np.isnan(array_or_list).any() == False # array should not contain NAN values


    def autocheck_io(self, inputs, outputs=None, verbose=False):
        # TODO: add check of output dimension
        if verbose:
            print("self.Win.shape[1]", self.Win.shape[1])
            print("sel.dim_inp", self.dim_inp)
            print("inputs[0].shape", inputs[0].shape)
        # check if inputs and outputs are lists
        assert type(inputs) is list # inputs should be a list of numpy arrays
        if outputs is not None:
            assert type(outputs) is list # outputs should be a list of numpy arrays
        # check if Win matrix has coherent dimensions with input dimensions
        assert self.Win.shape[1] == self.dim_inp
        if self.in_bias:
            assert self.Win.shape[1] == inputs[0].shape[1] + 1 # with biais, the 2nd dimension of input matrix Win should be = dimension_of_input + 1
        else:
            assert self.Win.shape[1] == inputs[0].shape[1] # without biais, the 2nd dimension of input matrix Win should be = dimension_of_input
        if outputs is not None:
            # TODO: add check of outputs
            # check feedback matrix
            if self.Wfb is not None:
                y = np.zeros((self.dim_out,1)) # I guess it's the best we can do, because we have no info on the teacher for this time step
                assert teachers[0].shape[1] == self.dim_out # check if output dimension is correct

    def train(self, inputs, teachers, wash_nr_time_step, reset_state=True, verbose=False):
        #TODO float32 : use float32 precision for training the reservoir instead of the default float64 precision
        #TODO: add a 'speed mode' where all asserts, print s and saved values are minimal
        #TODO: add option to enable direct connection from input to output to be learned
            # need to remember the input at this stage
        """
        Dimensions:
        Inputs:
            - inputs: list of numpy array item with dimension (nr_time_step, input_dimension)
            - teachers: list of numpy array item with dimension (nr_time_step, output_dimension)
        Outputs:
            - all_int_states: list of numpy array item with dimension
                - during the execution of this method : (N, nr_time_step)
                - returned dim (nr_time_step, N)

        - TODO float32 : use float32 precision for training the reservoir instead of the default float64 precision
        - TODO: add option to enable direct connection from input to output to be learned
            # need to remember the input at this stage
        - TODO: add a 'speed mode' where all asserts, prints and saved values are minimal
        """
        if verbose:
            print("len(inputs)", len(inputs))
            print("len(teachers)", len(teachers))
            print("self.N", self.N)
            print("self.W.shape", self.W.shape)
            print("self.Win.shape", self.Win.shape)
        self.autocheck_io(inputs=inputs, outputs=teachers)

        # 'pre-allocated' memory for the list that will collect the states
        all_int_states = [None]*len(inputs)
        x = np.zeros((self.N,1)) # current internal state initialized at zero
        x = np.zeros((self.N,1)) # current internal state initialized at zero
        # 'pre-allocated' memory for the list that will collect the teachers (desired outputs)
        all_teachers = [None]*len(teachers)

        # change of variable for conveniance in equation
        di = self.dim_inp
            #TODO: FINISH TO PUT ALL USEFUL VARIABLES IN float32
        inputs = [aa.astype(self.typefloat) for aa in inputs]
        teachers = [aa.astype(self.typefloat) for aa in teachers]
            #raise Exception("TODO: float32 option not finished yet!")

        # run reservoir over the inputs, to save the internal states and the teachers (desired outputs) in lists
        for (j, (inp, tea)) in enumerate(zip(inputs, teachers)):
            if verbose:
                print("j:", j)
                print("inp.shape", inp.shape)
                print("tea.shape", tea.shape)

            if self.in_bias:
                u = np.column_stack((np.ones((inp.shape[0],1)),inp))
            else:
                u = inp
            u = u.astype(self.typefloat)
            # reset the states or not, that is the question
            if reset_state:
                x = np.zeros((self.N,1),dtype=self.typefloat) # current internal state initialized at zero
                if self.Wfb is not None:
                    y = np.zeros((self.dim_out,1),dtype=self.typefloat)

            else:
                # keep the previously runned state
                pass

            all_int_states[j] = np.zeros((self.N,inp.shape[0]-wash_nr_time_step),dtype=self.typefloat)
            all_teachers[j] = np.zeros((tea.shape[1],inp.shape[0]-wash_nr_time_step),dtype=self.typefloat)

            for t in range(inp.shape[0]): # for each time step in the input
                # u = data[t]
                # u = np.atleast_2d(inp[t,:])
                if verbose:
                    print("inp[t,:].shape", inp[t,:].shape)
                    print("tea[t,:].shape", tea[t,:].shape)
                    print("u.shape", u.shape)
                    print("inp[t,:]", inp[t,:])
                    print("u[t].shape", u[t].shape)
                    print("u[t,:].shape", u[t,:].shape)
                    print("di", di)
                    print("np.atleast_2d(u[t,:]).shape", np.atleast_2d(u[t,:]).shape)
                    # print("np.atleast_2d(u[t,:]).reshape(di,1).shape", np.atleast_2d(u[t,:]).reshape(di,1).shape)
                    # print("u[t,:].reshape(di,1).shape", u[t,:].reshape(di,1).shape)
                    # print("u[t,:].reshape(di,1).shape", u[t,:].reshape(di,-1).shape)
                    # print("y", y)
                    print("self.dim_out", self.dim_out)
                    # print("tea[t,:].reshape(self.dim_out,1).T", tea[t,:].reshape(self.dim_out,1).T)
                    # print("y.T", y.T)
                    # print("np.atleast_2d(u[t,:]).shape", np.atleast_2d(u[t,:]).shape)
                    # print("np.atleast_2d(u[t,:].T).shape", np.atleast_2d(u[t,:].T).shape)
                    # print("np.atleast_2d(u[t]).T", np.atleast_2d(u[t]).T.shape)
                    # print("np.atleast_2d(u[t,:]).T", np.atleast_2d(u[t,:]).T.shape)
                # x = (1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, np.atleast_2d(u[t]).T ) + np.dot( self.W, x ) )
                # TODO: this one is equivalent, but don't know which one is faster #TODO have to be tested
                if self.Wfb is None:
                    if verbose:
                        print("u", u.shape)
                        print("x", x.shape)
                        print("self.Win", self.Win.shape)
                        print("self.W", self.W.shape)
                        print("u[t,:]", u[t,:].shape)
                        print("atleast_2d(u)[t,:]", np.atleast_2d(u)[t,:].shape)
                        print("u[t,:].reshape(di,1)", u[t,:].reshape(di,1).shape)
                        print("x", x)
                        print("DEBUG BEFORE")
                        print("self.W", self.W)
                        print("(1-self.lr) * x", (1-self.lr) * x)
                        print("np.dot( self.Win, u[t,:].reshape(di,1) )", np.dot( self.Win, u[t,:].reshape(di,1) ))
                        print("np.dot( self.W, x )", np.dot( self.W, x ))
                    x = ((1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, u[t,:].reshape(di,1) ) + np.dot( self.W, x ) )).astype(self.typefloat)
                    if verbose:
                        print("DEBUG AFTER")
                        print("x.shape", x.shape)
                        print("x", x)
                    # raw_input()
                else:
                    x = ((1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, u[t,:].reshape(di,1) ) + np.dot( self.W, x ) + np.dot( self.Wfb, self.fbfunc(y) ) )).astype(self.typefloat)
                    y = tea[t,:].reshape(self.dim_out,1).astype(self.typefloat)


                if t >= wash_nr_time_step:
                    # X[:,t-initLen] = np.vstack((1,u,x))[:,0]
                    if verbose:
                        print("x.shape", x.shape)
                        print("x", x)
                        print("x.reshape(-1,).shape", x.reshape(-1,).shape)
                        print("all_int_states[j][:,t-wash_nr_time_step].shape", all_int_states[j][:,t-wash_nr_time_step].shape)
                        # raw_input()
                        if self.Wfb is not None:
                            print("y.shape", y.shape)
                            print("y.reshape(-1,).shape", y.reshape(-1,).shape)
                            print("tea[t,:].shape", tea[t,:].shape)
                            print("tea[t,:].reshape(-1,).shape", tea[t,:].reshape(-1,).shape)
                            print("tea[t,:].reshape(-1,).T", tea[t,:].reshape(-1,).T)
                            print("y.T", y.T)
                            print("(y.reshape(-1,) == tea[t,:].reshape(-1,))", (y.reshape(-1,).shape == tea[t,:].reshape(-1,)))
                    if self.Wfb is not None:
                        assert all(y.reshape(-1,) == tea[t,:].reshape(-1,))
                    if verbose:
                        print("x", x)
                        print("x.reshape(-1,)", x.reshape(-1,))
                    #TODO: add option to enable direct connection from input to output to be learned
                        # need to remember the input at this stage
                    all_int_states[j][:,t-wash_nr_time_step] = x.reshape(-1,).astype(self.typefloat)
                    all_teachers[j][:,t-wash_nr_time_step] = tea[t,:].reshape(-1,).astype(self.typefloat)

        if verbose:
            print("all_int_states", all_int_states)
            print("len(all_int_states)", len(all_int_states))
            print("all_int_states[0].shape", all_int_states[0].shape)
            print("all_int_states[0][:5,:15] (5 neurons on 15 time steps)", all_int_states[0][:5,:15])
            print("all_int_states.count(None)", all_int_states.count(None))
        # TODO: change the 2 following lines according to this error:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # assert all_int_states.count(None) == 0 # check if some input/teacher pass was not done
        # assert all_teachers.count(None) == 0 # check if some input/teacher pass was not done
        self.check_values(array_or_list=all_int_states, value=None)
        self.check_values(array_or_list=all_teachers, value=None)

        # concatenate the lists
        X = np.hstack(all_int_states).astype(self.typefloat)
        Y = np.hstack(all_teachers).astype(self.typefloat)
        if verbose:
            print("X.shape", X.shape)
            print("Y.shape", Y.shape)
        # Adding ones for regression with biais b in (y = a*x + b)
        X = np.vstack((np.ones((1,X.shape[1]),dtype=self.typefloat),X))
        if verbose:
            print("X.shape", X.shape)
        # raw_input()

        # train the output
        X_T = X.T # dim of X_T (nr of time steps, nr of neurons)
        # Yt = Y.T # dim of Y_T (output_dim, nr_of_time_steps)
        if verbose:
            print("X_T.shape", X_T.shape)
            print("Y.shape", Y.shape)

        if self.reg_model is not None:
            # scikit learn linear models are used for interpolation
            Wout = self._linear_model_solving(X, Y)

        elif self.ridge is not None:
            # use ridge regression (linear regression with regularization)
            if verbose:
                print("USING RIDGE REGRESSION")
                print("X", X.shape)
                print("X_T", X_T.shape)
                print("Y", Y.shape)
                print("N", self.N)
            # Wout = np.dot(np.dot(Yt,X_T), linalg.inv(np.dot(X,X_T) + \
            ridgeid = (self.ridge*np.eye(1+self.N)).astype(self.typefloat)
            Wout = np.dot(np.dot(Y,X_T), linalg.inv(np.dot(X,X_T) + \
                    ridgeid ) )
                    # self.ridge*np.eye(1+inSize+resSize) ) )

            ### Just if you want to try the difference between scipy.linalg and numpy.linalg which does not give the same results
                ### For more info, see https://www.scipy.org/scipylib/faq.html#why-both-numpy-linalg-and-scipy-linalg-what-s-the-difference
        #    np_Wout = np.dot(np.dot(Yt,X_T), np.linalg.inv(np.dot(X,X_T) + \
        #        reg*np.eye(1+inSize+resSize) ) )
        #    print("Difference between scipy and numpy .inv() method:\n\tscipy_mean_Wout="+\
        #        str(np.mean(Wout))+"\n\tnumpy_mean_Wout="+str(np.mean(np_Wout)))
        else:
            # use pseudo inverse
            if verbose:
                print("USING PSEUDO INVERSE")
            # Wout = np.dot( Yt, linalg.pinv(X) )
            Wout = np.dot( Y, linalg.pinv(X) )

        # saving the output matrix in the ESN object for later use
        self.Wout = Wout
        # saving the last state of the reservoir, in case we want to run the reservoir from the last state on
        self.x = x
        y = all_teachers[-1][:,-1] #the last time step of the last teacher
        self.y = y #useful when we will have feedback
        if verbose:
            print("Wout.shape", Wout.shape)
            print("all_int_states[0].shape", all_int_states[0].shape)

        # return all_int_states
        return [st.T for st in all_int_states]

    def _linear_model_solving(self, X, Ytarget):
        """
            Uses regression method provided during network instanciation to return W such as W * X ~= Ytarget
            First row of X MUST be only ones.
        """
        # Learning of the model (first row of X, which contains only ones, is removed)
        self.reg_model.fit(X[1:, :].T, Ytarget.T)

        # linear_model provides Matrix A and Vector b such as A * X[1:, :] + b ~= Ytarget
        A = np.asmatrix(self.reg_model.coef_)
        b = np.asmatrix(self.reg_model.intercept_).T

        # Then the matrix W = "[b | A]" statisfies "W * X ~= Ytarget"
        return np.asarray(np.hstack([b, A]))


    def run(self, inputs, reset_state=True, verbose=False):
        """
        Dimensions:
            Inputs:
                - inputs: list of numpy array item with dimension (nr_time_step, input_dimension)
            Outputs:
                - all_int_states: list of numpy array item with dimension
                    - during the execution of this method : (N, nr_time_step)
                    - returned dim (nr_time_step, N)
                - all_outputs: list of numpy array item with dimension
                    - during the execution of this method : (output_dim, nr_time_step)
                    - returned dim (nr_time_step, output_dim)

        - float32 : use float32 precision for training the reservoir instead of the default float64 precision
        """
        ## Autochecks
        self.autocheck_io(inputs=inputs)

        if reset_state:
            x = np.zeros((self.N,1),dtype=self.typefloat)
            if self.Wfb is not None:
                y = np.zeros((self.dim_out,1),dtype=self.typefloat)
        else:
            # run the trained ESN in a generative mode. no need to initialize here,
            # because x is initialized with training data and we continue from there
            x = self.x
            if self.Wfb is not None:
                y = (self.y).astype(self.typefloat)

        # change of variable for conveniance in equation
        di = self.dim_inp

        all_int_states = [None]*len(inputs)
        all_outputs = [None]*len(inputs)

        # run reservoir over the inputs, to save the internal states and the teachers (desired outputs) in lists
        for (j, inp) in enumerate(inputs):

            if self.in_bias:
                u = np.column_stack((np.ones((inp.shape[0],1)),inp))
            else:
                u = inp
            u = u.astype(self.typefloat)
            # reset the states or not, that is the question
            if reset_state:
                x = np.zeros((self.N,1),dtype=self.typefloat)#.astype(self.typefloat) # current internal state initialized at zero
                if self.Wfb is not None:
                    y = np.zeros((self.dim_out,1),dtype=self.typefloat)

            all_int_states[j] = np.zeros((self.N,inp.shape[0]),dtype=self.typefloat)
            all_outputs[j] = np.zeros((self.Wout.shape[0],inp.shape[0]),dtype=self.typefloat)

            # out = np.zeros((self.Wout.shape[0],inp))

            # Y = np.zeros((outSize,testLen))
            # u = data[trainLen]
            for t in range(inp.shape[0]): # for each time step in the input
                # x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
                if self.Wfb is None:
                    x = (1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, u[t,:].reshape(di,1) ) + np.dot( self.W, x ) )
                else:
                    x = (1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, u[t,:].reshape(di,1) ) + np.dot( self.W, x ) + np.dot( self.Wfb, self.fbfunc(y) ) )
                # y = np.dot( Wout, np.vstack((1,u,x)) )


                y = np.dot( self.Wout, np.vstack((1,x)) ).astype(self.typefloat)

                """
                print("x :",x.dtype) #float32
                print("vstack : ",np.vstack((1,x)).dtype) #float64, can't change the dtype of vstack
                print("wout : ",self.Wout.dtype) #float32
                print("---->",y.dtype) #float64
                """

                #we have to convert y

                if verbose:
                    print("x.shape", x.shape)
                    print("np.vstack((1,x)).shape", np.vstack((1,x)).shape)
                    print("y.shape", y.shape)

                # Y[:,t] = y
                # u = data[trainLen+t+1]
                if verbose:
                    print("x.reshape(-1,).shape", x.reshape(-1,).shape)
                    print("y.reshape(-1,).shape", y.reshape(-1,).shape)
                    print("all_int_states[j][:,t].shape", all_int_states[j][:,t].shape)
                    print("all_outputs[j][:,t].shape", all_outputs[j][:,t].shape)
                all_int_states[j][:,t] = x.reshape(-1,)
                all_outputs[j][:,t] = y.reshape(-1,)


        # saving the last state of the reservoir, in case we want to run the reservoir from the last state on
        self.x = x
        self.y = y #useful when we will have feedback

        if verbose:
            print()
            print("len(all_int_states)", len(all_int_states))
            print("len(all_outputs)", len(all_outputs))
            print("all_int_states[0].shape", all_int_states[0].shape)
            print("all_outputs[0].shape", all_outputs[0].shape)

        # return all_outputs, all_int_states
        return [st.T for st in all_outputs], [st.T for st in all_int_states]

    def print_trained_esn_info(self):
        print("esn.Win", self.Win)
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
        print("esn.Wout", self.Wout)
        print("esn.Wout max", np.max(self.Wout))
        print("esn.Wout min", np.min(self.Wout))
        print("esn.Wout mean", np.mean(self.Wout))
        print("esn.Wout median", np.median(self.Wout))
        print("esn.Wout std", np.std(self.Wout))
