# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
@author: Xavier HINAUT
xavier.hinaut #/at\# inria.fr
Copyright Xavier Hinaut 2018
"""

import numpy as np
from scipy import linalg

class ESN():
    def __init__(self, lr, W, Win, input_bias=True, ridge=None, Wfb=None, fbfunc=None):
        """
        Dimensions of matrices:
            - W : (nr_neurons, nr_neurons)
            - Win : (nr_neurons, dim_input)
            - Wout : (dim_output, nr_neurons)
        """
        self.lr = lr # leaking rate
        self.W = W # reservoir matrix
        self.Win = Win # input matrix
        self.Wout = None
        self.Wfb = Wfb
        self.fbfunc = fbfunc
        self.N = self.W.shape[1] # nr of neurons
        self.in_bias = input_bias
        self.dim_inp = self.Win.shape[1] # dimension of inputs (including the bias at 1)
        self.ridge = ridge
        if self.in_bias:
            pass
        else:
            str_err = "TODO: the ESN class is uncomplete for the case you try to use "
            str_err += "-> the ESN without input bias is not implemented yet."
            raise ValueError, str_err
        if self.Wfb is not None:
            self.dim_out = self.Wfb.shape[1] # dimension of outputs
        else:
            self.dim_out = None




    def train(self, inputs, teachers, wash_nr_time_step, reset_state=True, float32=False, verbose=False):
        #TODO float32 : use float32 precision for training the reservoir instead of the default float64 precision
        """
        Dimensions:
        Inputs:
            - inputs: list of numpy array item with dimension (nr_time_step, input_dimension)
            - teachers: list of numpy array item with dimension (nr_time_step, output_dimension)
        Outputs:
            - all_int_states: list of numpy array item with dimension
                - during the execution of this method : (N, nr_time_step)
                - returned dim (nr_time_step, N)

        - #TODO float32 : use float32 precision for training the reservoir instead of the default float64 precision
        """
        if verbose:
            print "len(inputs)", len(inputs)
            print "len(teachers)", len(teachers)
            print "self.N", self.N
            print "self.W.shape", self.W.shape
            print "self.Win.shape", self.Win.shape

        # 'pre-allocated' memory for the list that will collect the states
        all_int_states = [None]*len(inputs)
        x = np.zeros((self.N,1)) # current internal state initialized at zero
        if self.Wfb is not None:
            y = np.zeros((self.dim_out,1)) # I guess it's the best we can do, because we have no info on the teacher for this time step
            assert teachers[0].shape[1] == self.dim_out # check if output dimension is correct
        x = np.zeros((self.N,1)) # current internal state initialized at zero
        # 'pre-allocated' memory for the list that will collect the teachers (desired outputs)
        all_teachers = [None]*len(teachers)
        # print "all_int_states.count(None)", all_int_states.count(None)
        # raw_input()

        # change of variable for conveniance in equation
        di = self.dim_inp

        if float32:
            #TODO: FINISH TO PUT ALL USEFUL VARIABLES IN float32
            inputs = [aa.astype('float32') for aa in inputs]
            teachers = [aa.astype('float32') for aa in teachers]
            raise Exception, "TODO: float32 option not finished yet!"

        # run reservoir over the inputs, to save the internal states and the teachers (desired outputs) in lists
        for (j, (inp, tea)) in enumerate(zip(inputs, teachers)):
            if verbose:
                print "j:", j
                print "inp.shape", inp.shape
                print "tea.shape", tea.shape

            if self.in_bias:
                u = np.column_stack((np.ones((inp.shape[0],1)),inp))
            else:
                u = inp
            # reset the states or not, that is the question
            if reset_state:
                x = np.zeros((self.N,1)) # current internal state initialized at zero
                if self.Wfb is not None:
                    y = np.zeros((self.dim_out,1))
            else:
                # keep the previously runned state
                pass

            all_int_states[j] = np.zeros((self.N,inp.shape[0]-wash_nr_time_step))
            all_teachers[j] = np.zeros((tea.shape[1],inp.shape[0]-wash_nr_time_step))

            for t in range(inp.shape[0]): # for each time step in the input
                # u = data[t]
                # u = np.atleast_2d(inp[t,:])
                if verbose:
                    print "inp[t,:].shape", inp[t,:].shape
                    print "tea[t,:].shape", tea[t,:].shape
                    print "u.shape", u.shape
                    print "inp[t,:]", inp[t,:]
                    print "u[t].shape", u[t].shape
                    print "u[t,:].shape", u[t,:].shape
                    print "di", di
                    print "u[t,:].reshape(di,1).shape", u[t,:].reshape(di,1).shape
                    # print "y", y
                    print "self.dim_out", self.dim_out
                    # print "tea[t,:].reshape(self.dim_out,1).T", tea[t,:].reshape(self.dim_out,1).T
                    # print "y.T", y.T
                    # print "np.atleast_2d(u[t,:]).shape", np.atleast_2d(u[t,:]).shape
                    # print "np.atleast_2d(u[t,:].T).shape", np.atleast_2d(u[t,:].T).shape
                    # print "np.atleast_2d(u[t]).T", np.atleast_2d(u[t]).T.shape
                    # print "np.atleast_2d(u[t,:]).T", np.atleast_2d(u[t,:]).T.shape
                # x = (1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, np.atleast_2d(u[t]).T ) + np.dot( self.W, x ) )
                # TODO: this one is equivalent, but don't know which one is faster #TODO have to be tested
                if self.Wfb is None:
                    x = (1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, u[t,:].reshape(di,1) ) + np.dot( self.W, x ) )
                else:
                    x = (1-self.lr) * x  +  self.lr * np.tanh( np.dot( self.Win, u[t,:].reshape(di,1) ) + np.dot( self.W, x ) + np.dot( self.Wfb, self.fbfunc(y) ) )
                    y = tea[t,:].reshape(self.dim_out,1)


                if t >= wash_nr_time_step:
                    # X[:,t-initLen] = np.vstack((1,u,x))[:,0]
                    if verbose:
                        print "x.shape", x.shape
                        print "x.reshape(-1,).shape", x.reshape(-1,).shape
                        print "all_int_states[j][:,t-wash_nr_time_step].shape", all_int_states[j][:,t-wash_nr_time_step].shape
                        if self.Wfb is not None:
                            print "y.shape", y.shape
                            print "y.reshape(-1,).shape", y.reshape(-1,).shape
                            print "tea[t,:].shape", tea[t,:].shape
                            print "tea[t,:].reshape(-1,).shape", tea[t,:].reshape(-1,).shape
                            print "tea[t,:].reshape(-1,).T", tea[t,:].reshape(-1,).T
                            print "y.T", y.T
                            print "(y.reshape(-1,) == tea[t,:].reshape(-1,))", (y.reshape(-1,).shape == tea[t,:].reshape(-1,))
                    if self.Wfb is not None:
                        assert all(y.reshape(-1,) == tea[t,:].reshape(-1,))
                    all_int_states[j][:,t-wash_nr_time_step] = x.reshape(-1,)
                    all_teachers[j][:,t-wash_nr_time_step] = tea[t,:].reshape(-1,)

        if verbose:
            print "all_int_states", all_int_states
            print "len(all_int_states)", len(all_int_states)
            print "all_int_states[0].shape", all_int_states[0].shape
            print "all_int_states.count(None)", all_int_states.count(None)
        # TODO: change the 2 following lines according to this error:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # assert all_int_states.count(None) == 0 # check if some input/teacher pass was not done
        # assert all_teachers.count(None) == 0 # check if some input/teacher pass was not done

        # concatenate the lists
        X = np.hstack(all_int_states)
        Y = np.hstack(all_teachers)
        if verbose:
            print "X.shape", X.shape
            print "Y.shape", Y.shape
        # Adding ones for regression with biais b in (y = a*x + b)
        X = np.vstack((np.ones((1,X.shape[1])),X))
        if verbose:
            print "X.shape", X.shape
        # raw_input()

        # train the output
        X_T = X.T # dim of X_T (nr of time steps, nr of neurons)
        # Yt = Y.T # dim of Y_T (output_dim, nr_of_time_steps)
        if verbose:
            print "X_T.shape", X_T.shape
            print "Y.shape", Y.shape
        if self.ridge is not None:
            # use ridge regression (linear regression with regularization)
            if verbose:
                print "USING RIDGE REGRESSION"
            # Wout = np.dot(np.dot(Yt,X_T), linalg.inv(np.dot(X,X_T) + \
            Wout = np.dot(np.dot(Y,X_T), linalg.inv(np.dot(X,X_T) + \
                    self.ridge*np.eye(1+self.N) ) )
                    # self.ridge*np.eye(1+inSize+resSize) ) )

            ### Just if you want to try the difference between scipy.linalg and numpy.linalg which does not give the same results
                ### For more info, see https://www.scipy.org/scipylib/faq.html#why-both-numpy-linalg-and-scipy-linalg-what-s-the-difference
        #    np_Wout = np.dot(np.dot(Yt,X_T), np.linalg.inv(np.dot(X,X_T) + \
        #        reg*np.eye(1+inSize+resSize) ) )
        #    print "Difference between scipy and numpy .inv() method:\n\tscipy_mean_Wout="+\
        #        str(np.mean(Wout))+"\n\tnumpy_mean_Wout="+str(np.mean(np_Wout))
        else:
            # use pseudo inverse
            if verbose:
                print "USING PSEUDO INVERSE"
            # Wout = np.dot( Yt, linalg.pinv(X) )
            Wout = np.dot( Y, linalg.pinv(X) )

        # saving the output matrix in the ESN object for later use
        self.Wout = Wout
        # saving the last state of the reservoir, in case we want to run the reservoir from the last state on
        self.x = x
        y = all_teachers[-1][:,-1] #the last time step of the last teacher
        self.y = y #useful when we will have feedback

        if verbose:
            print "Wout.shape", Wout.shape
            print "all_int_states[0].shape", all_int_states[0].shape

        # return all_int_states
        return [st.T for st in all_int_states]




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
        if reset_state:
            x = np.zeros((self.N,1))
            if self.Wfb is not None:
                y = np.zeros((self.dim_out,1))
        else:
            # run the trained ESN in a generative mode. no need to initialize here,
            # because x is initialized with training data and we continue from there.
            x = self.x
            if self.Wfb is not None:
                y = self.y

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
            # reset the states or not, that is the question
            if reset_state:
                x = np.zeros((self.N,1)) # current internal state initialized at zero
                if self.Wfb is not None:
                    y = np.zeros((self.dim_out,1))

            all_int_states[j] = np.zeros((self.N,inp.shape[0]))
            all_outputs[j] = np.zeros((self.Wout.shape[0],inp.shape[0]))

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
                y = np.dot( self.Wout, np.vstack((1,x)) )

                if verbose:
                    print "x.shape", x.shape
                    print "np.vstack((1,x)).shape", np.vstack((1,x)).shape
                    print "y.shape", y.shape

                # Y[:,t] = y
                # u = data[trainLen+t+1]
                if verbose:
                    print "x.reshape(-1,).shape", x.reshape(-1,).shape
                    print "y.reshape(-1,).shape", y.reshape(-1,).shape
                    print "all_int_states[j][:,t].shape", all_int_states[j][:,t].shape
                    print "all_outputs[j][:,t].shape", all_outputs[j][:,t].shape
                all_int_states[j][:,t] = x.reshape(-1,)
                all_outputs[j][:,t] = y.reshape(-1,)

        # saving the last state of the reservoir, in case we want to run the reservoir from the last state on
        self.x = x
        self.y = y #useful when we will have feedback

        if verbose:
            print ""
            print "len(all_int_states)", len(all_int_states)
            print "len(all_outputs)", len(all_outputs)
            print "all_int_states[0].shape", all_int_states[0].shape
            print "all_outputs[0].shape", all_outputs[0].shape

        # return all_outputs, all_int_states
        return [st.T for st in all_outputs], [st.T for st in all_int_states]

    def print_trained_esn_info(self):
        print "esn.Win", self.Win
        print "esn.Win", self.Win
        print "esn.Win max", np.max(self.Win)
        print "esn.Win min", np.min(self.Win)
        print "esn.Win mean", np.mean(self.Win)
        print "esn.Win median", np.median(self.Win)
        print "esn.Win std", np.std(self.Win)
        print "esn.W", self.W
        print "esn.W max", np.max(self.W)
        print "esn.W min", np.min(self.W)
        print "esn.W mean", np.mean(self.W)
        print "esn.W median", np.median(self.W)
        print "esn.W std", np.std(self.W)
        print "esn.Wout", self.Wout
        print "esn.Wout max", np.max(self.Wout)
        print "esn.Wout min", np.min(self.Wout)
        print "esn.Wout mean", np.mean(self.Wout)
        print "esn.Wout median", np.median(self.Wout)
        print "esn.Wout std", np.std(self.Wout)
