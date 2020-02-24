# -*- coding: utf-8 -*-
"""
Created on 17 feb. 2020

@author: Xavier HINAUT
xavier.hinaut@inria.fr
"""
import os
import json
from datetime import datetime

import numpy as np

class Config(object):
    """Configuration container for ESN training benchmark.
    """

    def __init__(self,
                n_inputs: int,
                n_outputs: int,
                input_bias=True,
                n_reservoir=300,
                leak_rate=0.2,
                warmup=1000,
                spectral_radius=1.25,
                input_scaling=1.,
                sparsity_W=0.2,
                sparsity_Win=1.,
                sparsity_Wfb=1.,
                regularization=1e-8,
                name: str=None):
        """Configuration container for ESN training benchmark.

        Arguments:
            n_inputs {int} -- Dimension of the input vectors.
            n_outputs {int} -- Dimension of the output vectors.

        Keyword Arguments:
            input_bias {bool} -- Enable or disable input bias. (default: {True})
            n_reservoir {int} -- Number of neurons instanciated. (default: {300})
            leak_rate {float} -- Leak rate of ESN. (default: {0.2})
            warmup {int} -- Number of timesteps used to "wash" the synaptic connections to initiate training. (default: {1000})
            spectral_radius {float} -- Spectral radius to scale weight matrix. (default: {1.25})
            input_scaling {[float]} -- Scaling of input weights. (default: {1.})
            sparsity_W {float} -- Probability of connection between reservoir neurons. (default: {0.2})
            sparsity_Win {[float]} -- Probability of connection between inputs and reservoir. (default: {1.})
            sparsity_Wfb {[float]} -- Probability of connection between feddback and reservoir. (default: {1.})
            regularization {[float]} -- Ridge regularization rate usedd for ridge regression. (default: {1e-8})
            name {[str]} -- Optionnal name or description of the model. (default: {None})

        Returns:
            [Config] -- A configuration container.
        """

        self.warmup         = warmup
        self.in_dim         = n_inputs
        self.out_dim        = n_outputs
        self.input_bias     = input_bias
        self.N              = n_reservoir
        self.lr             = leak_rate
        self.sr             = spectral_radius
        self.in_scaling     = input_scaling
        self.sparsity_W     = sparsity_W
        self.sparsity_Win   = sparsity_Win
        self.sparsity_Wfb   = sparsity_Wfb
        self.regularization = regularization

        self.serial = Config._get_serial()

        self.name = name
        if name is None:
            self.name = self.serial


    def __repr__(self):
        d = self.__dict__
        s = f"Config {self.serial}: {self.name}"
        for k, v in d.items():
            s += f"{k}:  {v}\n"
        return s


    @staticmethod
    def _get_serial():
        return datetime.today().strftime("%Y%m%d-%H%M%S")


    @staticmethod
    def from_json(file_path):
        """Load configuration from JSON file.
        
        Arguments:
            file_path {str or path-like object} -- Path to JSON configuration file.
        
        Returns:
            [Config] -- A configuration container.
        """

        d = {}
        with open(file_path, "r") as f:
            d = json.load(f)

        attributes = {}
        for k, v in d.items():
            if k != "serial":
                attributes[k] = v

        conf = Config(**attributes)
        conf.serial = d["serial"]

        return conf


    def W_conf(self):
        """Configuration of reservoir weights matrix.
        """
        W_conf = {
            "N": self.N,
            "spectral_radius": self.sr,
            "proba": self.sparsity_W
        }

        return W_conf


    def Win_conf(self):
        """Configuration of input weights matrix.
        """
        Win_conf = {
            "nbr_neuron": self.N,
            "dim_input": self.in_dim,
            "input_scaling": self.in_scaling,
            "proba": self.sparsity_Win,
            "input_bias": self.input_bias
        }

        return Win_conf


    def ESN_conf(self):
        """Configuration of ESN object.
        """
        ESN_conf = {
            "lr": self.lr,
            "input_bias": self.input_bias,
            "ridge": self.regularization
        }

        return ESN_conf


    def training_conf(self):
        """Configuration of ESN train function.
        """
        train_conf = {
            "wash_nr_time_step": self.warmup
        }

        return train_conf


    def save(self, directory=None, esn=None, plots=None):
        """Save a configuration container to JSON format.
        
        Keyword Arguments:
            file_path {[str or path-like object]} -- File where to write object. Default is ̀./logs/{datetime}`. (default: {None})
        """

        if directory is None:
            directory = os.path.join(".", "models")
            
        if not(os.path.isdir(directory)):
            os.mkdir(directory)
        
        sub_dir = os.path.join(directory, f"{self.name}-{self.serial}")

        if not(os.path.isdir(sub_dir)):
            os.mkdir(sub_dir)

        if not(esn is None):
            W = esn.W
            Win = esn.Win
            Wout = esn.Wout
            
            w_path = os.path.join(sub_dir, "weigths")
            if not(os.path.isdir(w_path)):
                os.mkdir(w_path)
                
            np.save(os.path.join(w_path, "W"), W)
            np.save(os.path.join(w_path, "Win"), Win)
            np.save(os.path.join(w_path, "Wout"), Wout)
            
        if not(plots is None):
            plots_path = os.path.join(sub_dir, "reports")
            if not(os.path.isdir(plots_path)):
                os.mkdir(plots_path)
            
            for title, p in plots.items():
                p.savefig(os.path.join(plots_path, f"{title}.png"))
            
        file_path = os.path.join(sub_dir, "config.json")
        with open(file_path, "w+") as f:
            json.dump(self.__dict__, f)

        self.serial = Config._get_serial()