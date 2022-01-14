"""*Hyperopt* wrapper tools for hyperparameters optimization.

"""
import os
import time
import json
import warnings

from os import path
from functools import partial
from glob import glob

import numpy as np


def _get_conf_from_json(confpath):
    if not(path.isfile(confpath)):
        raise FileNotFoundError(f"Training conf '{confpath}' not found.")
    else:
        config = {}
        with open(confpath, "r") as f:
            config = json.load(f)
        return _parse_config(config)


def _parse_config(config):

    import hyperopt as hopt

    required_args = ["exp", "hp_max_evals", "hp_method", "hp_space"]
    for arg in required_args:
        if config.get(arg) is None:
            raise ValueError(f"No {arg} argument found in config file.")

    if config["hp_method"] not in ["tpe", "random"]:
        raise ValueError(f"Unknow hyperopt algorithm: {config['hp_method']}. "
                         "Available algorithms: 'random', 'tpe'.")
    else:
        if config["hp_method"] == "random":
            config["hp_method"] = partial(hopt.rand.suggest)
        if config["hp_method"] == "tpe":
            config["hp_method"] = partial(hopt.tpe.suggest)

    space = {}
    for arg, specs in config["hp_space"].items():
        space[arg] = _parse_hyperopt_searchspace(arg, specs)

    config["hp_space"] = space

    return config


def _parse_hyperopt_searchspace(arg, specs):

    import hyperopt as hopt

    if specs[0] == "choice":
        return hopt.hp.choice(arg, specs[1:])
    if specs[0] == "randint":
        return hopt.hp.randint(arg, *specs[1:])
    if specs[0] == "uniform":
        return hopt.hp.uniform(arg, *specs[1:])
    if specs[0] == "quniform":
        return hopt.hp.quniform(arg, *specs[1:])
    if specs[0] == "loguniform":
        return hopt.hp.loguniform(arg, np.log(specs[1]), np.log(specs[2]))
    if specs[0] == "qloguniform":
        return hopt.hp.qloguniform(arg, np.log(specs[1]), np.log(specs[2]), specs[3])
    if specs[0] == "normal":
        return hopt.hp.normal(arg, *specs[1:])
    if specs[0] == "qnormal":
        return hopt.hp.qnormal(arg, *specs[1:])
    if specs[0] == "lognormal":
        return hopt.hp.lognormal(arg, np.log(specs[1]), np.log(specs[2]))
    if specs[0] == "qlognormal":
        return hopt.hp.qlognormal(arg, np.log(specs[1]), np.log[specs[2]], specs[3])


def _get_report_path(exp_name, base_path=None):

    base_path = '.' if base_path is None else base_path

    report_path = path.join(base_path, exp_name, 'results')

    if not(path.isdir(base_path)):
        os.mkdir(base_path)

    if not(path.isdir(path.join(base_path, exp_name))):
        os.mkdir(path.join(base_path, exp_name))

    if not(path.isdir(report_path)):
        os.mkdir(report_path)

    return report_path


def research(objective, dataset, config_path, report_path=None):
    """
    Wrapper for hyperopt fmin function. Will run hyperopt fmin on the
    objective function passed as argument, on the data stored in the
    dataset argument.

    Note
    ----

        Installation of :mod:`hyperopt` is required to use this function.

    Parameters
    ----------
    objective : Callable
        Objective function defining the function to
        optimize. Must be able to receive the dataset argument and
        all parameters sampled by hyperopt during the search. These
        parameters must be keyword arguments only without default value
        (this can be achieved by separating them from the other arguments
        with an empty starred expression. See examples for more info.)
    dataset : tuple or lists or arrays of data
        Argument used to pass data to the objective function during
        the hyperopt run. It will be passed as is to the objective
        function : it can be in whatever format.
    config_path : str or Path
        Path to the hyperopt experimentation configuration file used to
        define this run.
    report_path : str, optional
        Path to the directory where to store the results of the run. By default,
        this directory is set to be {name of the experiment}/results/.
    """
    import hyperopt as hopt

    config = _get_conf_from_json(config_path)
    report_path = _get_report_path(config["exp"], report_path)

    def objective_wrapper(kwargs):

        try:
            start = time.time()

            returned_dict = objective(dataset, config, **kwargs)

            end = time.time()
            duration = end - start

            returned_dict['status'] = hopt.STATUS_OK
            returned_dict['start_time'] = start
            returned_dict['duration'] = duration

            save_file = f"{returned_dict['loss']:.7f}_hyperopt_results"

        except Exception as e:
            raise e
            start = time.time()

            returned_dict = {
                'status': hopt.STATUS_FAIL,
                'start_time': start,
                'error': str(e),
            }

            save_file = f"ERR{start}_hyperopt_results"

        try:
            json_dict = {'returned_dict': returned_dict, 'current_params': kwargs}
            save_file = path.join(report_path, save_file)
            nb_save_file_with_same_loss = len(glob(f"{save_file}*"))
            save_file = f"{save_file}_{nb_save_file_with_same_loss+1}call.json"
            with open(save_file, "w+") as f:
                json.dump(json_dict, f)
        except Exception as e:
            warnings.warn("Results of current simulation were NOT saved "
                          "correctly to JSON file.")
            warnings.warn(str(e))

        return returned_dict

    search_space = config["hp_space"]

    trials = hopt.Trials()

    if config.get("seed") is None:
        rs = np.random.RandomState()
    else:
        rs = np.random.RandomState(config["seed"])

    best = hopt.fmin(objective_wrapper,
                     space=search_space,
                     algo=config["hp_method"],
                     max_evals=config['hp_max_evals'],
                     trials=trials,
                     rstate=rs)

    return best, trials
