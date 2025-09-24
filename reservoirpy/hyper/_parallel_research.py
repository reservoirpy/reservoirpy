"""Tools for CPU-parallelized hyperparameter optimization."""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import json
import os
import time
import warnings
from glob import glob

import numpy as np

from reservoirpy.utils.random import rand_generator


def _get_conf_from_json(confpath):
    if not (os.path.isfile(confpath)):
        raise FileNotFoundError(f"Training conf '{confpath}' not found.")
    else:
        config = {}
        with open(confpath, "r") as f:
            config = json.load(f)
        return _parse_config(config)


def _parse_config(config):
    required_args = ["exp", "hp_max_evals", "hp_method", "hp_space"]
    for arg in required_args:
        if config.get(arg) is None:
            raise ValueError(f"No {arg} argument found in configuration file.")

    if config["hp_method"] not in ["random"]:
        raise ValueError(
            f"Unknown algorithm: {config['hp_method']}. "
            "Available algorithms for CPU parallelized research: 'random'."
        )

    config["hp_space"] = {arg: _parse_searchspace(specs) for arg, specs in config["hp_space"].items()}

    return config


def _parse_searchspace(specs):
    if specs[0] == "choice":
        # specs: ['choice', e0, ..., eN]
        return lambda rng: specs[1:][rng.integers(0, len(specs) - 1)]
    if specs[0] == "randint":
        # specs: ['randint', low, high]
        return lambda rng: rng.integers(specs[1], specs[2])
    if specs[0] == "uniform":
        # specs: ['uniform', low, high]
        return lambda rng: rng.uniform(specs[1], specs[2])
    if specs[0] == "quniform":
        # specs: ['quniform', low, high, q]
        return lambda rng: np.round(rng.uniform(specs[1], specs[2]) / specs[3]) * specs[3]
    if specs[0] == "loguniform":
        # specs: ['loguniform', low, high]
        return lambda rng: np.exp(rng.uniform(np.log(specs[1]), np.log(specs[2])))
    if specs[0] == "qloguniform":
        # specs: ['qloguniform', low, high, q]
        return lambda rng: np.round(np.exp(rng.uniform(np.log(specs[1]), np.log(specs[2]))) / specs[3]) * specs[3]
    if specs[0] == "normal":
        # specs: ['normal', mu, sigma]
        return lambda rng: specs[1] + specs[2] * rng.randn()
    if specs[0] == "qnormal":
        # specs: ['qnormal', mu, sigma, q]
        return lambda rng: np.round((specs[1] + specs[2] * rng.randn()) / specs[3]) * specs[3]
    if specs[0] == "lognormal":
        # specs: ['lognormal', mu, sigma]
        return lambda rng: np.exp(np.log(specs[1]) + np.log(specs[2]) * rng.randn())
    if specs[0] == "qlognormal":
        # specs: ['qlognormal', mu, sigma, q]
        return lambda rng: np.round(np.exp(np.log(specs[1]) + np.log(specs[2]) * rng.randn()) / specs[3]) * specs[3]


def _worker(objective, dataset, config, kwargs):
    """
    Worker function to evaluate the objective with given parameters.
    Measures execution time and handles exceptions.
    Returns the parameters, result dictionary, and filename for saving results.
    """
    try:
        start = time.time()

        returned_dict = objective(dataset, config, **kwargs)

        end = time.time()
        duration = end - start

        returned_dict["status"] = "ok"
        returned_dict["start_time"] = start
        returned_dict["duration"] = duration

        save_file = f"{returned_dict['loss']:.7f}_results"

    except Exception as e:
        raise e
        start = time.time()

        returned_dict = {
            "status": "fail",
            "start_time": start,
            "error": str(e),
        }

        save_file = f"ERR{start}_results"

    return kwargs, returned_dict, save_file


def _get_report_path(exp_name, base_path=None):
    base_path = "." if base_path is None else base_path

    report_path = os.path.join(base_path, exp_name, "results")

    if not (os.path.isdir(base_path)):
        os.mkdir(base_path)

    if not (os.path.isdir(os.path.join(base_path, exp_name))):
        os.mkdir(os.path.join(base_path, exp_name))

    if not (os.path.isdir(report_path)):
        os.mkdir(report_path)

    return report_path


def parallel_research(
    objective,
    dataset,
    config_path,
    report_path=None,
    n_jobs=-1,
    inner_max_num_threads=None,
):
    """
    Executes a parallelized hyperparameter search on the provided
    objective function using the configuration specified.
    This function uses **process-based parallelism** through the `loky` backend of `joblib`.

    For more details on hyper-parameter search using ReservoirPy, take a look at
    :ref:`/user_guide/hyper.ipynb`.

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
    n_jobs: int, default to -1
        Maximum number of concurrent processes to run.
        If set to -1, it is set to the number of CPUs.
    inner_max_num_threads: int, optional
        Maximum number of threads allowed within each parallel job (e.g., used by
        libraries like NumPy). If set to a positive number, it will limit the
        number of threads per process. If set to -1, there will be no restriction.
        By default, it is calculated as `max(1, number of trials // number of jobs)`.
    """
    from joblib import Parallel, cpu_count, delayed, parallel_config
    from tqdm import tqdm

    config = _get_conf_from_json(config_path)
    report_path = _get_report_path(config["exp"], report_path)

    search_space = config["hp_space"]
    max_evals = config["hp_max_evals"]

    if inner_max_num_threads is None:
        inner_max_num_threads = max(1, cpu_count() // max_evals)
    elif inner_max_num_threads == -1:
        inner_max_num_threads = None

    rng = rand_generator(config.get("seed"))

    params_list = [{arg: f(rng) for arg, f in search_space.items()} for _ in range(max_evals)]

    best_params = None
    best_loss = float("inf")

    with parallel_config(backend="loky", inner_max_num_threads=inner_max_num_threads):
        it = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_worker)(objective, dataset, config, kwargs) for kwargs in params_list
        )

    pbar = tqdm(iterable=it, total=len(params_list), unit="trial", postfix="best loss=?")

    for kwargs, returned_dict, save_file in pbar:
        try:
            for key in kwargs:
                if isinstance(kwargs[key], np.integer):
                    kwargs[key] = int(kwargs[key])
            json_dict = {"returned_dict": returned_dict, "current_params": kwargs}
            save_file = os.path.join(report_path, save_file)
            nb_save_file_with_same_loss = len(glob(f"{save_file}*"))
            save_file = f"{save_file}_{nb_save_file_with_same_loss+1}call.json"
            with open(save_file, "w+") as f:
                json.dump(json_dict, f, indent=2)
        except Exception as e:
            warnings.warn("Results of current simulation were NOT saved " "correctly to JSON file.")
            warnings.warn(str(e))

        if returned_dict["loss"] < best_loss:
            best_params = kwargs
            best_loss = returned_dict["loss"]
            pbar.set_postfix({"best loss": best_loss})

    return best_params, best_loss