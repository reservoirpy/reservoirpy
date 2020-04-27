# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
@author: xavier.hinaut #/at\# inria.fr
Copyright Xavier Hinaut 2019-2020
MIT License
"""
import numpy as np
import matplotlib.pyplot as plt

import pickle

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def byteify(input):
    """
    There's no built-in option to make the json module functions return byte
    strings instead of unicode strings. However, this short and simple recursive
    function will convert any decoded JSON object from using unicode strings to
    UTF-8-encoded byte strings
    Just call this on the output you get from a json.load or json.loads call.

    http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python#13105359
    """
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

if __name__=='__main__':
    """
    Example how to run it (on 2 lines):
    > python hyperopt_analyse_error.py -l -t your_folder_with_hp_exploration_files/hyperopt_trials.pkl
      -o your_output_folder/error_vs_
    > python hyperopt_analyse_error.py -t hyper-explorations/2019-04-24_19h32__user1_1556127146038513/hyperopt_trials.pkl -o hyper-explorations/2019-04-24_19h32__user1_1556127146038513/error_vs_
    """
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=str, help= "file containing all the trials of an experiment")
    parser.add_argument("--title", type = str, help= "title of the figure", default = None)
    parser.add_argument("-o", "--output", type=str, help="file name to save the figure", default = None)
    parser.add_argument("-l", "--logparam", action = "store_true", help= "if set then the axis of the param will be in log scale")
    parser.add_argument("-p", "--plot", action = "store_true", help= "if set then we will plot the images in windows instead of saving them")
    parser.add_argument("-v", "--verbose", action = "store_true", help= "show more information")
    parser.add_argument("-s", "--subset", type=int, default = None, help= "Do not use all trials, but keep the S number of trials indicated by -s S option. This option is useful when you analyse trials saved before the hp search ended.")
    # parser.add_argument("-f", "--filename", type=str, help="json file containing parameters for experiment (root directory is assumed to be params/)")
    # parser.add_argument("-s", "--simulationname", type=str, help="simulation_name: this will be use for the file and saving directory)")
    # parser.add_argument("-l", "--lookup", action='store_true', help="lookup_search_space: Plot the search space given instead of launching a parameter search.")
    args = parser.parse_args()

    print("args.trials", args.trials)
    trials = load(args.trials)
    nr_of_evals = len(trials.trials)
    print("nr of evals:", nr_of_evals)
    if args.subset is not None:
        loss = np.empty((args.subset,))
    else:
        loss = np.empty((nr_of_evals,))
    param = np.empty_like(loss)

    # print("par", trials.trials["misc"]["vals"]

    ss_params = trials.trials[0]["misc"]["vals"].keys()
    print("params from search space:", ss_params)

    if args.verbose:
        print("keys of trials.result dictionary:", trials.trials[0]["result"].keys())

    # params to plot without log_scale
    params_not_log = ['fb_func','fb_proba','thres_repl_unfrq_w', 'wash_nr_time_step']

    for current_param in ss_params:
        for i, t in enumerate(trials.trials):
            if args.verbose:
                print("i=", i)
            if args.subset is not None and i >= args.subset:
                break
            # if i >= nr_of_evals - 1:
            #     break
            loss[i] = t["result"]["loss"]
            param[i] = t["misc"]["vals"][current_param][0]

        f = plt.figure(figsize = (10,10))
        p = f.add_subplot(1,1,1)
        if args.title != None:
            p.set_title(args.title)
        else:
            p.set_title("error vs. "+current_param)
        p.set_yscale("log")
        if args.logparam:
            if current_param not in params_not_log:
                p.set_xscale("log")
        p.set_xlabel("Param: "+current_param)
        p.set_ylabel("Error")
        if current_param == 'fb_func':
            p.set_xlabel("Param: "+current_param+"\n[heavyside01, linear01, linear]")

        # p.scatter(x = param, y = loss, color = "blue")
        p.scatter(x = param, y = loss)
        if args.plot:
            plt.show()
        else:
            if args.output is not None:
                save_name = args.output+'-'+current_param
            else:
                dir_path_of_pickle_file = '/'.join(args.trials.split('/')[:-1])
                if dir_path_of_pickle_file == '':
                    dir_path_of_pickle_file = '.'
                dir_path_of_pickle_file += '/'
                print("saving directory:", dir_path_of_pickle_file)
                if args.subset:
                    sub_name = str(args.subset)
                else:
                    sub_name = ''
                print("saving under file name:", dir_path_of_pickle_file+sub_name+'trials_error-vs-'+current_param)
                save_name = dir_path_of_pickle_file+sub_name+'trials_error-vs-'+current_param
            f.savefig(save_name)
