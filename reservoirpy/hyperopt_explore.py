# -*- coding: utf-8 -*-
#!/usr/bin/env python -W ignore::DeprecationWarning
"""
@author: xavier.hinaut #/at\# inria.fr
Copyright Xavier Hinaut 2019-2020
MIT License

Some interesting comments on Hyperopt:
http://stackoverflow.com/questions/24673739/hyperopt-set-timeouts-and-modify-space-during-execution
"""
# imports relative to hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval, rand
from hyperopt.pyll.stochastic import sample
from functools import partial
# general imports
import numpy as np
import time
import csv
import json
import pprint
import pickle

""" TODO: KEEP AND MODIFY THESE GLOBAL VARIABLES """
### Variables globales modifiables
SAVE_DIR = "hyper-explorations/"
VERBOSE = False
## Variables globales spécifiques à hyperopt (l'optimisateur bayésien)
HP_MAX_EVALS = 20 # TODO: MAXIMUM NUMBER OF TRIALS THAT WILL BE PERFORMED
HP_WARMUP_EVALS = round(HP_MAX_EVALS/3) # NUMBER OF RANDOM TRIALS TO BOOTSTRAP THE ALGORITHM # Nombre de stimulus aléatoires qui vont être exploré avant que l'optimisation bayésienne se mette en marche !
SAVE_TRIAL_OBJ_EVERY = 3 # # TODO: (change if you want less files produced) THE TRIAL OBJECT (containing all trials) IS SAVED EVERY ... TIMES # On sauvegarde le fichier général tous les ... fois
CURR_EVAL = 0 #INIT OF VARIABLE

""" TODO: MAKE YOUR OWN GLOBAL VARIABLES BELOW:
USE MIN AND MAX SCORE IF YOU NEED YOUR METHOD TO BE OPTIMIZED GIVES A SCORE (instead of a loss) AND YOU NEED TO NORMALIZE IT """
# INSERT YOUR OWN PARAMETERS BELOW >>>
SCORE_MIN = 0 #L'utilisateur voit le score s'afficher entre -50 et +50, le score renvoyer est comme le reste des paramètres codé sur int[0;127]
SCORE_MAX = 127
FICHIER_OPTIM2LIVE = '_PythonToLive.txt' #fichier de sortie dans lequel on va écrire les paramètres à tester
FICHIER_LIVE2OPTIM = '_LiveToPython.txt' #fichier d'entrée dans lequel on va lire les résultats pour les paramètres envoyés précédemment
DELIMITATEUR = " "
WAIT_TIME_READ_FILE = 0.5 # in seconds
# <<< INSERT YOUR OWN PARAMETERS HERE


"""
MAIN FUNCTION which exchange data with Ableton Live module
"""
def function_that_returns_the_loss_given_the_parameters(params):
    """
    TODO: MODIFY THIS FUNCTION TO CALL YOUR METHOD, MODEL, NEURAL NETWORK, etc. THAT NEED TO BE OPTIMIZED
    """

    # Here are done the computations, discrimination, model evaluation, ....
    pass
    score = SCORE_MAX ## dummy score

    # Normaliser le score pour en faire une "erreur" (loss) entre 0 et 1
    score_normalise = 1 - (score - SCORE_MIN)/(SCORE_MAX - SCORE_MIN)

    print("SCORE USER + SCORE NORMALISé :", score, score_normalise)

    return score_normalise, score, volume

def objective_dic(params):
    """
    Objective function that takes a dictionary as argument.
    This is the function the optimizer will try to minimize by sampling parameters that are given in input.
    """


    global SAVE_TRIAL_OBJ_EVERY
    global CURR_EVAL
    CURR_EVAL += 1

    ### saving the trial data every SAVE_TRIAL_OBJ_EVERY
    # we save the trail at the beginning because we cannot save the results of
    #   the current trial during the call of the method we are in.
    if (CURR_EVAL-1)%SAVE_TRIAL_OBJ_EVERY == 0:
        try:
            save(trials, SAVE_DIR+"hyperopt_trials_eval"+str(CURR_EVAL-1)+".pkl")
            # save([t for i, t in enumerate(trials.trials) if i < len(trials.trials) ], SAVE_DIR+"hyperopt_trials_eval"+str(CURR_EVAL)+".pkl")
        except:
            print("!!! WARNING: COULD NOT SAVE THE PARTIAL TRIAL OBJECT.")

    print("")
    print("***      HP: params given to objective function:", params)
    print("         current eval : ", CURR_EVAL)
    print("")

    current_params = params

    """ TODO: we do not need seed for this experiment, but maybe you need to do so"""
    # current_params.update({'seed': seed})
    seed = int(time.time()*10**6)

    print("HP: all current params:", current_params)
    print("")

    """ TODO: INIT EXPERIMENT """
    # compexp = experiment.Experiment(current_params)
    # compexp.launch_exp()
    """ no init needed here, but maybe you need it """

    start_time = time.time()
    try:
        """ TODO: LAUNCH EXPERIMENT & RETRIEVE LOSS and some supplementary params """
        score_norm, score_user, volume_user = function_that_returns_the_loss_given_the_parameters(params)
        # std = None
        """ TODO: GET OTHER USEFUL DETAILS ON EACH TRIAL """
        end_time = time.time()
        run_time = end_time-start_time
        # loss_tmp, time_loss = geht_los_einfach(mean, std, run_time) #geht_los(mean, std, run_time)
        loss_tmp, time_loss = score_norm, None
        returned_dic = {'loss': loss_tmp,
                    'status': STATUS_OK,
                    # -- store other results like this
                    # 'loss_variance': std**2,
                    'eval_time': time.time(),
                    'true_loss': score_norm,
                    'start_time': start_time,
                    'end_time': end_time,
                    'run_time': run_time,
                    'time_loss': time_loss,
                    'score_user': score_user,
                    'volume_user': volume_user,
                    # For info
    #                    'true_loss': type float if you pre-compute a test error for a validation error loss, store it here so that Hyperopt plotting routines can find it,
    #                    'true_loss_variance': type float variance in test error estimator,
    #                    'other_stuff': {'list_args': list_args,
    #                                    'general_param': general_param},
    #                    # -- attachments are handled differently
    #                    'attachments':
    #                        {'time_module': pickle.dumps(time.time)}
                        }
    except Exception as e:
        print("error", str(e))
        returned_dic = {'status': STATUS_FAIL,
                      'loss': 1.0, #debug: probably useless
                      'exception': str(e),
                    # -- store other results like this
                        'eval_time': time.time(),
#                        'other_stuff': {'list_args': list_args,
#                                        'general_param': general_param},
#                        # -- attachments are handled differently
#                        'attachments':
#                            {'time_module': pickle.dumps(time.time)}
                            }

    ### SAVE FILE OF PAST SIMULATION
    try:
        if score_norm is not None:
            json_filename = 'err%.3f_'%score_norm+'hyperopt_results_1call_s'+str(seed)
        else:
            json_filename = 'exception_error__'+'hyperopt_results_1call_s'+str(seed)
        json_dic = {'returned_dic': returned_dic,
                    'current_params': current_params}

        with open(SAVE_DIR+json_filename+'.json','w') as fout:
            # json.dump(json_dic, fout)
            json.dump(json_dic, fout, separators=(',', ':'), sort_keys=True, indent=4)
    except:
        print("WARNING: Results of current simulation were NOT saved correctly to JSON file.")
    # and test if results were saved correctly
    try:
        with open(SAVE_DIR+json_filename+'.json','r') as fin:
            data = json.load(fin)
            #pprint.pprint(byteify(data)) # in order to remove "u" in all keys of dictionary
        print("### Results of current simulation were saved correctly to JSON file. ###")
    except:
        print("WARNING: Results of current simulation were NOT saved correctly to JSON file.")
    ###

    print("returned_dic:", returned_dic)

    return returned_dic


def hp_search(sim_name):
    """
    Main method that is called by the __main__ method when executed.
    It is setting all things necessary for the parameter search and saving the results.
    """

    global SAVE_DIR
    """ Create a unique folder for the experiment with ID value (i.e. time stamp) """
    SAVE_DIR += time.strftime("%Y-%m-%d_%Hh%M__")+sim_name+"_"+str(int(time.time()*10**6))+"/"
    import os
    os.mkdir(SAVE_DIR)
    global sim_name_
    sim_name_ = sim_name

    # Save global parameters in the params variable
    params = {
                'SAVE_DIR': SAVE_DIR,
                'SAVE_TRIAL_OBJ_EVERY': SAVE_TRIAL_OBJ_EVERY,
                'HP_WARMUP_EVALS': HP_WARMUP_EVALS,
                'HP_MAX_EVALS': HP_MAX_EVALS,
                # TODO: INSERT YOUR OWN PARAMETERS BELOW >>>
                'FICHIER_OPTIM2LIVE': sim_name_+FICHIER_OPTIM2LIVE,
                'FICHIER_LIVE2OPTIM': sim_name_+FICHIER_LIVE2OPTIM,
                'DELIMITATEUR': DELIMITATEUR,
                'SCORE_MIN': SCORE_MIN,
                'SCORE_MAX': SCORE_MAX,
                'WAIT_TIME_READ_FILE': WAIT_TIME_READ_FILE
                # <<< INSERT YOUR OWN PARAMETERS HERE
                }

    # save parameters with simulation name before launching simulation
    with open(SAVE_DIR+sim_name_+'_START.json', 'w') as f:
        json.dump(params, f, separators=(',', ':'), sort_keys=True, indent=4)

    global trials
    trials = Trials()

    """ TODO: SOME EXEMPLE OF SEARCH SPACES - MAKE YOUR OWN BELOW >>>"""
    ### NOTE : pour ne pas faire bouger un paramètre il faut faire comme ceci :
    # 'shape': hp.quniform('shape', 0, 0.1, 1),
    search_space = {
                    # LOOK AT THE VARIETY OF KIND OF SPACE PARAMETERS YOU CAN HAVE (uniform, quniform, loguniform, choice, ...)
                    'sr': hp.quniform('sr', 1, 1.1, 1),
                    'iss':hp.loguniform('iss',np.log(10**-2), np.log(5*10**2)),
                    'N':hp.quniform('N', 100, 101, 100),
                    'leak':hp.loguniform('leak', np.log(10**-4), np.log(1.)), # equivalent to hp.uniform('leak', 4.5399929762484854e-05, 1.),
                    'thres_repl_unfrq_w':hp.quniform('thres_repl_unfrq_w', 0, 15,3),
                    'ridge': hp.loguniform('ridge', np.log(10**-11), np.log(1.1*10**1)),
                    'Wstd': hp.loguniform('Wstd', np.log(2*10**-2), np.log(2*10**1)),
                    ## FEEDBACK
                    'fb_func': hp.choice('fb_func', ['heavyside01', 'linear01', 'linear']),
                    'fb_scale': hp.loguniform('fb_scale', np.log(10**-4), np.log(10**1)),
                    'fb_proba': hp.quniform('fb_proba',  0.05, 1.0, 0.05)
                    }
    # search_space = {
    #                  'shape': hp.quniform('shape', 0, 127, 1), #Shape : [0..42] = Sinus / [43..84] = Scie / [85..126] = Carré / 127 = Noise
    #                  'freq-Mod-Env': hp.quniform('freq-Mod-Env', 0, 127, 1), #Freq Mod < Env
    #                  'octave': hp.quniform('octave', 0, 127, 1),
    #                  'VCA-decay': hp.quniform('VCA-decay', 0, 127, 1),
    #                  'VCA-release': hp.quniform('VCA-release', 0, 127, 1),
    #                  'VCF-attack': hp.quniform('VCF-attack', 0, 127, 1),
    #                  'VCF-decay': hp.quniform('VCF-decay', 0, 127, 1),
    #                  'transpose': hp.quniform('transpose', 0, 127, 1) # (-12st..+12st) (j'ai rajouté ça pour changer de note un peu !!!!)
    #                  # volume : non exploré, car réglé par l'utilisateur
    #                 }

    best = fmin(objective_dic,
            space=search_space,
            algo=partial(tpe.suggest, n_startup_jobs=HP_WARMUP_EVALS), #TPE with 'n_startup_jobs' initial random exploration
            # algo=partial(rand.suggest, n_startup_jobs=HP_WARMUP_EVAL), #RANDOM EXPLORATION
            max_evals=HP_MAX_EVALS,
            trials=trials)

    """ Show results with best """
    print("*** trials ***")
    pprint.pprint(trials.trials)
    print("*** trials[0]['result'] ***")
    pprint.pprint(trials.trials[0]['result'])
    print("space_eval"), space_eval(search_space, best)
    print("best"), best


    """ Save results with particular format for hp_analyse_error.py """
    all_loss = [t['result']['loss'] for t in trials.trials]
    all_result = [t['result'] for t in trials.trials]
    all_vals = [t['misc']['vals'] for t in trials.trials]
#    all_res = [t['misc']['vals'].update({'loss':t['result']['loss']}) for t in trials.trials]
    tup_result_vals = [(t['result'], t['misc']['vals']) for t in trials.trials] #for saving useful and seriazable data in a JSON file
    print("all_result")
    pprint.pprint(all_result)
    print("all_values")
    pprint.pprint(all_vals)
    print("max(all_loss)"), max(all_loss)
    print("min(all_loss)"), min(all_loss)

    # save results to JSON file
    json_dic = {'best': best,
                # 'general_param':general_param,
                'params':params,
                'tuples_result_vals': tup_result_vals}
#    with open('hyperopt_results_ter_iss075.json','w') as fout:
    with open(SAVE_DIR+'hyperopt_results_demo.json','w') as fout:
        # json.dump(json_dic, fout)
        json.dump(json_dic, fout, separators=(',', ':'), sort_keys=True, indent=4)
    # and test if results were saved correctly
    try:
#        with open('hyperopt_results_ter_iss075.json','r') as fin:
        with open(SAVE_DIR+'hyperopt_results_demo.json','r') as fin:
            data = json.load(fin)
            # pprint.pprint(byteify(data)) # in order to remove "u" in all keys of dictionary
        print("### Results were saved correctly to JSON file. ###")
    except Exception as e:
        # except:
        print(e)
        print("WARNING: Results were NOT saved correctly to JSON file.")


    save(trials, SAVE_DIR+"hyperopt_trials.pkl")
    save(best, SAVE_DIR+"hyperopt_best.pkl")

    test_open_saved_pickle(SAVE_DIR+"hyperopt_trials.pkl")
    test_open_saved_pickle(SAVE_DIR+"hyperopt_best.pkl")

    # save parameters with simulation name after launching simulation
    # with open(SAVE_DIR+args.filename+'_HP-SEARCH_'+sim_name+'_END.json', 'w') as f:
    with open(SAVE_DIR+sim_name+'_END.json', 'w') as f:
        # json.dump(params,f)
        json.dump(params, f, separators=(',', ':'), sort_keys=True, indent=4)


    #ring bell to say it's finished
    # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 1000))


if __name__=='__main__':
    """
    Examples of how to run the hyper-parameter search
    > python hyperapero.py -s vrai
    ou
    > python3 hyperapero.py -s vrai
    """
    # import json
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--filename", type=str, help="json file containing parameters for experiment (root directory is assumed to be params/)")
    parser.add_argument("-s", "--simulation_name", type=str, default="", help="simulation_name: this will be use for the file and saving directory)")
    # parser.add_argument("-l", "--lookup", action='store_true', help="lookup_search_space: Plot the search space given instead of launching a parameter search.")
    args = parser.parse_args()

    # Test to call dummy function to see if read/write files is fine
    # function_that_returns_the_loss_given_the_parameters(params={'shape':0, 'freq-Mod-Env':0})

    # On test si on arrive bien à accéder aux fichiers input/output d'échanges de paramètres ...
    print("\nOn check(e) si tout va bien ...\n")
    test_file_existance(sim_name=args.simulation_name)
    # ... puis on affiche un message pour dire qu'on est prêt
    input("... c'est bon ! On peut commencer l'hyper quest ! (appuie sur une touche s'il te plait !)\n")

    hp_search(sim_name=args.simulation_name)
