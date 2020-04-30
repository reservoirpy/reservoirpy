import os
import gc
import time
import math
import json
import warnings
from os import path
from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

from reservoirpy import ESN, mat_gen
from reservoirpy import activationsfunc as F

import hyperopt as hopt


def _get_conf_from_json(confpath):
    if not(path.isfile(confpath)):
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
            raise ValueError(f"No {arg} argument found in config file.")
        
    if config.get("seed") is None:
        config["seed"] = np.random.randint(10000)
    
    if config["hp_method"] not in ["tpe", "random"]:
        raise ValueError(f"Unknow hyperopt algorithm: {config['hp_method']}. Available algs: random, tpe.")
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
    
    base_dir = base_path or '.'
    
    report_path = path.join(base_path, exp_name, 'results')

    if not(path.isdir(base_path)):
        os.mkdir(base_path)

    if not(path.isdir(path.join(base_path, exp_name))):
        os.mkdir(path.join(base_path, exp_name))
    
    if not(path.isdir(report_path)):
        os.mkdir(report_path)
        
    return report_path


def research(loss_function, dataset, config_path, report_path=None):
    
    config = _get_conf_from_json(config_path)
    report_path = _get_report_path(config["exp"], report_path)
    
    train_data, test_data = dataset

    def objective(kwargs):
        
        try:
            start = time.time()
            
            returned_dict = loss_ESN(train_data, test_data, config, **kwargs)
            
            end = time.time()
            duration = end - start
            
            returned_dict['status'] = hopt.STATUS_OK
            returned_dict['start_time'] = start
            returned_dict['duration'] = duration
            
            save_file = f"{returned_dict['loss']:.4f}_hyperopt_results_1call.json"
            
        except Exception as e:
            raise e
            start = time.time()
            
            returned_dict = {
                'status': hopt.STATUS_FAIL,
                'start_time': start,
                'error': str(e),
            }
            
            save_file = f"ERR{start}_hyperopt_results_1call.json"
            
        try:
            json_dict = {'returned_dict': returned_dict, 'current_params': kwargs}
            with open(path.join(report_path, save_file), "w+") as f:
                json.dump(json_dict, f)
        except Exception as e:
            warnings.warn("Results of current simulation were NOT saved correctly to JSON file.")
            warnings.warn(str(e))
            
        return returned_dict

    search_space = config["hp_space"]

    trials = hopt.Trials()

    best = hopt.fmin(objective,
            space=search_space,
            algo=config["hp_method"],
            max_evals=config['hp_max_evals'],
            trials=trials)
    
    return best



def logscale_plot(ax, xrange, yrange, base=10):
    if xrange is not None:
        ax.set_xscale("log", basex=base)
        ax.set_xlim([np.min(xrange), np.max(xrange)])
        locmin = ax.xaxis.get_major_locator()
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if yrange is not None:
        ax.set_yscale("log", basey=base)
        ax.set_ylim([yrange.min() - 0.1*yrange.min(), yrange.max() + 0.1*yrange.min()])
        locmin = ax.yaxis.get_major_locator()
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    

def scale(x):
    return (x - x.min()) / (x.ptp())


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def get_results(result_dir):
    report_path = path.join(result_dir, "results")
    results = []
    for file in os.listdir(report_path):
        if path.isfile(path.join(report_path, file)) and "ERR" not in file:
            with open(path.join(report_path, file), 'r') as f:
                results.append(json.load(f))
    return results


def plot_opt_results(result_dir, params, score="loss", not_log=None, fig_title=None):
        
    N = len(params)
    title = fig_title or ""
    
    results = get_results(result_dir) 
    values = {p: np.array([r['current_params'][p] for r in results]) for p in params}
    
    l = np.array([r['returned_dict']['loss'] for r in results])
    s = np.array([r['returned_dict'][score] for r in results])
    
    lmin = l > l.min()
    cl = np.log10(l[lmin])

    percent = math.ceil(len(s) * 0.05)
    max001 = s.argsort()[-percent:][::-1]
    c_max001 = scale(s[max001])

    ## gridspecs
    fig = plt.figure(figsize=(20, 25), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1/30, 29/30])
    fig.suptitle(f"Hyperopt trials summary - {title}", size=15)

    gs0 = gs[0].subgridspec(1, 3)
    gs1 = gs[1].subgridspec(N + 1, N)

    ## axes 
    axes = []
    for a in range(N + 1):
        line = []
        for b in range(N):
            line.append(fig.add_subplot(gs1[a, b]))
        axes.append(line)
        
    axes = np.array(axes)


    lbar_ax = fig.add_subplot(gs0[0, 0])
    fbar_ax = fig.add_subplot(gs0[0, 1])
    rad_ax = fig.add_subplot(gs0[0, 2])
    rad_ax.axis('off')

    ## plot
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if p1 == p2:
                xrange = values[p2].copy()
                if not_log and p2 in not_log:
                    logscale_plot(axes[i, j], None, l)
                else:
                    logscale_plot(axes[i, j], xrange, l)
                
                axes[i, j].scatter(xrange[lmin], l[lmin], s[lmin]*100, color="orange")
                axes[i, j].scatter(xrange[~lmin], l.min(), s[~lmin]*100, color="red")
                axes[i, j].scatter(xrange[max001], l[max001], s[max001]*100, c=c_max001, cmap='YlGn')
                axes[i, j].set_xlabel(p2)
                axes[i, j].set_ylabel("loss")
            else:
                xrange = values[p2].copy()
                yrange = values[p1].copy()
                if not_log and p2 in not_log:
                    logscale_plot(axes[i, j], None, yrange)
                elif not_log and p1 in not_log:
                    logscale_plot(axes[i, j], xrange, None)
                else:
                    logscale_plot(axes[i, j], xrange, yrange)
                sc = axes[i, j].scatter(xrange[lmin], yrange[lmin], s[lmin]*100, c=l[lmin], cmap="inferno")
                fmax = axes[i, j].scatter(xrange[max001], yrange[max001], s[max001]*100, c=c_max001, cmap="YlGn")
                axes[i, j].scatter(xrange[~(lmin)], yrange[~(lmin)], s[~(lmin)]*100, color="red")
                
                axes[i, j].set_xlabel(p2)
                axes[i, j].set_ylabel(p1)

    # legends

    handles, labels = sc.legend_elements(prop="sizes")
    legend = rad_ax.legend(handles, labels, loc="center left", title=f"% {score}", mode='expand', ncol=len(labels))

    l_cbar = fig.colorbar(sc, cax=lbar_ax, ax=axes.ravel().tolist(), orientation="horizontal")
    l_cbar.ax.set_title("Loss value")

    f_cbar = fig.colorbar(fmax, cax=fbar_ax, ax=axes.ravel().tolist(), orientation="horizontal", ticks=[0, 0.5, 1])
    f_cbar.ax.set_title(f"{score} best population")
    f_cbar.ax.set_xticklabels(["5% best", "2.5% best", "Best"])

    # violinplots

    for i, p in enumerate(params):
        y = values[p].copy()[max001]
        violin = axes[-1, i].violinplot(y, showmeans=True, showextrema=False)
        
        for pc in violin['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('white')

        quartile1, medians, quartile3 = np.percentile(y, [25, 50, 75])
        whiskers = np.array([
            adjacent_values(y, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        axes[-1, i].scatter(1, medians, marker='o', color='white', s=30, zorder=3)
        axes[-1, i].vlines(1, quartile1, quartile3, color='orange', linestyle='-', lw=8)
        axes[-1, i].vlines(1, whiskersMin, whiskersMax, color='white', linestyle='-', lw=2)
        
        axes[-1, i].scatter(1, values[p][l.argmin()], color='red', zorder=4)
        axes[-1, i].scatter(1, values[p][s.argmax()], color="green", zorder=5)
        axes[-1, i].set_xlabel(f"5% best {score} parameter distribution")
        axes[-1, i].set_ylabel(p)

    return fig
    
    
def logscale_plot(ax, xrange, yrange, base=10):
    if xrange is not None:
        ax.set_xscale("log", basex=base)
        ax.set_xlim([np.min(xrange), np.max(xrange)])
        locmin = ax.xaxis.get_major_locator()
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if yrange is not None:
        ax.set_yscale("log", basey=base)
        ax.set_ylim([yrange.min() - 0.1*yrange.min(), yrange.max() + 0.1*yrange.min()])
        locmin = ax.yaxis.get_major_locator()
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    

def scale(x):
    return (x - x.min()) / (x.ptp())


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


if __name__ == "__main__":
    
    def loss_ESN(train_data, test_data, config, *, iss, N, sr, leak, ridge):
    
        # unpack train and test data, with target values.
        x_train, y_train = train_data
        x_test, y_test = test_data

        x_train, y_train = x_train.reshape(-1, 1), y_train.reshape(-1, 1)
        x_test, y_test = x_test.reshape(-1, 1), y_test.reshape(-1, 1)

        nb_features = x_train.shape[1]
        seed = config['seed']
        
        # builds an ESN given the input parameters
        W = mat_gen.generate_internal_weights(N=N, spectral_radius=sr,
                                            seed=seed)

        Win = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=nb_features, 
                                            input_bias=True, input_scaling=iss,
                                            seed=seed)


        reservoir = ESN(lr=leak, W=W, Win=Win, input_bias=True, ridge=ridge)


        # train and test the model
        reservoir.train(inputs=[x_train], teachers=[y_train], 
                        wash_nr_time_step=20, verbose=False)

        outputs, _ = reservoir.run(inputs=[x_test], verbose=False)

        # returns a dictionnary of metrics. The 'loss' key is mandatory when
        # using Hyperopt.
        return {'loss': metrics.mean_squared_error(outputs[0], y_test)}
    
    # 10,000 timesteps of Mackey-Glass timeseries
    mackey_glass = np.loadtxt('examples/MackeyGlass_t17.txt')
    
    # split data
    train_frac = 0.6
    train_start, train_end = 0, int(train_frac * len(mackey_glass))
    test_start, test_end = train_end, len(mackey_glass) - 2
    
    # pack it
    train_data = (mackey_glass[train_start:train_end], mackey_glass[train_start+1:train_end+1])
    test_data = (mackey_glass[test_start:test_end], mackey_glass[test_start+1:test_end+1])
    
    dataset = (train_data, test_data)
    
    # run the random search
    best = research(loss_ESN, dataset, "examples/mackeyglass-config.json", "examples/report")
    
    # fig = plot_opt_results("examples/report/hpt-mackeyglass", params=["sr", "leak", "iss", "ridge"])
    
    # fig.savefig("test.png")