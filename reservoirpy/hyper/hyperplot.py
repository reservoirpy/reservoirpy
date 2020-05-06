import os
import math
import json
import time

from os import path

import numpy as np


HP_REPORTS = os.path.join("examples", "report")


def get_results(exp):
    report_path = path.join(HP_REPORTS, exp, "results")
    results = []
    for file in os.listdir(report_path):
        if path.isfile(path.join(report_path, file)):
            with open(path.join(report_path, file), 'r') as f:
                results.append(json.load(f))
    return results


def outliers_idx(values, max_deviation):
    mean = values.mean()
    std = values.std()
    
    dist = abs(values - mean)
    corrected = dist < mean + max_deviation
    return corrected
    

def logscale_plot(ax, xrange, yrange, base=10):
    if xrange is not None:
        ax.xaxis.set_minor_formatter(ticker.LogFormatter())
        ax.xaxis.set_major_formatter(ticker.LogFormatter())
        ax.set_xscale("log", basex=base)
        ax.set_xlim([np.min(xrange), np.max(xrange)])
    if yrange is not None:
        ax.yaxis.set_minor_formatter(ticker.LogFormatter())
        ax.yaxis.set_major_formatter(ticker.LogFormatter())
        ax.set_yscale("log", basey=base)
        ax.set_ylim([yrange.min() - 0.1*yrange.min(), yrange.max() + 0.1*yrange.min()])


def scale(x):
    return (x - x.min()) / (x.ptp())


def cross_parameter_plot(ax, values, scores, loss, smaxs, cmaxs, lmaxs, p1, p2, log1, log2):
    
    X = values[p2].copy()
    Y = values[p1].copy()
    
    if log1:
        logscale_plot(ax, X, None)
    if log2:
        logscale_plot(ax, None, Y)
    
    ax.tick_params(axis='both', which='both')
    ax.tick_params(axis='both', labelsize="xx-small")
    ax.grid(True, which="both", ls="-", alpha=0.75)
    
    ax.set_xlabel(p2)
    ax.set_ylabel(p1)
    
    sc_l = ax.scatter(X[lmaxs], Y[lmaxs], scores[lmaxs]*100, c=loss[lmaxs], cmap="inferno")
    sc_s = ax.scatter(X[smaxs], Y[smaxs], scores[smaxs]*100, c=cmaxs, cmap="YlGn")
    sc_m = ax.scatter(X[~(lmaxs)], Y[~(lmaxs)], scores[~(lmaxs)]*100, color="red")
    
    return sc_l, sc_s, sc_m


def loss_plot(ax, values, scores, loss, smaxs, cmaxs, lmaxs, p, log, legend):
    
    X = values[p].copy()
    
    if log:
        logscale_plot(ax, X, loss)
    else:
        logscale_plot(ax, None, loss)
    
    ax.set_xlabel(p)
    ax.set_ylabel("loss")
    
    ax.tick_params(axis='both', which='both')
    ax.tick_params(axis='both', labelsize="xx-small")
    ax.grid(True, which="both", ls="-", alpha=0.75)
    
    sc_l = ax.scatter(X[lmaxs], loss[lmaxs], scores[lmaxs]*100, color="orange")
    sc_s = ax.scatter(X[smaxs], loss[smaxs], scores[smaxs]*100, c=cmaxs, cmap="YlGn")
    sc_m = ax.scatter(X[~(lmaxs)], loss.min(), scores[~(lmaxs)]*100, color="red", label="Loss min.")

    if legend:
        ax.legend()
    
    return sc_l, sc_s, sc_m

def parameter_violin(ax, values, scores, loss, smaxs, cmaxs, p, log, legend):
    
    import matplotlib.pyplot as plt
    
    y = np.log10(values[p].copy()[smaxs])
    all_y = np.log10(values[p].copy())
    
    # if log:
    #     logscale_plot(ax, y, None)

    ax.get_yaxis().set_ticks([])
    ax.tick_params(axis='x', which='both')
    
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    
    def format_func(value, tick_number):
        return "$10^{"+str(int(np.floor(value)))+"}$"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.set_xlabel(p)
    ax.grid(True, which="both", ls="-", alpha=0.75)
    #ax.set_xscale('log', basex=10)
    ax.set_xlim([np.min(all_y), np.max(all_y)])
    
    
    violin = ax.violinplot(y, vert=False, showmeans=False, showextrema=False)
    
    for pc in violin['bodies']:
        pc.set_facecolor('forestgreen')
        pc.set_edgecolor('white')

    quartile1, medians, quartile3 = np.percentile(y, [25, 50, 75])

    ax.scatter(medians, 1, marker='o', color='orange', s=30, zorder=4, label="Median")
    ax.hlines(1, quartile1, quartile3, color='gray', linestyle='-', lw=4, label="Q25/Q75")
    ax.vlines(y.mean(), 0.5, 1.5, color='blue', label="Mean")
    
    ax.scatter(np.log10(values[p][scores.argmax()]), 1, color="red", zorder=5, label="Best score")
    ax.scatter(y, np.ones_like(y), c=cmaxs, cmap="YlGn", alpha=0.5, zorder=3)
    
    if legend:
        ax.legend(loc=2)


def plot_hyperopt_report(results, test_params, metric="rmse", not_log=None, title=None):
            
    import matplotlib.pyplot as plt
    import seaborn as sns
            
    sns.set(context="paper", style="darkgrid", font_scale=1.5)
    N = len(test_params)
    not_log = not_log or []

    loss = np.array([r['returned_dict']['loss'] for r in results])
    scores = np.sqrt(np.array([r['returned_dict']['loss'] for r in results]))
    
    not_outliers = outliers_idx(loss, 10)
    loss = loss[not_outliers]
    scores = scores[not_outliers]
    
    if scores.max() > 1.0:
        scores = 1 - scale(scores)
        
    values = {p: np.array([r['current_params'][p] for r in results])[not_outliers] for p in test_params}
        
    ## loss and f1 values

    lmaxs = loss > loss.min()
    c_l = np.log10(loss[lmaxs])

    percent = math.ceil(len(scores) * 0.05)
    smaxs = scores.argsort()[-percent:][::-1]
    cmaxs = scale(scores[smaxs])    
    
    ## gridspecs

    fig = plt.figure(figsize=(15, 19), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2/30, 28/30])
    fig.suptitle(f"Hyperopt trials summary - {title}", size=15)

    gs0 = gs[0].subgridspec(1, 3)
    gs1 = gs[1].subgridspec(N + 1, N)


    lbar_ax = fig.add_subplot(gs0[0, 0])
    fbar_ax = fig.add_subplot(gs0[0, 1])
    rad_ax = fig.add_subplot(gs0[0, 2])
    rad_ax.axis('off')

    # plot
    axes = []
    for i, p1 in enumerate(test_params):
        for j, p2 in enumerate(test_params):
            ax = fig.add_subplot(gs1[i, j])
            axes.append(ax)
            if p1 == p2:
                sc_l, sc_s, sc_m = loss_plot(ax, values, scores, loss, smaxs, 
                                             cmaxs, lmaxs, p2, not(p2 in not_log),
                                             (i==0 and j==0))
            else:
                sc_l, sc_s, sc_m = cross_parameter_plot(ax, values, scores, loss, 
                                     smaxs, cmaxs, lmaxs, p1, p2, 
                                     not(p1 in not_log), not(p2 in not_log))

    #legends

    handles, labels = sc_l.legend_elements(prop="sizes")
    legend = rad_ax.legend(handles, labels, loc="center left", 
                           title=f"Normalized {metric} (%)", mode='expand', 
                           ncol=len(labels) // 2 + 1)

    l_cbar = fig.colorbar(sc_l, cax=lbar_ax, ax=axes, orientation="horizontal")
    _ = l_cbar.ax.set_title("Loss value")

    f_cbar = fig.colorbar(sc_s, cax=fbar_ax, ax=axes, 
                          orientation="horizontal", ticks=[0, 0.5, 1])
    _ = f_cbar.ax.set_title(f"{metric} best population")
    _ = f_cbar.ax.set_xticklabels(["5% best", "2.5% best", "Best"])

    # violinplots

    for i, p in enumerate(test_params):
        ax = fig.add_subplot(gs1[-1, i])
        legend = True if i == 0 else False
        parameter_violin(ax, values, scores, 
                         loss, smaxs, cmaxs, p, not(p in not_log), legend)
        if legend:
            ax.set_ylabel(f"5% best {metric}\nparameter distribution")
    return fig
    