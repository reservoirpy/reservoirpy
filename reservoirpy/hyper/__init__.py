"""
==============================================================
Hyperparameter optimization helpers (:mod:`reservoirpy.hyper`)
==============================================================

Utility module for optimization of hyperparameters
using *hyperopt*.

Note
----

This module is meant to be used alongside *hyperopt*
and *Matplotlib* packages, which are not installed
with ReservoirPy by default. Before using the
:py:mod:`reservoirpy.hyper` module, consider installing
these packages if they are not already installed.


.. autosummary::
    :toctree: generated/

    research
    plot_hyperopt_report
"""
from ._hypersearch import research
from ._hyperplot import plot_hyperopt_report

__all__ = ["research", "plot_hyperopt_report"]
