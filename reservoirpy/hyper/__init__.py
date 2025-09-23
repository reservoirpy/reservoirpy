"""
==============================================================
Hyperparameter optimization helpers (:mod:`reservoirpy.hyper`)
==============================================================

Utility module for optimizing hyperparameters, with support for both
*hyperopt*-based search and CPU-parallelized custom search using *joblib*.

Note
----

This module is meant to be used alongside *hyperopt*
and *Matplotlib* packages, which are not installed
with ReservoirPy by default. Before using the
:py:mod:`reservoirpy.hyper` module, consider installing
these packages if they are not already installed.

The function `parallelized_research` of this module
do not rely on *hyperopt* and can be used independently.


.. autosummary::
    :toctree: generated/

    research
    parallelized_research
    plot_hyperopt_report
"""
from ._hyperplot import plot_hyperopt_report
from ._hypersearch import research
from ._parallelized_hypersearch import parallel_research

__all__ = ["research", "parallel_research", "plot_hyperopt_report"]