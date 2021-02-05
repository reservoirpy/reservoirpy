.. ReservoirPy documentation master file, created by
   sphinx-quickstart on Fri Jan 29 09:46:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========
ReservoirPy
===========

**A simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**

.. image:: https://badge.fury.io/py/reservoirpy.svg
    :target: https://badge.fury.io/py/reservoirpy
.. image:: https://travis-ci.org/reservoirpy/reservoirpy.svg?branch=master
    :target: https://travis-ci.org/reservoirpy/reservoirpy

ReservoirPy is a simple user-friendly library based on Python scientific modules.
It provides a flexible interface to implement efficient Reservoir Computing (RC)
architectures with a particular focus on Echo State Networks (ESN). Advanced features of
ReservoirPy allow to improve computation time efficiency on a simple laptop compared to
basic Python implementation. Some of its features are: offline and online training,
parallel implementation, sparse matrix computation, fast spectral initialization, etc.
Moreover, graphical tools are included to easily explore hyperparameters with the help
of the `hyperopt` library.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   whats_rc
   getting_started
   using_reservoirpy
   api/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Cite
====

   Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut.
   ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks.
   2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026
