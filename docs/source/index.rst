.. ReservoirPy documentation master file, created by sphinx-quickstart on Fri Jan 29 09:46:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: reservoirpy

===========
ReservoirPy
===========

**A simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**

.. image:: https://badge.fury.io/py/reservoirpy.svg
    :target: https://badge.fury.io/py/reservoirpy
.. image:: https://travis-ci.org/reservoirpy/reservoirpy.svg?branch=master
    :target: https://travis-ci.org/reservoirpy/reservoirpy

:mod:`reservoirpy` is a simple user-friendly library based on Python scientific modules.
It provides a flexible interface to implement efficient Reservoir Computing (RC)
architectures with a particular focus on Echo State Networks (ESN).

.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started
    user_guide/index
    api/index
    developer_guide/index


.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/getting_started.svg

    Getting started
    ^^^^^^^^^^^^^^^

    A quick introduction to ReservoirPy basic concepts, from installation
    to your first Reservoir Computing models.

    +++

    .. link-button:: getting_started
            :type: ref
            :text: To the getting started guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/user_guide.svg

    User guide
    ^^^^^^^^^^

    A complete guide to the ReservoirPy project, exploring key concepts through
    documentation, tutorials and examples.

    +++

    .. link-button:: user_guide
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/api.svg

    API reference
    ^^^^^^^^^^^^^

    The ReservoirPy API documentation, with detailed descriptions of all
    its components.

    +++

    .. link-button:: api
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/dev_guide.svg

    Developer guide
    ^^^^^^^^^^^^^^^

    A guide to help us make ReservoirPy a better project, from correcting typos to
    creating new tools within the API.

    +++

    .. link-button:: development
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link


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


Advanced features of ReservoirPy allow to improve computation time efficiency on a simple laptop compared to
basic Python implementation. Some of its features are: offline and online training,
parallel implementation, sparse matrix computation, fast spectral initialization, etc.
Moreover, graphical tools are included to easily explore hyperparameters with the help
of the `hyperopt` library.
