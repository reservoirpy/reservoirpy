.. ReservoirPy documentation master file, created by sphinx-quickstart on Fri Jan 29 09:46:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: reservoirpy

===========
ReservoirPy
===========

**A simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**

.. image:: https://badge.fury.io/py/reservoirpy.svg
    :target: https://badge.fury.io/py/reservoirpy.svg
.. image:: https://img.shields.io/pypi/pyversions/reservoirpy
    :target: https://img.shields.io/pypi/pyversions/reservoirpy
.. image:: https://github.com/reservoirpy/reservoirpy/actions/workflows/test.yml/badge.svg?branch=master
    :target: https://github.com/reservoirpy/reservoirpy/actions/workflows/test.yml/badge.svg?branch=master
.. image:: https://codecov.io/gh/reservoirpy/reservoirpy/branch/master/graph/badge.svg?token=JC8R1PB5EO
    :target: https://codecov.io/gh/reservoirpy/reservoirpy/branch/master/graph/badge.svg?token=JC8R1PB5EO


:mod:`reservoirpy` is a simple user-friendly library based on Python scientific modules.
It provides a flexible interface to implement efficient Reservoir Computing (RC)
architectures with a particular focus on Echo State Networks (ESN).

.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started
    user_guide/index
    API reference <api/index>
    developer_guide/index


.. grid:: 1 2 2 2

    .. grid-item-card::
        :class-card: intro-card
        :shadow: md
        :img-top: _static/getting_started.svg
        :class-img-top: dark-light

        Getting started
        ^^^^^^^^^^^^^^^

        A quick introduction to ReservoirPy basic concepts, from installation
        to your first Reservoir Computing models.

        +++

        .. button-ref:: get_started
            :color: secondary
            :click-parent:
            :expand:

            To the getting started guide

    .. grid-item-card::
        :class-card: intro-card
        :shadow: md
        :img-top: _static/user_guide.svg
        :class-img-top: dark-light

        User guide
        ^^^^^^^^^^

        A complete guide to the ReservoirPy project, exploring key concepts through
        documentation, tutorials and examples.

        +++

        .. button-ref:: user_guide
            :color: secondary
            :click-parent:
            :expand:

            To the user guide

    .. grid-item-card::
        :class-card: intro-card
        :shadow: md
        :img-top: _static/api.svg
        :class-img-top: dark-light

        API reference
        ^^^^^^^^^^^^^

        The ReservoirPy API documentation, with detailed descriptions of all
        its components.

        +++

        .. button-ref:: api
            :color: secondary
            :click-parent:
            :expand:

            To the reference guide

    .. grid-item-card::
        :class-card: intro-card
        :shadow: md
        :img-top: _static/dev_guide.svg
        :class-img-top: dark-light

        Developer guide
        ^^^^^^^^^^^^^^^

        A guide to help us make ReservoirPy a better project, from correcting typos to
        creating new tools within the API.

        +++

        .. button-ref:: developer_guide
            :color: secondary
            :click-parent:
            :expand:

            To the development guide


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
parallel implementation, sparse array computation, fast spectral initialization, etc.
Moreover, graphical tools are included to easily explore hyperparameters with the help
of the `hyperopt` library.
