.. _api:

=========================
ReservoirPy API reference
=========================


.. grid:: 1 3 3 2

    .. grid-item-card:: ReservoirPy Nodes
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/reservoir_node.svg
        :class-img-top: dark-light
        :link: nodes
        :link-type: ref

        Atomic components used to create models.

    .. grid-item-card:: Datasets
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/lorenz_dataset.png
        :class-img-top: dark-light
        :link: datasets
        :link-type: ref

        Useful timeseries and helper functions

    .. grid-item-card:: Metrics and observables
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/mse_observable.svg
        :class-img-top: dark-light
        :link: observables
        :link-type: ref

        Metrics on timeseries

    .. grid-item-card:: Hyperparameter search
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/plot_hyper.svg
        :class-img-top: dark-light
        :link: hyper
        :link-type: ref

        Explore your hyperparameters

    .. grid-item-card:: Weights initialization
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/normal_matgen.png
        :class-img-top: dark-light
        :link: mat_gen
        :link-type: ref

        Matrix generators for reservoir's weights initialization

    .. grid-item-card:: Activation functions
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/relu_activationsfunc.svg
        :class-img-top: dark-light
        :link: activationsfunc
        :link-type: ref

        Basic activation functions

    .. grid-item-card:: Experimental features
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/getting_started.svg
        :class-img-top: dark-light
        :link: experimental
        :link-type: ref

        Features that aims to be part of ReservoirPy but that are still under
        development, and may break semantic versioning.

Other modules
-------------

.. toctree::
   :maxdepth: 1

   reservoirpy.nodes
   reservoirpy.mat_gen
   reservoirpy.activationsfunc
   reservoirpy.observables
   reservoirpy.datasets
   reservoirpy.hyper
   reservoirpy.experimental
   reservoirpy.node
   reservoirpy.model
   reservoirpy.compat
   reservoirpy.ops
