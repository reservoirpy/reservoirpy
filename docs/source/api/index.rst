.. _api:

=========================
ReservoirPy API reference
=========================

.. autosummary::

    reservoirpy.ESN - Simple Echo State Network.

.. grid:: 1 3 3 2

    .. grid-item-card:: ``nodes``
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/reservoir_node.svg
        :class-img-top: dark-light
        :link: nodes
        :link-type: ref

        Atomic components used to create models.

    .. grid-item-card:: ``datasets``
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/lorenz_dataset.png
        :class-img-top: dark-light
        :link: datasets
        :link-type: ref

        Loadable datasets and helper functions

    .. grid-item-card:: ``observables``
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/mse_observable.svg
        :class-img-top: dark-light
        :link: observables
        :link-type: ref

        Metrics for timeseries and reservoirs

    .. grid-item-card:: ``hyper``
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/plot_hyper.svg
        :class-img-top: dark-light
        :link: hyper
        :link-type: ref

        Explore and optimize your hyperparameters

    .. grid-item-card:: ``mat_gen``
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/normal_matgen.png
        :class-img-top: dark-light
        :link: mat_gen
        :link-type: ref

        Matrix generators for reservoir's weights

    .. grid-item-card:: ``activationsfunc``
        :columns: 12 6 6 4
        :class-card: api-card
        :shadow: md
        :img-top: ../_static/api_icons/tanh_activationsfunc.svg
        :class-img-top: dark-light
        :link: activationsfunc
        :link-type: ref

        Common activation functions

All modules
-------------

.. toctree::
   :maxdepth: 1

   reservoirpy.nodes
   reservoirpy.mat_gen
   reservoirpy.activationsfunc
   reservoirpy.observables
   reservoirpy.datasets
   reservoirpy.hyper
   reservoirpy.node
   reservoirpy.model
   reservoirpy.ops
   reservoirpy.ESN
