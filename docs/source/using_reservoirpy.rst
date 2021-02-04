.. _`rc with reservoirpy`:

====================================
Reservoir Computing with ReservoirPy
====================================


Untangle temporal complexity
============================

The core component of any flavor of reservoir computing is the *reservoir*, a type of recurrent artificial
neural network. To learn in details how reservoirs work, you can visit the page :ref:`whats rc`.
Long story short, reservoirs are neural networks typically sparsely connected, i.e. only a few (generally around
10% or 20%) of all the possible connections between the neuronal units are used. These networks are usually quite
large (a thousand units or more), and all their connections are initialized randomly, **and kept constant during
training**. Initialization of connections weights is therefore a very important step. Two major aspects of this
initialization are covered by ReservoirPy:

* the **spectral radius**, a parameter controlling the choatic dynamics emerging in the neuronal activities of the reservoir,
* the **probability of connection**, or *sparsity* of the reservoir, which controls how sparsely connected the neuronal units are inside the reservoir.

.. note::

    🚧 This is  unfinished 🚧 Come back later for the full version ! 🚧