.. _get_started:

===============
Getting started
===============


Python version support
======================

For now, full support is guaranteed for Python 3.9 and higher.
(see :ref:`advanced_install`).


Installation
============

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card::
        :class-card: install-card
        :shadow: md

        Installing stable release (v0.4.0)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        ReservoirPy can be installed via pip from `PyPI <https://pypi.org/project/reservoirpy>`__.

        ++++++++++++++++++++++

        .. code-block:: bash

            pip install reservoirpy

    .. grid-item-card::
        :class-card: install-card
        :shadow: md

        Complete installation guide
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^

        A complete walkthrough for beginners and some instructions for developers.

        .. button-ref:: ../developer_guide/advanced_install
            :color: secondary
            :click-parent:

            Learn more


Project philosophy
==================

Most of machine learning work these days is becoming more and more complicated.
To deal with this increasing complexity, a lot of open source projects have
blossomed, and allow one to go from very simple models to over-complicated
architectures.

This is not the philosophy of reservoir computing, and thus, not the philosophy
of ReservoirPy.

Because Reservoir Computing aims at making complexity emerge from apparent simplicity,
ReservoirPy provides its users with very simple tools that can achieve a wide range
of machine learning tasks, in particular when it comes to deal with sequential data.

These tools are all based on `NumPy <https://numpy.org/>`_ and `SciPy <https://www.scipy.org/>`_,
the Python *basic* scientific libraries, and can therefore be mastered by any Python enthusiast,
from beginners to experts.


Learn more
==========

You can now start using ReservoirPy! Learn more about the software and its capabilities in the :ref:`user_guide`.
You can also find tutorials and examples in the `GitHub repository
<https://github.com/reservoirpy/reservoirpy/tree/master/tutorials>`_.
