.. _getting_started:

===============
Getting started
===============


Python version support
======================

For now, full suppport is guaranteed for Python 3.8 and higher. Support is partial for Python 3.6 and 3.7
(see :ref:`distributed`).


Installation
============

.. panels::
    :card: + install-card
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Installing stable release (v0.3.0)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    ReservoirPy can be installed via pip from `PyPI <https://pypi.org/project/reservoirpy>`__.

    ++++++++++++++++++++++

    .. code-block:: bash

        pip install reservoirpy

    ---

    Installing previous stable release (v0.2.4)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    User guide of v0.2.4 can be found at :ref:`compat`.

        ++++++++++++++++++++++

        .. code-block:: bash

            pip install reservoirpy==0.2.4

    ---
    :column: col-12 p-3

    Complete installation guide
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A complete walkthrough for beginners and some instructions for developers.

    .. link-button:: ../developer_guide/advanced_install.html
        :type: url
        :text: Learn more
        :classes: btn-secondary stretched-link


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

You can now start using ReservoirPy ! Learn more about the software and its capabilities in the :ref:`user_guide`.
You can also find tutorials and examples in the `GitHub repository
<https://github.com/reservoirpy/reservoirpy/tree/master/tutorials>`_.
