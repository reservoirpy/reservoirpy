# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from cycler import cycler

sys.path.insert(0, os.path.abspath('..'))

import reservoirpy
from reservoirpy import __version__

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# The suffix of source filenames.
source_suffix = [".rst"]

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# -- Project information -----------------------------------------------------

project = 'ReservoirPy'
copyright = '2021, Xavier Hinaut, Nathan Trouvain'
author = 'Xavier Hinaut, Nathan Trouvain'

# The full version, including alpha/beta/rc tags
release = str(__version__)

language = "en"

pygments_style = "sphinx"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'sphinx.ext.autosummary',
    'sphinx_panels',
    'sphinx.ext.intersphinx',
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    ]

# Intersphinx links
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None)
    }

# matplotlib plot directive
plot_include_source = False
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_pre_code = """import numpy as np;import matplotlib.pyplot as plt;from reservoirpy import set_seed;set_seed(42)"""
plot_rcparams = {'axes.prop_cycle': cycler(color=
    ['#F54309', '#78A6F5', '#FFC240', '#00D1C7', '#5918C2',
     '#A4E3FA', '#F5250A', '#3AFA98', '#923ADB', '#D1B971',])}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = "_static/rpy_navbar.png"

html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/reservoirpy/reservoirpy",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_css_files = [
    "css/reservoirpy.css",
]

header = f"""\
.. currentmodule:: reservoirpy
.. ipython:: python
   :suppress:
   
   import numpy as np
   import matplotlib.pyplot as plt
   from reservoirpy import set_seed
   from reservoirpy import verbosity
   verbosity(0)
   set_seed(42)
   np.set_printoptions(precision=4, suppress=True)
   import os
   os.chdir(r'{os.path.dirname(os.path.dirname(__file__))}')
"""

html_context = {
    "header": header
    }

# If false, no module index is generated.
html_use_modindex = True


ipython_warning_is_error = False
ipython_execlines = [
    "import numpy as np",
]

numfig = True


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_default_options = {
    'inherited-members': None,
}
autodoc_typehints = 'none'


# from pandas conf.py (https://github.com/pandas-dev/pandas/blob/master/doc/source/conf.py)
def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # https://www.ericholscher.com/blog/2016/jul/25/integrating-jinja-rst-sphinx/
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered

import re
import inspect
from os.path import relpath, dirname

# from scipy conf.py (https://github.com/scipy/scipy/blob/3da3fb3de8beffc79797b7b62ea3c98cc8075d2e/doc/source/conf.py)
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(reservoirpy.__file__), '..'))
    fn = relpath(fn, start=startdir).replace(os.path.sep, '/')

    if fn.startswith('reservoirpy/'):
        return "https://github.com/reservoirpy/reservoirpy/blob/master/%s%s" % (
            fn, linespec)
    else:
        return None


def setup(app):
    app.connect("source-read", rstjinja)
