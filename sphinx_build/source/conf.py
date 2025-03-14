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

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../ArchPy'))
sys.path.insert(0, os.path.abspath('../../examples'))

# sys.path.append("C:/Users/schorppl/switchdrive/Thèse/prog/sphinx/ArchPy")
from ArchPy import __version__
import base

# -- Project information -----------------------------------------------------

project = 'ArchPy'
copyright = '2025, University of Neuchâtel'
author = 'LS'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
			'sphinx.ext.autosectionlabel',
			'sphinx.ext.napoleon',
			'sphinx.ext.autosummary',
			'sphinx.ext.autodoc',
			'sphinx.ext.duration',
			'myst_parser',
            'sphinxemoji.sphinxemoji',
            "nbsphinx",
            'sphinx_gallery.load_style',
]

source_suffix = [".rst", ".md"]

sphinxemoji_style = 'twemoji'
autosectionlabel_prefix_document = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# 
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'piccolo_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

# def setup(app):
#     app.add_stylesheet('theme_overrides.css')