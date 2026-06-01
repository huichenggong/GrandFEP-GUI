# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the installed package importable when building without installing.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'grandfep_gui'
copyright = '2026, Chenggong Hui'
author = 'Chenggong Hui'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # NumPy / Google-style docstrings
    "sphinx.ext.viewcode",   # [source] links in the HTML
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'page_width'       : '90%',
    'body_max_width'   : 'auto',
    'fixed_sidebar'    : True,
    'github_user'      : 'huichenggong',
    'github_repo'      : 'GrandFEP',
    'github_banner'    : True

}