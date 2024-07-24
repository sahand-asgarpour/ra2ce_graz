#!/usr/bin/env python
#
# ra2ce documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import shutil
import sys
from distutils.dir_util import copy_tree
from pathlib import Path

import sphinx_autosummary_accessors

# This is not needed
sys.path.insert(0, os.path.abspath(".."))
import ra2ce

print("ra2ce", ra2ce)
print("dir", dir(ra2ce))


# -- Helper functions ---------------------------------
def remove_dir_content(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    if os.path.isdir(path):
        shutil.rmtree(path)


# NOTE: the examples/ folder in the root should be copied to docs/_examples after running sphinx
# # -- Copy notebooks to include in docs -------
if os.path.isdir("build"):
    remove_dir_content("build")
if os.path.isdir("_examples"):
    remove_dir_content("_examples")

os.makedirs("_examples")
copy_tree("../examples", "_examples")

# Exclude some of the examples content:
_files_to_include = ["summary_"]


def remove_extra_files_from_dir(dir_path: Path):
    assert dir_path.exists(), "Examples dir was not correctly copied!"
    for _file in dir_path.rglob("*"):
        if _file.suffix.lower() in [".md", ".ipynb"]:
            if not any(_fi in _file.stem for _fi in _files_to_include):
                _file.unlink()


_examples_dir = Path("_examples")
_examples_dir.joinpath("README.md").unlink()
remove_extra_files_from_dir(_examples_dir.joinpath("hackathons"))


if os.path.isdir("docs"):
    remove_dir_content("docs")

_src_diagrams = "../docs/_diagrams/"
_dst_diagrams = "docs/_diagrams/"
os.makedirs(_dst_diagrams)
for _img_file in os.listdir(_src_diagrams):
    if not _img_file.endswith(".png"):
        continue
    shutil.copy((_src_diagrams + _img_file), (_dst_diagrams + _img_file))

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx_design",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autosummary_accessors",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Risk Assessment and Adaptation for Critical infrastructurE"
copyright = "2024, Deltares"
author = "Margreet van Marle\\Frederique de Groen\\Lieke Meijer\\Sahand Asgarpour\\Carles Soriano Perez"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = ra2ce.__version__
# The full version, including alpha/beta/rc tags.
release = ra2ce.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_preprocess_types = True

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_resources/ra2ce_logo.svg"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "show_nav_level": 2,
    "navbar_align": "content",
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Deltares/ra2ce",  # required
            "icon": "../../_resources/ra2ce_banner.png",
            "type": "url",
        },
        {
            "name": "Deltares",
            "url": "https://www.deltares.nl/en/",
            "icon": "../../_resources/deltares-blue.svg",
            "type": "local",
        },
    ],
    "logo": {
        "text": "RA2CE",
    },
    "navbar_end": ["navbar-icon-links"],  # remove dark mode switch
}

html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise interprise
    "github_user": "Deltares",
    "github_repo": "ra2ce",
    "github_version": "master",  # FIXME
    "doc_path": "docs",
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_resources"]
html_css_files = ["theme-deltares.css"]


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "ra2cedoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "ra2ce.tex",
        "Risk Assessment and Adaptation for Critical infrastructurE Documentation",
        "Margreet van Marle",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "ra2ce",
        "Risk Assessment and Adaptation for Critical infrastructurE Documentation",
        [author],
        1,
    )
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "ra2ce",
        "RA2CE Documentation",
        author,
        "ra2ce",
        "Risk Assessment and Adaptation for Critical infrastructurE Documentation",
        "Miscellaneous",
    ),
]

# Allow errors in notebooks
nbsphinx_allow_errors = True
# Do not execute the scripts during the build process.
nbsphinx_execute = "never"
