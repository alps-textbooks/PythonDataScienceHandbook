# Tools

These are tools for managing the notebooks in this repository.

- ``generate_contents.py``: this will generate a markdown table of contents for use in the README and in the Index.ipynb notebook

- ``add_navigation.py``: this script adds navigation links at the top and bottom of each notebook.

- ``add_book_info.py``: this script adds book information to the top of each notebook.

- ``build_bundled_datasets.py``: this builds the scikit-learn datasets the Alps edition bundles in ``notebooks/data/sklearn`` (Olivetti faces, 20 Newsgroups, and the LFW faces of everyone with at least 20 photos), and checks ``notebooks/bundled_datasets.py`` against scikit-learn. Its docstring gives the library versions to build with.
- ``check_image_patches.py``: this checks ``notebooks/image_patches.py``, whose ``PatchExtractor`` the Alps edition's face detection page uses, against scikit-learn's: its slicing fallback must give exactly scikit-learn's patches. Its docstring gives the library versions to run it with.
