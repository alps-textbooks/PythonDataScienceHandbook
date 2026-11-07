"""scikit-learn datasets loaded from copies that come with this edition.

scikit-learn's ``fetch_olivetti_faces``, ``fetch_20newsgroups`` and
``fetch_lfw_people`` download their data from figshare the first time they
run. In the browser that download can be refused, and the photos behind
``fetch_lfw_people`` (243 MB) are more than the browser can load, so this
edition bundles the data in ``data/sklearn``. The functions here take the
arguments of the scikit-learn functions of the same names and return the same
objects, loaded from those files.

``tools/build_bundled_datasets.py`` makes the files and checks these functions
against scikit-learn's own.
"""

import os
import shutil

import numpy as np
from sklearn import datasets
from sklearn.utils import Bunch

__all__ = ["fetch_olivetti_faces", "fetch_20newsgroups", "fetch_lfw_people"]

# The LFW bundle holds every person with at least this many photos.
LFW_MIN_FACES = 20
LFW_FILE = "lfw_people_min20.npz"
_LFW_SLICE = (slice(70, 195), slice(78, 172))


def _open_bundled(name):
    """Open a file from data/sklearn, next to this module or the notebook."""
    tried = []
    for base in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = os.path.join(base, "data", "sklearn", name)
        if path in tried:
            continue
        tried.append(path)
        try:
            return open(path, "rb")
        except OSError:
            pass
    raise FileNotFoundError(f"{name} is not in this edition's data/sklearn folder")


def _install(name, data_home):
    """Copy a bundled scikit-learn cache file into scikit-learn's data folder,
    where scikit-learn's own loader finds it instead of downloading."""
    home = datasets.get_data_home(data_home)
    target = os.path.join(home, name)
    if not os.path.exists(target):
        partial = target + ".part"
        with _open_bundled(name) as src, open(partial, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(partial, target)
    return home


def fetch_olivetti_faces(*, data_home=None, download_if_missing=True, **kwargs):
    """``sklearn.datasets.fetch_olivetti_faces``, from the bundled copy."""
    home = _install("olivetti_py3.pkz", data_home)
    return datasets.fetch_olivetti_faces(
        data_home=home, download_if_missing=False, **kwargs
    )


def fetch_20newsgroups(*, data_home=None, download_if_missing=True, **kwargs):
    """``sklearn.datasets.fetch_20newsgroups``, from the bundled copy."""
    home = _install("20news-bydate_py3.pkz", data_home)
    return datasets.fetch_20newsgroups(
        data_home=home, download_if_missing=False, **kwargs
    )


def _lfw_description():
    try:
        from sklearn.datasets._base import load_descr

        return load_descr("lfw.rst")
    except Exception:
        return ""


def fetch_lfw_people(
    *,
    data_home=None,
    funneled=True,
    resize=0.5,
    min_faces_per_person=0,
    color=False,
    slice_=_LFW_SLICE,
    download_if_missing=True,
    return_X_y=False,
    n_retries=3,
    delay=1.0,
):
    """``sklearn.datasets.fetch_lfw_people``, from the bundled copy.

    The bundle holds the faces of everyone with at least 20 photos, cropped
    and scaled as scikit-learn does by default. Any ``min_faces_per_person``
    of 20 or more gives exactly the faces, labels and order scikit-learn
    returns; other sizes, crops or colour are not bundled.
    """
    if (
        not funneled
        or resize != 0.5
        or color
        or tuple(slice_ or ()) != _LFW_SLICE
        or min_faces_per_person < LFW_MIN_FACES
    ):
        raise ValueError(
            "This edition bundles the LFW faces of people with at least "
            f"{LFW_MIN_FACES} photos, in grayscale at scikit-learn's default "
            "size and crop. Use min_faces_per_person of 20 or more and leave "
            "funneled, resize, color and slice_ at their defaults."
        )
    with _open_bundled(LFW_FILE) as f:
        with np.load(f) as bundle:
            faces = bundle["images"]
            names = bundle["names"]

    # As scikit-learn selects: everyone with enough photos, in the bundle's
    # order, which is scikit-learn's own (people, then photos, by name).
    people, counts = np.unique(names, return_counts=True)
    keep = np.isin(names, people[counts >= min_faces_per_person])
    if not keep.any():
        raise ValueError(
            "min_faces_per_person=%d is too restrictive" % min_faces_per_person
        )
    faces, person_names = faces[keep], names[keep].tolist()
    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    # scikit-learn's fixed shuffle, so the faces come in the same order.
    indices = np.arange(len(faces))
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]

    data = faces.reshape(len(faces), -1)
    if return_X_y:
        return data, target
    return Bunch(
        data=data,
        images=faces,
        target=target,
        target_names=target_names,
        DESCR=_lfw_description(),
    )
