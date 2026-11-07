"""Build the scikit-learn datasets this edition bundles, then check them.

Writes notebooks/data/sklearn/:
  olivetti_py3.pkz        scikit-learn's own cache of the Olivetti faces
  20news-bydate_py3.pkz   scikit-learn's own cache of 20 Newsgroups, with the
                          file names made relative to the data folder
  lfw_people_min20.npz    the LFW faces of everyone with at least 20 photos, as
                          fetch_lfw_people crops and scales them, in
                          scikit-learn's order, with each face's person

notebooks/bundled_datasets.py loads them. After building, this script compares
its results with scikit-learn's for the calls the book makes, and fails on any
difference.

Build with the library versions the browser runs (Pyodide 0.29.3), so the
caches load there unchanged:

  uv run --no-project --python 3.13 --with scikit-learn==1.7.0 \\
    --with numpy==2.2.5 --with joblib==1.4.2 --with pillow==11.3.0 --with scipy \\
    python tools/build_bundled_datasets.py --download-dir <scratch folder>

The download folder receives scikit-learn's downloads (about 270 MB) and can be
deleted afterwards.
"""

import argparse
import codecs
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn import datasets
from sklearn.datasets._lfw import _check_fetch_lfw, _load_imgs

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
OUT = NOTEBOOKS / "data" / "sklearn"
LFW_MIN_FACES = 20
LFW_SLICE = (slice(70, 195), slice(78, 172))
NEWS_CATEGORIES = ["talk.religion.misc", "soc.religion.christian", "sci.space", "comp.graphics"]


def build_olivetti(download_dir):
    datasets.fetch_olivetti_faces(data_home=download_dir)
    shutil.copyfile(Path(download_dir) / "olivetti_py3.pkz", OUT / "olivetti_py3.pkz")


def build_20newsgroups(download_dir):
    datasets.fetch_20newsgroups(data_home=download_dir)
    cache_path = Path(download_dir) / "20news-bydate_py3.pkz"
    cache = pickle.loads(codecs.decode(cache_path.read_bytes(), "zlib_codec"))
    # The cache records where the files were unpacked on this machine; keep
    # only the part below the data folder.
    home = str(Path(download_dir).resolve()) + os.sep
    for part in cache.values():
        part.filenames = np.array(
            [str(Path(f).resolve()).replace(home, "") for f in part.filenames]
        )
    (OUT / "20news-bydate_py3.pkz").write_bytes(
        codecs.encode(pickle.dumps(cache), "zlib_codec")
    )


def build_lfw(download_dir):
    _, folder = _check_fetch_lfw(data_home=download_dir, funneled=True)
    # The same scan fetch_lfw_people makes, without its shuffle.
    names, paths = [], []
    for person in sorted(os.listdir(folder)):
        person_dir = os.path.join(folder, person)
        if not os.path.isdir(person_dir):
            continue
        photos = [os.path.join(person_dir, f) for f in sorted(os.listdir(person_dir))]
        if len(photos) >= LFW_MIN_FACES:
            names.extend([person.replace("_", " ")] * len(photos))
            paths.extend(photos)
    faces = _load_imgs(paths, LFW_SLICE, False, 0.5)
    np.savez_compressed(OUT / "lfw_people_min20.npz", images=faces, names=np.array(names))


def check(download_dir):
    """Compare bundled_datasets with scikit-learn on the calls the book makes."""
    sys.path.insert(0, str(NOTEBOOKS))
    import bundled_datasets as bundled

    fresh = tempfile.mkdtemp(prefix="sklearn-data-")
    failures = []

    def equal(a, b):
        # Lists (20 Newsgroups' documents and names) compare as lists: as a
        # NumPy array, every document would be padded to the longest one.
        if isinstance(b, list):
            return isinstance(a, list) and a == b
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and np.array_equal(a, b)

    def same(label, ours, theirs):
        ok = all(equal(ours[k], theirs[k]) for k in theirs if k not in ("DESCR", "filenames"))
        print(f"  {'same' if ok else 'DIFFERENT'}  {label}")
        if not ok:
            failures.append(label)

    print("Olivetti faces")
    for kw in ({}, {"shuffle": True, "random_state": 0}):
        same(f"fetch_olivetti_faces({kw})",
             bundled.fetch_olivetti_faces(data_home=fresh, **kw),
             datasets.fetch_olivetti_faces(data_home=download_dir, **kw))
    print("20 Newsgroups")
    for kw in ({}, {"subset": "train", "categories": NEWS_CATEGORIES},
               {"subset": "test", "categories": NEWS_CATEGORIES},
               {"subset": "all", "remove": ("headers", "footers", "quotes")}):
        same(f"fetch_20newsgroups({kw})",
             bundled.fetch_20newsgroups(data_home=fresh, **kw),
             datasets.fetch_20newsgroups(data_home=download_dir, **kw))
    print("LFW people")
    for k in (20, 30, 60, 70):
        same(f"fetch_lfw_people(min_faces_per_person={k})",
             bundled.fetch_lfw_people(min_faces_per_person=k),
             datasets.fetch_lfw_people(data_home=download_dir, min_faces_per_person=k))
    shutil.rmtree(fresh, ignore_errors=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--download-dir", required=True,
                        help="scratch folder for scikit-learn's downloads")
    parser.add_argument("--check-only", action="store_true",
                        help="check the files already in notebooks/data/sklearn")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.check_only:
        build_olivetti(args.download_dir)
        build_20newsgroups(args.download_dir)
        build_lfw(args.download_dir)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
    failures = check(args.download_dir)
    if failures:
        sys.exit(f"{len(failures)} check(s) differ from scikit-learn")
    print("All checks match scikit-learn.")


if __name__ == "__main__":
    main()
