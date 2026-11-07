"""Check that notebooks/image_patches.py gives exactly scikit-learn's patches.

The module's fallback (slicing each patch) only runs where scikit-learn's own code cannot, which on
64-bit Python is never, so this calls the fallback directly and compares it with scikit-learn's
PatchExtractor on the same inputs: grey and colour images, one image and several, integer, fractional
and no max_patches, and integer and RandomState random states. It also runs the 05.14 notebook's own
call. Run it with the library versions the browser uses, for example:

    uv run --no-project --python 3.12 --with scikit-learn==1.7.0 --with numpy==2.2.5 \\
        --with scikit-image==0.25.2 python tools/check_image_patches.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.image import PatchExtractor as SkPatchExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
from image_patches import PatchExtractor  # noqa: E402


def compare(name, X, **params):
    """Run scikit-learn and the fallback on the same input, each with its own RandomState(seed) when
    the case uses one, and require the same patches (or the same error) and the same final random state."""
    seed = params.pop("seed", None)

    def run(transform):
        rs = np.random.RandomState(seed) if seed is not None else params.get("random_state")
        try:
            out = transform(dict(params, random_state=rs) if seed is not None else params)
        except ValueError as error:
            out = f"ValueError: {error}"
        state = rs.get_state()[1].tobytes() if isinstance(rs, np.random.RandomState) else None
        return out, state

    expected, expected_state = run(lambda p: SkPatchExtractor(**p).transform(X))
    got, got_state = run(lambda p: PatchExtractor(**p)._transform_by_slicing(X))
    if isinstance(expected, str) or isinstance(got, str):
        same = isinstance(expected, str) and isinstance(got, str) and expected == got
        shown = got if isinstance(got, str) else f"{got.shape} {got.dtype}, expected {expected}"
    else:
        same = expected.dtype == got.dtype and expected.shape == got.shape and np.array_equal(expected, got)
        shown = f"{got.shape} {got.dtype}"
    same = same and expected_state == got_state
    print(f"{'ok  ' if same else 'DIFF'} {name}: {shown}")
    return same


def main():
    rng = np.random.RandomState(1)
    grey = rng.rand(2, 40, 37)
    colour = rng.randint(0, 255, size=(3, 30, 33, 3)).astype(np.uint8)
    one = rng.rand(1, 25, 25)
    checks = [
        compare("grey, 2 images, 50 patches, seed 0", grey, patch_size=(6, 5), max_patches=50, random_state=0),
        compare("grey, fraction 0.3, seed 7", grey, patch_size=(9, 4), max_patches=0.3, random_state=7),
        compare("grey, every patch", one, patch_size=(5, 6)),
        compare("grey, max_patches above the total", one, patch_size=(20, 20), max_patches=10_000, random_state=3),
        compare("colour uint8, 3 images, 40 patches, seed 2", colour, patch_size=(7, 8), max_patches=40, random_state=2),
        compare("colour, every patch", colour[:1], patch_size=(25, 30)),
        compare("RandomState shared across images", grey, patch_size=(6, 5), max_patches=25, seed=0),
        compare("zero-height default patch: scikit-learn's error", np.zeros((2, 8, 60), dtype=np.uint8),
                max_patches=2, seed=17),
        compare("default patch size", grey, max_patches=10, random_state=4),
    ]

    # The notebook's own call: a skimage test image at the three scales of LFW's face size.
    from skimage import color, data, transform
    img = color.rgb2gray(data.chelsea())
    for scale in (0.5, 1.0, 2.0):
        size = tuple((scale * np.array((62, 47))).astype(int))
        checks.append(compare(f"05.14 call, chelsea, scale {scale}", img[np.newaxis], patch_size=size,
                              max_patches=1000, random_state=0))

    # The real transform() takes scikit-learn's path wherever it works, and fails over only on
    # NumPy's size error.
    X = grey
    assert np.array_equal(PatchExtractor(patch_size=(6, 5), max_patches=5, random_state=0).transform(X),
                          SkPatchExtractor(patch_size=(6, 5), max_patches=5, random_state=0).transform(X))
    # Any other error is scikit-learn's own, passed through unchanged.
    errors = []
    for cls in (SkPatchExtractor, PatchExtractor):
        try:
            cls(patch_size=(60, 5), max_patches=5, random_state=0).transform(X)
        except ValueError as error:
            errors.append(str(error))
    assert len(errors) == 2 and errors[0] == errors[1], errors

    print(f"{sum(checks)}/{len(checks)} identical")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
