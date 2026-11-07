"""scikit-learn's PatchExtractor, with a fallback for 32-bit Python.

This edition's Face Detection Pipeline imports PatchExtractor from here instead of from
sklearn.feature_extraction.image.

scikit-learn builds a strided view of every patch position in an image before it samples patches
from it. The view copies nothing, but NumPy refuses any view whose nominal size exceeds what the
platform can index. In 32-bit Python, which is what runs in a web browser, a large image with large
patches passes that limit, and NumPy raises "array is too big".

When that happens, this class draws the same random positions from the same random state and slices
those patches out of the image one by one. The result is exactly what scikit-learn returns wherever
its view fits. Every call that scikit-learn can serve runs scikit-learn's own code unchanged.

The fallback follows scikit-learn 1.7's sampling; tools/check_image_patches.py compares the two.
"""
import numpy as np
from sklearn.feature_extraction import image as _sk_image
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import validate_data

__all__ = ["PatchExtractor"]


class PatchExtractor(_sk_image.PatchExtractor):
    """sklearn.feature_extraction.image.PatchExtractor, falling back to slicing when NumPy cannot
    build scikit-learn's view of every patch position."""

    def transform(self, X):
        try:
            return super().transform(X)
        except ValueError as error:
            if "array is too big" not in str(error):
                raise
        # scikit-learn fails while building the view, before it draws any random numbers, so the
        # random state below is exactly where scikit-learn's own draws would have started.
        return self._transform_by_slicing(X)

    def _transform_by_slicing(self, X):
        X = validate_data(self, X=X, ensure_2d=False, allow_nd=True, ensure_min_samples=1,
                          ensure_min_features=1, reset=False)
        random_state = check_random_state(self.random_state)
        n_imgs, img_height, img_width = X.shape[:3]
        if self.patch_size is None:
            patch_size = img_height // 10, img_width // 10
        else:
            if len(self.patch_size) != 2:
                raise ValueError("patch_size must be a tuple of two integers. Got"
                                 f" {self.patch_size} instead.")
            patch_size = self.patch_size

        X = np.reshape(X, (n_imgs, img_height, img_width, -1))
        n_channels = X.shape[-1]
        patch_height, patch_width = patch_size
        n_patches = _sk_image._compute_n_patches(img_height, img_width, patch_height,
                                                 patch_width, self.max_patches)
        patches_shape = (n_imgs * n_patches,) + tuple(patch_size)
        if n_channels > 1:
            patches_shape += (n_channels,)

        patches = np.empty(patches_shape)
        for ii, image in enumerate(X):
            patches[ii * n_patches:(ii + 1) * n_patches] = _patches_by_slicing(
                image, patch_size, self.max_patches, random_state)
        return patches


def _patches_by_slicing(image, patch_size, max_patches, random_state):
    """sklearn.feature_extraction.image.extract_patches_2d, slicing each patch out of the image
    instead of indexing a view of all of them."""
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size
    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height of the image.")
    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    n_patches = _sk_image._compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
    else:
        # Every position, in the order scikit-learn's view lists them: row by row.
        i_s, j_s = np.divmod(np.arange(n_patches), i_w - p_w + 1)

    patches = np.empty((n_patches, p_h, p_w, n_colors), dtype=image.dtype)
    for k, (i, j) in enumerate(zip(i_s, j_s)):
        patches[k] = image[i:i + p_h, j:j + p_w]
    # scikit-learn's own reshape, which also rejects a zero-sized patch.
    patches = patches.reshape(-1, p_h, p_w, n_colors)
    if n_colors == 1:
        return patches.reshape((n_patches, p_h, p_w))
    return patches
