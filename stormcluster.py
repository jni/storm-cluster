from collections import namedtuple

import numpy as np
import pandas as pd
import numba
import javabridge
import bioformats
from sklearn.cluster import DBSCAN
from skimage import io


TableParams = namedtuple('TableParams',
                         ['image_shape', 'pixel_scale', 'invert_y'])

DEFAULTPARAMS = TableParams(image_shape=(2048, 2048),
                            pixel_scale=10,
                            invert_y=True)


def find_header_line(filename):
    with open(filename) as fin:
        for i, line in enumerate(fin):
            if line.rstrip() == '##':
                return (i - 1)
    return None


def read_locations_table(filename):
    header_line = find_header_line(filename)
    skiprows = list(range(header_line)) + [header_line + 1]
    table = pd.read_csv(filename, skiprows=skiprows, delimiter='\t')
    return table


def image_coords(location_table, params=DEFAULTPARAMS):
    image_shape = params.image_shape
    scale = params.pixel_scale
    xs, ys = location_table[['X_COORD', 'Y_COORD']].values.T
    rows, cols = ys / scale, xs / scale
    if params.invert_y:
        rows = image_shape[0] - rows
    return rows, cols


def image_coords_indices(location_table, params=DEFAULTPARAMS):
    rows, cols = image_coords(location_table, params)
    rrows, rcols = np.round(rows).astype(int), np.round(cols).astype(int)
    filter_rows = (0 <= rrows) & (rrows < params.image_shape[0])
    filter_cols = (0 <= rcols) & (rcols < params.image_shape[1])
    filter_all = filter_cols & filter_rows
    return rrows[filter_all], rcols[filter_all], filter_all


@numba.njit
def _fill_image(image, rows, cols):
    for i, j in zip(rows, cols):
        image[i, j] += 1


def _stretchlim(image, bottom=0.001, top=None, in_place=True):
    """Stretch the image so new image range corresponds to given quantiles.

    Parameters
    ----------
    image : array, shape (M, N, [...,] P)
        The input image.
    bottom : float, optional
        The lower quantile.
    top : float, optional
        The upper quantile. If not provided, it is set to 1 - `bottom`.
    in_place : bool, optional
        If True, modify the input image in-place (only possible if
        it is a float image).

    Returns
    -------
    out : np.ndarray of float
        The stretched image.
    """
    if in_place and np.issubdtype(image.dtype, np.float):
        out = image
    else:
        out = np.empty(image.shape, np.float32)
        out[:] = image
    if top is None:
        top = 1 - bottom
    q0, q1 = np.percentile(image, [100*bottom, 100*top])
    out -= q0
    out /= q1 - q0
    out = np.clip(out, 0, 1, out=out)
    return out


def image_from_table(location_table, params=DEFAULTPARAMS, stretch=0.001):
    rows, cols, _ = image_coords_indices(location_table)
    image = np.zeros(params.image_shape, dtype=float)
    _fill_image(image, rows, cols)
    image = _stretchlim(image, stretch)
    return image
