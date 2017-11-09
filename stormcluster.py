from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
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


def select_roi(image, alpha=0.4, ax=None):
    """Return a label image based on polygon selections made with the mouse.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparency value for polygons drawn over the image.

    return_all : bool, optional
        If True, an array containing each separate polygon drawn is returned.
        (The polygons may overlap.) If False (default), latter polygons
        "overwrite" earlier ones where they overlap.

    Returns
    -------
    labels : array of int, shape ([Q, ]M, N)
        The segmented regions. If mode is `'separate'`, the leading dimension
        of the array corresponds to the number of regions that the user drew.

    Notes
    -----
    Use left click to select the vertices of the polygon
    and right click to confirm the selection once all vertices are selected.

    Examples
    --------
    >>> from skimage import data, future, io
    >>> camera = data.camera()
    >>> mask = future.manual_polygon_segmentation(camera)  # doctest: +SKIP
    >>> io.imshow(mask)  # doctest: +SKIP
    >>> io.show()  # doctest: +SKIP
    """
    list_of_vertex_lists = []
    polygons_drawn = []

    temp_list = []
    preview_polygon_drawn = []

    if image.ndim not in (2, 3):
        raise ValueError('Only 2D grayscale or RGB images are supported.')

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image, cmap="magma")
    ax.set_axis_off()
    rois = []

    def toggle_selector(event):
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            toggle_selector.RS.set_active(True)

    def onselect(eclick, erelease):
        starts = round(eclick.ydata), round(eclick.xdata)
        ends = round(erelease.ydata), round(erelease.xdata)
        slices = tuple((int(s), int(e)) for s, e in zip(starts, ends))
        rois.append(slices)

    from matplotlib.widgets import RectangleSelector
    toggle_selector.RS = RectangleSelector(ax, onselect)
    ax.figure.canvas.mpl_connect('key_press_event', toggle_selector)
    toggle_selector.RS.set_active(True)
    return rois


def _in_range(arr, tup):
    low, high = tup
    return (low <= arr) & (arr < high)


def cluster(coordinates, radius, core_size):
    scan = DBSCAN(eps=radius, min_samples=core_size).fit(coordinates)
    return scan


def analyse_clustering(scan):
    labels = scan.labels_
    unclustered = labels == -1
    num_unclustered = np.sum(unclustered)
    cluster_sizes = np.bincount(labels[~unclustered])
    counts, bin_edges = np.histogram(cluster_sizes, bins='auto')
    histogram = np.convolve(bin_edges, [0.5, 0.5], 'valid'), counts
    print(f'There are {len(cluster_sizes)} clusters, and {num_unclustered} '
          f'outlier points, out of {labels.size}. The largest cluster size is '
          f'{np.max(cluster_sizes)} and the median is '
          f'{np.median(cluster_sizes)}')
    return labels, cluster_sizes, histogram


def image_from_clustering(scan, coordinates, roi, params=DEFAULTPARAMS,
                          stretch=0.001):
    rows, cols = np.round(coordinates).astype(int).T
    green = np.zeros(params.image_shape, dtype=float)
    _fill_image(green, rows[scan.labels_ > -1], cols[scan.labels_ > -1])
    green = _stretchlim(green, stretch)
    red = np.zeros(params.image_shape, dtype=float)
    _fill_image(red, rows[scan.labels_ == -1], cols[scan.labels_ == -1])
    red = _stretchlim(red, stretch)
    blue = np.zeros_like(red)
    image = np.stack((red, green, blue), axis=-1)
    return image[slice(*roi[0]), slice(*roi[1])]


def parameter_scan_image(coordinates,
                         radii=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
                         core_sizes=(3, 6, 12, 24)):
    num_clustered = np.zeros((len(radii), len(core_sizes)))
    largest_cluster = np.zeros_like(num_clustered)
    for i, r in enumerate(radii):
        for j, c in enumerate(core_sizes):
            scan = cluster(coordinates, r, c)
            clustered = scan.labels_ != -1
            num_clustered[i, j] = np.sum(clustered)
            if np.any(clustered):
                largest_cluster[i, j] = \
                                np.max(np.bincount(scan.labels_[clustered]))
    return num_clustered, largest_cluster
