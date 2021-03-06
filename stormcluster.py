import os
import sys
from collections import namedtuple

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg
                                                as FigureCanvas)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numba
from sklearn.cluster import DBSCAN
from skimage import io, color


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
    if q1 > q0:
        out /= q1 - q0
    out = np.clip(out, 0, 1, out=out)
    return out


def image_from_table(location_table, params=DEFAULTPARAMS, stretch=0.001):
    rows, cols, _ = image_coords_indices(location_table)
    image = np.zeros(params.image_shape, dtype=float)
    _fill_image(image, rows, cols)
    image = _stretchlim(image, stretch)
    return image


def select_roi(image, rois=None, ax=None, axim=None, qtapp=None):
    """Return a label image based on polygon selections made with the mouse.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.
    rois : list, optional
        If given, append ROIs to this existing list. Otherwise a new list
        object will be created.
    ax : matplotlib Axes, optional
        The Axes on which to do the plotting.
    axim : matplotlib AxesImage, optional
        An existing AxesImage on which to show the image.
    qtapp : QtApplication
        The main Qt application for ROI selection. If given, the ROIs will
        be inserted at the right location for the image index.

    Returns
    -------
    rois : list of tuple of ints
        The selected regions, in the form
        [[(row_start, row_end), (col_start, col_end)]].

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
    if image.ndim not in (2, 3):
        raise ValueError('Only 2D grayscale or RGB images are supported.')

    if ax is None and axim is None:
        fig, ax = plt.subplots()
    if axim is None:
        ax.clear()
        axim = ax.imshow(image, cmap="magma")
        ax.set_axis_off()
    else:
        axim.set_array(image)
    rois = rois or []

    def toggle_selector(event):
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            toggle_selector.RS.set_active(True)

    def onselect(eclick, erelease):
        starts = round(eclick.ydata), round(eclick.xdata)
        ends = round(erelease.ydata), round(erelease.xdata)
        slices = tuple((int(s), int(e)) for s, e in zip(starts, ends))
        if qtapp is None:
            rois.append(slices)
        else:
            index = qtapp.image_index
            rois[index] = slices
            qtapp.rectangle_selector.set_active(False)
            qtapp.select_next_image()
            qtapp.rectangle_selector.set_active(True)

    selector = RectangleSelector(ax, onselect, useblit=True)
    if qtapp is None:
        # Ensure that the widget remains active by creating a reference to it.
        # There's probably a better place to put that reference but this will do
        # for now. (From the matplotlib RectangleSelector gallery example.)
        toggle_selector.RS = selector
    else:
        qtapp.rectangle_selector = selector
    ax.figure.canvas.mpl_connect('key_press_event', toggle_selector)
    selector.set_active(True)
    return rois


def _in_range(arr, tup):
    low, high = tup
    return (low <= arr) & (arr < high)


def cluster(coordinates, radius, core_size):
    scan = DBSCAN(eps=radius, min_samples=core_size).fit(coordinates)
    return scan


def _centroids(labels, sizes, coords):
    clustered = labels > -1
    labels, coords = labels[clustered], coords[clustered]
    grouping = np.argsort(labels)
    sums = np.add.reduceat(coords[grouping], np.cumsum(sizes)[:-1])
    means = sums / sizes[1:, np.newaxis]
    return means


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


def bbox_diameter(coords):
    return np.hypot(*np.max(coords, axis=0) - np.min(coords, axis=0))


def summarise_clustering(scan, coords):
    labels, sizes, hist = analyse_clustering(scan)
    diameters = labeled_comprehension(coords, scan.labels_,
                                      bbox_diameter)[1:, np.newaxis]
    cluster_centroids = _centroids(labels, sizes, coords)
    cluster_sq_centroids = _centroids(labels, sizes, coords ** 2)
    centroid_vars = np.sqrt(cluster_sq_centroids - cluster_centroids ** 2)
    column_names = ['centroid row', 'centroid column',
                    'std dev row', 'std dev column',
                    'detections', 'cluster id', 'diameter']
    idxs = np.arange(1, np.max(labels) + 1)
    columns = (cluster_centroids, centroid_vars,
               sizes[1:, np.newaxis], idxs[:, np.newaxis], diameters)
    print([c.shape for c in columns])
    data = np.concatenate(columns, axis=1)
    df = pd.DataFrame(data=data, columns=column_names)
    return df


def labeled_comprehension(features, labels, function, *args,
                          extra_args=(), extra_kwargs={}, **kwargs):
    """Like ndi.labeled_comprehension, but features can be higher D."""
    nonneg = labels >= 0
    features = features[nonneg]
    labels = labels[nonneg]
    sorter = np.argsort(labels)
    labels = labels[sorter]
    features = features[sorter]
    boundaries = np.concatenate([[0], np.flatnonzero(np.diff(labels)) + 1])
    args = args + extra_args
    kwargs = {**kwargs, **extra_kwargs}
    if hasattr(function, 'reduceat'):
        result = function.reduceat(features, boundaries, *args, **kwargs)
    else:
        boundaries = np.concatenate((boundaries, [labels.size]))
        test_index = min(3, features.shape[0])
        test_value = function(features[:test_index], *args, **kwargs)
        result_shape = (boundaries.size - 1,) + np.shape(test_value)
        result = np.empty(result_shape)
        for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            result[i] = function(features[start:stop], *args, **kwargs)
    return result


def cluster_circle_plot(image, coordinates, data, roi, params=DEFAULTPARAMS,
                        stretch=0.001, size_threshold=None, max_size=None,
                        ax=None):
    colors = ['gray', 'C9', 'C1', 'C2']
    ax = ax or plt.subplots()[1]
    image = _stretchlim(image, stretch)
    ax.imshow(image, cmap='gray')
    ax.set_xlim(*roi[1])
    ax.set_ylim(*roi[0])
    if size_threshold is None:
        size_threshold = 0
    if max_size is None:
        max_size = np.inf
    for y, x, *_, d in data.values:
        color_index = int(np.digitize(d, [0, size_threshold, max_size]))
        color = colors[color_index]
        alpha = 0.9 - (color_index > 2) * 0.5
        ax.add_artist(plt.Circle((x, y), d/2, fill=False, color=color,
                                 linewidth=0.5, alpha=alpha))
    return ax


def image_from_clustering(scan, coordinates, roi, params=DEFAULTPARAMS,
                          stretch=0.001, size_threshold=None, max_size=None,
                          use_diameter=True):
    unclustered = scan.labels_ == -1
    if size_threshold is not None:
        temp_labels = np.copy(scan.labels_)
        temp_labels[unclustered] = 0
        if use_diameter:
            cluster_sizes = labeled_comprehension(coordinates, scan.labels_,
                                                  bbox_diameter)
        else:
            cluster_sizes = np.bincount(scan.labels_[~unclustered])
        large = ((cluster_sizes > size_threshold)[temp_labels] &
                 (~unclustered))
        if max_size is not None:
            too_large = (((cluster_sizes > max_size)[temp_labels] &
                         (~unclustered)))
        else:
            too_large = np.zeros_like(large)
        large &= ~too_large
        small = (~large) & (~too_large) & (~unclustered)
    else:
        large = ~unclustered
        small = np.zeros_like(large)
        too_large = small
    rows, cols = np.round(coordinates).astype(int).T

    green = np.zeros(params.image_shape, dtype=float)
    if np.any(large):
        _fill_image(green, rows[large], cols[large])
        green = _stretchlim(green, stretch)
        green[np.isnan(green)] = 0

    red = np.zeros_like(green, dtype=float)
    if np.any(small):
        _fill_image(red, rows[small], cols[small])
        red = _stretchlim(red, stretch)
        red[np.isnan(red)] = 0

    blue = np.zeros_like(red)
    if np.any(unclustered):
        _fill_image(blue, rows[unclustered], cols[unclustered])
        blue = _stretchlim(blue, stretch)
        blue[np.isnan(blue)] = 0
    # green += blue  # make cyan

    gray = np.zeros_like(red)
    if np.any(too_large):
        _fill_image(gray, rows[too_large], cols[too_large])
        gray = _stretchlim(gray, stretch)
        gray[np.isnan(gray)] = 0

    image = np.stack((red, green, blue), axis=-1)
    image[gray > 0] = (gray[gray > 0] * 0.5)[..., np.newaxis]

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


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100,
                 image_size=(2048, 2048), image_cmap='magma'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_axes([0, 0, 1, 1])
        self.image_size = image_size

        self.axim = self.axes.imshow(np.broadcast_to(0., image_size),
                                     cmap=image_cmap, vmin=0, vmax=1)
        self.axes.set_axis_off()
        fig.set_facecolor('black')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def set_image(self, image):
        self.axim.set_array(image)
        self.draw_idle()

    def new_image(self, image):
        if image.ndim == 3:
            self.axim = self.axes.imshow(image)
        else:
            self.axim = self.axes.imshow(image, cmap='magma')
        self.axes.set_axis_off()
        self.draw_idle()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_index = 0
        self.files = []
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Select ROIs')

        self.main_widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.image_canvas = ImageCanvas(self.main_widget)
        layout.addWidget(self.image_canvas)

        select_files = QtWidgets.QPushButton(text='Select Files')
        select_files.clicked.connect(self.open_files)
        previous_image = QtWidgets.QPushButton(text='◀')
        previous_image.clicked.connect(self.select_previous_image)
        next_image = QtWidgets.QPushButton(text='▶')
        next_image.clicked.connect(self.select_next_image)
        nav_layout = QtWidgets.QHBoxLayout(self.main_widget)
        nav = QtWidgets.QGroupBox(self.main_widget)
        nav_layout.addWidget(previous_image)
        nav_layout.addWidget(next_image)
        nav.setLayout(nav_layout)
        buttons = QtWidgets.QHBoxLayout(self.main_widget)
        buttons.addWidget(select_files, alignment=QtCore.Qt.AlignLeft)
        buttons.addWidget(nav, alignment=QtCore.Qt.AlignRight)
        layout.addLayout(buttons)
        self.image_canvas.draw()

        self.rois = []
        self._mode = 'roi'

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    @QtCore.pyqtSlot()
    def open_files(self):
        files, resp = QtWidgets.QFileDialog.getOpenFileNames(
                                    self.main_widget, filter='*LDCTracked.txt')
        self.open_file_list(files)

    def open_file_list(self, files):
        self.files = files
        print(self.files)
        self.set_image_index(0)
        self.rois = [(slice(None), slice(None))] * len(self.files)
        image = self.image_canvas.axim.get_array()
        axes = self.image_canvas.axes
        axim = self.image_canvas.axim
        select_roi(image, self.rois, ax=axes, axim=axim, qtapp=self)

    def set_image_index(self, i=0):
        if i == len(self.files):
            if self._mode == 'roi':
                print(f'rois: {self.rois}')
                self.clustering_mode()
                i = 0
            else:  # _mode == 'clustering'
                self.save_clustering_results()
                return
        if len(self.files) > 0:
            i = np.clip(i, 0, len(self.files) - 1)
            self.image_index = i
            if self._mode == 'roi':
                file = self.files[i]
                table = read_locations_table(file)
                image = image_from_table(table)
                self.image_canvas.set_image(image)
            else:
                if i == len(self.images):
                    scan, coords = self.make_cluster_image(i)
                    df = summarise_clustering(scan, coords)
                    df['filename'] = os.path.basename(self.files[i])
                    self.cluster_data.append(df)
                image = self.images[i]
                self.image_canvas.new_image(image)

    @QtCore.pyqtSlot()
    def select_previous_image(self):
        self.set_image_index(self.image_index - 1)

    @QtCore.pyqtSlot()
    def select_next_image(self):
        self.set_image_index(self.image_index + 1)

    def clustering_mode(self):
        self._mode = 'clustering'
        self.images = []
        self.cluster_data = []
        self.set_image_index(0)

    def make_cluster_image(self, i, radius=1.6, core_size=6):
        cluster_size_threshold = 10
        cluster_max_size = 30
        file = self.files[i]
        roi = self.rois[i]
        table = read_locations_table(file)
        coords = np.stack(image_coords(table), axis=1)
        coords_roi = coords[_in_range(coords[:, 0], roi[0]) &
                            _in_range(coords[:, 1], roi[1])]
        scan = cluster(coords_roi, radius=radius, core_size=core_size)
        image = image_from_clustering(scan, coords_roi, roi,
                                      size_threshold=cluster_size_threshold,
                                      max_size=cluster_max_size)
        if i >= len(self.images):
            self.images.append(image)
        else:
            self.images[i] = image
        return scan, coords_roi

    def save_clustering_results(self):
        input_dir = os.path.dirname(self.files[0])
        name = QtWidgets.QFileDialog.getSaveFileName(self.main_widget,
                                                     'Save clusters to file',
                                                     directory=input_dir,
                                                     filter='*.xlsx')[0]
        print(name)
        cluster_table = pd.concat(self.cluster_data)
        print(name)
        cluster_table.to_excel(name)


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv[:1])
    aw = ApplicationWindow()
    aw.show()
    if len(sys.argv) > 1:
        aw.open_file_list(sys.argv[1:])
    sys.exit(qApp.exec_())
