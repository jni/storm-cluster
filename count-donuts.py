import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Functions to read `_LDCTracked.txt` files
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


# Translate the manual coordinates file into a tidy table
coords = pd.read_csv('coordinates.csv')
coords.filename = coords.filename + '_LDCTracked.txt'
coords_puncta = coords[['cell', 'filename', 'x', 'y']].copy()
coords_puncta['type'] = 'puncta'
coords_donuts = coords[['cell', 'filename', 'X', 'Y']].rename(columns={'X': 'x', 'Y': 'y'})
coords_donuts['type'] = 'doughnuts'
coords = pd.concat([coords_puncta, coords_donuts])


def select_in(table, x, y, w):
    """Given a detection table and square ROI, return detections within it.

    Parameters
    ----------
    table : pandas DataFrame
        A dSTORM detection table, drift corrected. (Usually in file ending in
        ``_LDCTracked.txt``.)
    x, y : float
        x and y coordinates of the top left corner of a square ROI. In µm.
    w : float
        Width (and height) in µm of the ROI.

    Returns
    -------
    roi : pandas DataFrame
        The subset of detections in `table` that fell within the square ROI

    Notes
    -----
    This function hardcodes many details that may not generalise across dSTORM
    microscopes or even experiments, such as the field of view width and the
    scale of coordinates in the table (nm). Do some sanity checks (see below)
    to make sure that the coordinate translations are working for you.
    """
    # convert input coordinates (in µm) to table coordinates (nm)
    x *= 1000
    y *= 1000
    w *= 1000
    # Compute subset selection in x
    x_in = (x < table['X_COORD']) & (table['X_COORD'] < x + w)
    # Compute subset selection in y. For some reason, y coordinates are
    # inverted in the table compared to the image in Fiji.
    y_in = (y < 20480 - table['Y_COORD']) & (20480 - table['Y_COORD'] < y + w)
    # Return the subsetted table
    return table.loc[x_in & y_in][['Y_COORD', 'X_COORD']]

# The code below is useless in production, but helps to check that our
# coordinate transforms are correct: we should see a donut made by the
# detection points.
# tab1 = sc.read_locations_table('CS2_22_01_405_01_LDCTracked.txt')
# tab_donut = select_in(tab1, 8.71, 8.63, 0.2)
# tab_donut.shape
# plt.scatter(tab_donut['X_COORD'], tab_donut['Y_COORD'])
# tab_donut2 = select_in(tab1, 11.81, 8.63, 0.2)
# plt.scatter(tab_donut2['X_COORD'], tab_donut2['Y_COORD'])
# tab_donut3 = select_in(tab1, 12.13, 8.77, 0.2)
# plt.scatter(tab_donut3['X_COORD'], tab_donut3['Y_COORD'])


# For each ROI (row in the dataset), include the detection table that it came
# from.
filenames = list(set(coords.filename))
tables = list(map(read_locations_table, filenames))
file2table = dict(zip(filenames, tables))
coords['table'] = coords['filename'].apply(file2table.get)

# Iterate over the rows to find the number of detections corresponding to each
# ROI.
detections = []
for _, _, x, y, _, tab in coords.itertuples(index=False, name=None):
    detections.append(select_in(tab, x, y, 0.2).shape[0])
coords['detections'] = detections

# Making the figure
# -----------------

## First, capitalise titles
coords_fig = coords.rename(columns=str.capitalize)
coords_fig['Type'] = coords_fig['Type'].str.capitalize()

# make subplot of correct size
fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=221)

# set the font type and size.
# font size should be 14pt but Seaborn specifies relative to matplotlib
# default, which turns out to be 12pt.
sns.set(font='Arial', font_scale=14/12)

# Use a seaborn stripplot to show all the data
sns.stripplot(x='Type', y='Detections', hue='Cell', data=coords_fig,
              jitter=0.2, size=4, palette=sns.husl_palette(11, l=0.6))
# create the legend. Use bbox_to_anchor to place outside axes as per SO:
# https://stackoverflow.com/a/43439132/224254
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1,
                   fontsize='x-small', markerscale=0.4, framealpha=0.3,
                   title='Cell')
plt.setp(legend.get_title(), fontsize='small')

# ensure everything fits
fig.tight_layout()

# uncomment the line below if working interactively
# plt.show()

# save to file
fig.savefig('detections.tiff', dpi=300)

# Save the dataset
# ----------------

# compute the ratios, including a manual pivot
mean_per_cell = coords.groupby(['cell', 'type'])['detections'].mean().reset_index()
puncta_cell_means = mean_per_cell.loc[mean_per_cell['type'] == 'puncta'].set_index('cell')
donuts_cell_means = mean_per_cell.loc[mean_per_cell['type'] == 'doughnuts'].set_index('cell')
ratios = donuts_cell_means['detections'] / puncta_cell_means['detections']

# save the data
(coords[['cell', 'filename', 'x', 'y', 'type', 'detections']]
       .to_csv('detection-counts.csv', index=False))
ratios.to_csv('per-cell-ratios.csv', index=True)
