import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/home/jni/projects/storm-cluster/')
import stormcluster as sc


coords = pd.read_csv('coordinates.csv')
coords.filename = coords.filename + '_LDCTracked.txt'
coords_puncta = coords[['cell', 'filename', 'x', 'y']].copy()
coords_puncta['type'] = 'puncta'
coords_donuts = coords[['cell', 'filename', 'X', 'Y']].rename(columns={'X': 'x', 'Y': 'y'})
coords_donuts['type'] = 'doughnuts'
coords = pd.concat([coords_puncta, coords_donuts])


def select_in(table, x, y, w):
    x *= 1000
    y *= 1000
    w *= 1000
    x_in = (x < table['X_COORD']) & (table['X_COORD'] < x + w)
    y_in = (y < 20480 - table['Y_COORD']) & (20480 - table['Y_COORD'] < y + w)
    return table.loc[x_in & y_in][['Y_COORD', 'X_COORD']]

# tab_donut = select_in(tab1, 8.71, 8.63, 0.2)
# tab_donut.shape
# plt.scatter(tab_donut['X_COORD'], tab_donut['Y_COORD'])
# tab_donut2 = select_in(tab1, 11.81, 8.63, 0.2)
# plt.scatter(tab_donut2['X_COORD'], tab_donut2['Y_COORD'])
# tab_donut3 = select_in(tab1, 12.13, 8.77, 0.2)
# plt.scatter(tab_donut3['X_COORD'], tab_donut3['Y_COORD'])
# plt.scatter(tab_donut3['X_COORD'], tab_donut3['Y_COORD'])

filenames = list(set(coords.filename))
tables = list(map(sc.read_locations_table, filenames))
file2table = dict(zip(filenames, tables))
coords['table'] = coords['filename'].apply(file2table.get)

# Making the figure
coords_fig = coords.rename(columns=str.capitalize)
coords_fig['Type'] = coords_fig['Type'].str.capitalize()

fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=221)

sns.set(font='Arial', font_scale=14/12)
detections = []
for _, _, x, y, _, tab in coords.itertuples(index=False, name=None):
    detections.append(select_in(tab, x, y, 0.2).shape[0])
coords_fig['Detections'] = detections
coords['detections'] = detections

sns.stripplot(x='Type', y='Detections', hue='Cell', data=coords_fig, jitter=True, size=3)
ax.legend(loc='upper left', ncol=2, fontsize='x-small', markerscale=0.3, framealpha=0.3, title='Cell')
fig.tight_layout()

# plt.show()

fig.savefig('detections.tiff', dpi=300)

# compute the ratios, including a manual pivot
mean_per_cell = coords.groupby(['cell', 'type'])['detections'].mean().reset_index()
puncta_cell_means = mean_per_cell.loc[mean_per_cell['type'] == 'puncta'].set_index('cell')
donuts_cell_means = mean_per_cell.loc[mean_per_cell['type'] == 'donuts'].set_index('cell')
ratios = donuts_cell_means['detections'] / puncta_cell_means['detections']

# save the data
(coords[['cell', 'filename', 'x', 'y', 'type', 'detections']]
       .to_csv('detection-counts.csv', index=False))
ratios.to_csv('per-cell-ratios.csv', index=True)
