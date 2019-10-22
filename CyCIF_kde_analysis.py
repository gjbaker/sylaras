import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import OrderedDict
from numpy.random import randint
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import seaborn as sns
import numpy as np

# map matplotlib color codes to the default seaborn palette
sns.set()
sns.set_color_codes()
_ = plt.plot([0, 1], color='r')
sns.set_color_codes()
_ = plt.plot([0, 2], color='b')
sns.set_color_codes()
_ = plt.plot([0, 3], color='g')
sns.set_color_codes()
_ = plt.plot([0, 4], color='m')
sns.set_color_codes()
_ = plt.plot([0, 5], color='y')
plt.close('all')

# display adjustments
pd.set_option('display.width', None)
pd.options.display.max_rows = 150
pd.options.display.max_columns = 33

if len(sys.argv) != 2:
    print('Usage: CycIF.py <path_to_project>')
    sys.exit()

project_path = sys.argv[1]

if not os.path.exists(project_path):
    print('Project path does not exist')
    sys.exit()
if not os.path.exists(os.path.join(project_path, 'raw')):
    print("This does not look like a project folder ('raw' data folder must"
          " exist; however, do not include in file path.)")
    sys.exit()


def makeColors(vals):
    colors = np.zeros((len(vals), 3))
    norm = Normalize(vmin=vals.min(), vmax=vals.max())

    # put any colormap you like here
    colors = [
        cm.ScalarMappable(
            norm=norm, cmap='Greys').to_rgba(val) for val in vals]  # 'jet'

    return colors


data_path = '/Users/gjbaker/Dropbox (HMS)/Baker_et_al_2018/' \
    'Baker_2019_04_11/CycIF_9(36dpi_mouse1_rep_7)'
save_dir = '/Users/gjbaker/Desktop/CycIF_9_output'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    pass

os.chdir(data_path)
data = pd.read_csv('export_SS_data_TME.csv')

current_cols = data.columns.tolist()
update_cols = ['area', 'dapi_3', 'dapi_4', 'foxp3', 'ki67', 'vimentin', 'ly6c',
               'ly6g', 'tile', 'local_x', 'local_y', 'b220', 'tile_x',
               'tile_y', 'global_x', 'global_y', 'cd11b', 'cd4', 'cd49b',
               'cd68', 'cd8a', 'dapi_1', 'dapi_2']

data.columns = update_cols

channel_biases = {'foxp3': 200, 'b220': 400, 'cd11b': 10000, 'cd4': 75,
                  'cd68': 8000, 'cd8a': 400}

Boo_data = data.copy()
for key, value in channel_biases.items():

    print('Converting ' + key + ' protein expression data into its '
          'Boolean represenation.')

    data_col = Boo_data[key]

    for i, v in enumerate(data_col):

        if v > value:
            data_col.iat[i] = 1
        elif v <= value:
            data_col.iat[i] = 0

    Boo_data[key] = data_col.astype(int)
print()

bips = Boo_data.groupby(
    list(channel_biases.keys())).size().reset_index().rename(
    columns={0: 'count'}).sort_values(
        by='count', ascending=False, inplace=False).reset_index(drop=True)

# compute percentages with tumor cells included
t_bips = bips.copy()
t_bips['percent'] = t_bips['count']/t_bips['count'].sum()*100

t_to_map = t_bips[list(channel_biases.keys())].head(8)

t_celltypes = {'GL261': 'gainsboro', 'CD11b+': 'limegreen', 'CD4+': 'red',
               'CD8a+': 'yellow', 'CD68+': 'cyan', 'B220+': 'magenta',
               'Foxp3+CD4+': 'blue', 'B220+CD8a+': 'grey'}

t_celltypes_order = ['GL261', 'CD11b+', 'CD4+', 'CD8a+', 'CD68+',
                     'B220+', 'Foxp3+CD4+', 'B220+CD8a+']

t_celltypes = OrderedDict(
    sorted(t_celltypes.items(), key=lambda x: t_celltypes_order.index(x[0])))

t_to_map['cell_type'] = list(t_celltypes.keys())
print('Check that BIP names match Boolean vectors.')
print(t_to_map)

t_mapped = Boo_data.merge(
    t_to_map, on=list(channel_biases.keys()), how='right')

# plot BIP cell count
t_counts = t_mapped.groupby('cell_type').size().reset_index().rename(
    columns={0: 'count'}).sort_values(
        by='count', ascending=False, inplace=False).reset_index(drop=True)

# filter out GL261 cells
sns.set(style='whitegrid')
filt_t_counts = t_counts[t_counts['cell_type'] != 'GL261']
plt.plot(
    filt_t_counts['cell_type'], filt_t_counts['count'])
    # color=[
    #     v for (k, v) in t_celltypes.items()
    #     if k in list(filt_t_counts['cell_type'])]
plt.xticks(rotation=90)
plt.xlabel('cell type')
plt.ylabel('count', size=15)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cell_counts.pdf'))
plt.close()

# plot global x and y coordinates pseudocolored according to their
# kernel density estimate
densObj = kde(t_mapped.loc[:, ['global_x', 'global_y']].T)
colors = makeColors(
    densObj.evaluate(t_mapped.loc[:, ['global_x', 'global_y']].T))

vals = densObj.evaluate(t_mapped.loc[:, ['global_x', 'global_y']].T)
cmap = plt.cm.get_cmap('Greys')  # 'jet'
norm = mpl.colors.Normalize(vmin=vals.min(), vmax=vals.max())

fig = plt.figure(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
gs_inner = GridSpec(8, 21)
scat = fig.add_subplot(gs_inner[0:19, 0:19])
map = fig.add_subplot(gs_inner[0:19, 20:21])

scat.scatter(
    t_mapped['global_x'], t_mapped['global_y'], color=colors, s=3,
    edgecolor='k', alpha=1.0, linewidths=0.25,
    zorder=randint(1, high=len(t_celltypes.keys())))

scat.grid(False)
scat.set_xlabel('global [x] coordinate')
scat.set_ylabel('global [y] coorinate')

mpl.colorbar.ColorbarBase(map, cmap=cmap, norm=norm, orientation='vertical')

plt.savefig(os.path.join(save_dir, 'global_coordinates.pdf'))
plt.close()

# plot local x and y coordinates for tile pseudocolored according to their
# kernel density estimate
tile = 76
tile_data = t_mapped[t_mapped['tile'] == tile]

plt.figure(figsize=(8, 8))
for celltype in t_celltypes_order:

    celltype_data = tile_data[tile_data['cell_type'] == celltype]

    if celltype == 'GL261':

        plt.scatter(
            tile_data['local_x'], tile_data['local_y'],
            color=t_celltypes[celltype], s=100, edgecolor=None, alpha=1.0,
            linewidths=None, zorder=1)
    else:
        plt.scatter(
            celltype_data['local_x'], celltype_data['local_y'],
            color=t_celltypes[celltype], s=100, edgecolor='k', alpha=1.0,
            linewidths=0.5, zorder=randint(2, high=len(t_celltypes.keys())))

plt.grid(False)
plt.xlabel('local [x] coordinate')
plt.ylabel('local [y] coorinate')
plt.savefig(
    os.path.join(save_dir, 'tile_' + str(tile) + '_local_coordinates.pdf'))
plt.close()

# plot BIP spatial map
plt.figure(figsize=(8, 8))
for celltype in t_mapped['cell_type'].unique():
    if celltype == 'GL261':

        x = t_mapped['global_x'][t_mapped['cell_type'] == celltype]
        y = t_mapped['global_y'][t_mapped['cell_type'] == celltype]

        plt.scatter(x, y, c=t_celltypes[celltype], s=0, edgecolor=None,
                    alpha=1.0, linewidths=None, zorder=1)

    elif celltype in ['CD11b+', 'CD4+', 'CD8a+', 'CD68+',
                      'B220+', 'Foxp3+CD4+']:
        x = t_mapped['global_x'][t_mapped['cell_type'] == celltype]
        y = t_mapped['global_y'][t_mapped['cell_type'] == celltype]

        plt.scatter(x, y, c=t_celltypes[celltype], s=10, edgecolor='k',
                    alpha=1.0, linewidths=0.5,
                    zorder=randint(2, high=len(t_celltypes.keys())-1))

    elif celltype in ['B220+CD8a+']:
        x = t_mapped['global_x'][t_mapped['cell_type'] == celltype]
        y = t_mapped['global_y'][t_mapped['cell_type'] == celltype]

        plt.scatter(x, y, c=t_celltypes[celltype], s=50, edgecolor='k',
                    alpha=1.0, linewidths=0.5,
                    zorder=randint(
                        len(t_celltypes.keys())-1,
                        high=len(t_celltypes.keys())))
plt.grid(False)
plt.savefig(
    os.path.join(save_dir, 'cell_type_global_coordinates.pdf'))
plt.close()

y_range = [t_mapped['global_y'].min(), t_mapped['global_y'].max()]
x_range = [t_mapped['global_x'].min(), t_mapped['global_x'].max()]
hist_range = [x_range, y_range]
hists = {}
num_bins = 30


def correlation(hist_a, hist_b):
    raw = (hist_a * hist_b).sum()
    scale = ((hist_a * hist_a).sum()*(hist_b * hist_b).sum())**0.5
    return (raw/scale)


# get B220/CD8 and CD8 superimposed KDE plots
fig, ax = plt.subplots(figsize=(8, 8), dpi=800, facecolor='w', edgecolor='k')

ax.scatter(
    t_mapped['global_x'][t_mapped['cell_type'] == 'GL261'],
    t_mapped['global_y'][t_mapped['cell_type'] == 'GL261'],
    color=colors, s=4, edgecolor='gray', alpha=0.4, linewidths=0.0)

for celltype in t_mapped['cell_type'].unique():
    if celltype in ['CD8a+', 'B220+CD8a+']:

        if celltype == 'CD8a+':
            cmap = 'Blues'
        elif celltype == 'B220+CD8a+':
            cmap = 'Oranges'

        x = t_mapped['global_x'][t_mapped['cell_type'] == celltype]
        y = t_mapped['global_y'][t_mapped['cell_type'] == celltype]

        hist = np.histogram2d(
            x, y, bins=num_bins,
            range=[x_range, y_range])[0]

        hists[celltype] = hist

        sns.kdeplot(
            x, y, shade=False, n_levels=4, cmap=cmap,
            linewidths=5.0, cbar=False, ax=ax)

        plt.grid(False)

plt.savefig(
    os.path.join(save_dir, 'overlay_KDE.pdf'))
plt.close()

# get individual KDE plots
for celltype in t_mapped['cell_type'].unique():
    plt.figure(
        num=None, figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
    plt.scatter(
        t_mapped['global_x'][t_mapped['cell_type'] == 'GL261'],
        t_mapped['global_y'][t_mapped['cell_type'] == 'GL261'],
        color=colors, s=3, edgecolor='k', alpha=0.25, linewidths=0.25)

    x = t_mapped['global_x'][t_mapped['cell_type'] == celltype]
    y = t_mapped['global_y'][t_mapped['cell_type'] == celltype]

    hist = np.histogram2d(x, y, bins=num_bins, range=[x_range, y_range])[0]
    hists[celltype] = hist

    sns.kdeplot(
        x, y, shade=False, shade_lowest=False, cmap='Blues', linewidths=5.0,
        cbar=False)  # scalar

    plt.grid(False)
    plt.savefig(
        os.path.join(save_dir, celltype + '_KDE.pdf'))
    plt.close()

# get correlation clustermap
c = pd.DataFrame(index=hists.keys(), columns=hists.keys(), dtype=float)

for k1, v1 in hists.items():
    for k2, v2 in hists.items():
        c.loc[k1, k2] = correlation(v1, v2)

sns.clustermap(c, cmap='Blues', annot=True)
plt.savefig(
    os.path.join(save_dir, 'correlation_clustermap.pdf'))
plt.close()

# compute percentages without tumor cells included
wot_bips = bips.copy().iloc[1:, :].reset_index(drop=True)
wot_bips['percent'] = wot_bips['count']/wot_bips['count'].sum()*100

wot_to_map = wot_bips[list(channel_biases.keys())].head(15)

colors = plt.cm.tab20c((4./3*np.arange(20*3/4)).astype(int))

wot_celltypes = {'CD11b+': colors[0], 'CD4+': colors[1], 'CD8a+': colors[2],
                 'B220+': colors[3], 'CD68+': colors[4], 'Ly6C+': colors[5],
                 'B220+CD8a+': colors[6], 'CD49b+Ly6C+': colors[7],
                 'Ki67+CD4+': colors[8], 'Foxp3+CD4+': colors[9],
                 'Ki67+B220+': colors[10], 'CD49b+': colors[11],
                 'DPT': colors[12], 'Ki67+CD8a+': colors[13],
                 'CD4+CD8a+': colors[14]}

wot_celltypes_order = ['CD11b+', 'CD4+', 'CD8a+', 'CD68+', 'B220+',
                       'Foxp3+CD4+', 'B220+CD8a+', 'CD4+CD8a+', 'Ly6C+',
                       'CD49b+Ly6C+', 'Ki67+CD4+',
                       'Ki67+B220+', 'CD49b+', 'DPT',
                       'Ki67+CD8a+']

wot_celltypes = OrderedDict(
    sorted(
        wot_celltypes.items(), key=lambda x: wot_celltypes_order.index(x[0])))

wot_to_map['cell_type'] = list(wot_celltypes.keys())
wot_mapped = Boo_data.merge(
    wot_to_map, on=list(channel_biases.keys()), how='right')

counts = wot_mapped.groupby(['cell_type']).size().reset_index().rename(
    columns={0: 'count'}).sort_values(
        by='count', ascending=False, inplace=False).reset_index(drop=True)

# plot BIP percentage
wot_counts = wot_mapped.groupby('cell_type').size().reset_index().rename(
    columns={0: 'count'}).sort_values(
        by='count', ascending=False, inplace=False).reset_index(drop=True)

plt.bar(
    wot_counts['cell_type'], wot_counts['count'], color=colors)
plt.xticks(rotation=90)
plt.ylabel('count', size=15)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, 'test.pdf'))
plt.close()
