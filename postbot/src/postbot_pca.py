import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter
from natsort import natsorted
import collections
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime
import matplotlib.lines as mlines
import matplotlib as mpl
import math
import operator
import seaborn as sns
import pickle

# use this project path when copy/pasting code into ipython
project_path = '/Users/gjbaker/projects/gbm_immunosuppression'

# path to pickles from postbot analysis
postbot_pickle_dir = os.path.join(project_path, 'postbot', 'data',
                                  'logicle_e20', 'pickled_global_vars')

# map matplotlib color codes to the default seaborn palette
sns.set_color_codes()


# time script run
startTime = datetime.now()

# project path
project_path = '/Users/gjbaker/projects/gbm_immunosuppression'

# grab color_dict from postbot pickle directory
os.chdir(postbot_pickle_dir)
pi_color_dict = open('color_dict.pickle', 'rb')
color_dict = pickle.load(pi_color_dict)

pi_vector_classification = open('vector_classification.pickle', 'rb')
vector_classification = pickle.load(pi_vector_classification)

vector_classification['unspecified'] = {
    'lineage': 'other', 'signature': []}

# make lineage dict
lineage_dict = {}
for key1, value1 in vector_classification.items():
    value1 = collections.OrderedDict(
        sorted(value1.items()))
    for key2, value2 in value1.items():
        if key2 == 'lineage':
            lineage_dict[key1] = value2

class_lut = {'B': '#F87660', 'CD4posT': '#DE8C00',
             'CD8aposT': '#7CAE00', 'DC': '#B79F00', 'NK': '#00B4F0',
             'PMN': '#F564E3', 'Mono': '#619CFF', 'Mac': '#C77CFF',
             'DPT': '#FF64B0', 'ISPT': '#00BFC4', 'DNT': '#00BA38',
             'Precursor': '#00C08B', 'Non-immune': 'grey',
             'unspecified': 'black'}

# generate data folder for PCA analysis
pca_savedir = os.path.join(
    project_path, 'postbot', 'data', 'logicle_e20', 'pca_data')
os.makedirs(pca_savedir)

# change directory to read overall.csv (from postbot.py)
overall_data = os.path.join(
    project_path, 'postbot', 'data', 'logicle_e20', '00_output_FULL')
os.chdir(overall_data)
print('Reading overall.csv')
print()
overall = pd.read_csv('overall.csv')

marker_dict = {'7': 'o', '14': 's', '30': '*'}

status_dict = {'naive': [], 'gl261': []}

time_style = {7: 'o', 14: 's', 30: '*'}

size_style = {('naive', '*'): 7.5, ('gl261', '*'): 7.0,
              ('naive', 'o'): 6.5, ('gl261', 'o'): 6.0,
              ('naive', 's'): 6.0, ('gl261', 's'): 5.5}

# process 'overall' DataFrame as input into PCA
print('Running PCA analysis')
print()
pca1 = overall.groupby(
    ['time_point', 'tissue', 'status', 'replicate', 'cell_type']).size()
pca2 = pca1.unstack()
pca3 = pca2.replace(to_replace=np.nan, value=0.0)
pca4 = pca3.stack()
pca5 = pca4.groupby(
    level=['time_point', 'tissue', 'status', 'replicate']).apply(
        lambda x: (x / x.sum())*100)
pca6 = pca5.unstack()
pca7 = pca6.reset_index().sort_values(
    ['time_point', 'tissue', 'status', 'replicate'], inplace=False)

# store "tumor_burden" and "tissue' columns as a numpy array
sample_names = pca7[['time_point', 'tissue', 'status', 'replicate']]

# drop sample_names columns from pca7
pca_input = pca7.drop(['time_point', 'tissue', 'status', 'replicate'], axis=1)

# convert pca_input into a numpy array
pca_input = pca_input.as_matrix(columns=None)

# standardize the data
pca_input = StandardScaler().fit_transform(pca_input)

# compute mean vector of total dataset
mean_vec = np.mean(pca_input, axis=0)

# compute covariance matrix
cov_mat = (
    pca_input - mean_vec).T.dot(
        (pca_input - mean_vec)) / (pca_input.shape[0]-1)

# compute eigenvalues and eigenvectors from covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

# list eigenvalue/eigenvectors
eig_pairs = [
    (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(
        len(eig_val_cov))]

# sort eigenvectors in terms of their eigenvalues in decending order
eig_pairs.sort()
eig_pairs.reverse()

# compute explained variance
tot = sum(eig_val_cov)
var_exp = [(i / tot)*100 for i in sorted(eig_val_cov, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# generate scree plot save folder
screeplot_savdir = os.path.join(pca_savedir, 'scree_plot')
os.makedirs(screeplot_savdir)
os.chdir(screeplot_savdir)

# plot explained variance as a scree plot
print('Plotting scree plot')
print()
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(pca_input.shape[1]), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(pca_input.shape[1]), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(screeplot_savdir, 'scree_plot' + '.pdf'))
    plt.close('all')

# generate a list of the PCs of interest based in scree plot
pc_array = [1, 2, 3, 4, 5, 6]
pc_list = [(x-1) for x in pc_array]

# construct a d(number of measured parameters) x k(3)-dimensional
# eigenvector matrix, w
print('Constructing matrix_w')
print()
matrix_w = np.hstack((eig_pairs[0][1].reshape(pca_input.shape[1], 1),
                      eig_pairs[1][1].reshape(pca_input.shape[1], 1),
                      eig_pairs[2][1].reshape(pca_input.shape[1], 1),
                      eig_pairs[3][1].reshape(pca_input.shape[1], 1),
                      eig_pairs[4][1].reshape(pca_input.shape[1], 1),
                      eig_pairs[5][1].reshape(pca_input.shape[1], 1)))

# transform standardized pca_input onto new subspace (i.e. matrix_w)
# using equation: (y = W^T X x)
transformed = matrix_w.T.dot(pca_input.T)

# PC scores and standard deviations by sample
print('Calculating sample PC scores and standard deviations')
print()
PCs = transformed.T

# add PC column to sample_names DataFrame
sample_names_ext = pd.concat([sample_names]*matrix_w.shape[1])
sample_names_ext.reset_index(drop=True, inplace=True)
pc_array_df = pd.DataFrame(
    pc_array * int((len(sample_names_ext)/len(pc_array))))
pc_array_df.columns = ['PC']
pc_array_df.sort_values(by='PC', inplace=True)
pc_array_df.reset_index(drop=True, inplace=True)
sample_names_ext = pd.concat([sample_names_ext, pc_array_df], axis=1)
PC_values = pd.DataFrame(np.hstack(PCs.T))
PC_values.columns = ['PC_score']
sample_names_ext = pd.concat([sample_names_ext, PC_values], axis=1)
PC_score_means = sample_names_ext.groupby(
    ['time_point', 'tissue', 'status', 'PC']).mean().drop('replicate', axis=1)
PC_score_stds = sample_names_ext.groupby(
    ['time_point', 'tissue', 'status', 'PC']).std().drop('replicate', axis=1)
PC_data = pd.merge(
    PC_score_means, PC_score_stds, left_index=True, right_index=True,
    suffixes=('_mean', '_std'))
PC_data.reset_index(drop=False, inplace=True)

# look for PC correlations per sample
print('Calculating Spearman correlations on PCA scores')
print()
sample_names_ext['name'] = sample_names_ext[
    'time_point'].map(str) + '_' + sample_names_ext['tissue']
sample_names_ext['name'] = sample_names_ext[
    'name'].map(str) + '_' + sample_names_ext['status']
sample_names_ext.sort_values(
    ['time_point', 'tissue', 'status', 'PC', 'PC_score'], inplace=True)
sample_names_ext.reset_index(drop=True, inplace=True)
corr_input = sample_names_ext.drop(['time_point', 'tissue'], axis=1)
corr_input.sort_values(['PC', 'name'], inplace=True)

naive_list = []
gl261_list = []
for name, group in corr_input.groupby(['PC', 'status']):
    if all(group.status.values == 'naive'):
        naive = group.drop(['PC'], axis=1)
        naive = naive.pivot_table(
            index='replicate', columns='name', values=['PC_score'])
        naive.reset_index(drop=False, inplace=True)
        naive.set_index(['replicate'], inplace=True)
        naive.columns = naive.columns.droplevel(0)
        naive.columns = [s + '_PC' + (str(name).rsplit(',', 1)[0]).strip(
            '(') for s in naive.columns]
        naive_list.append(naive)
    elif all(group.status.values == 'gl261'):
        gl261 = group.drop(['PC'], axis=1)
        gl261 = gl261.pivot_table(
            index='replicate', columns='name', values=['PC_score'])
        gl261.reset_index(drop=False, inplace=True)
        gl261.set_index(['replicate'], inplace=True)
        gl261.columns = gl261.columns.droplevel(0)
        gl261.columns = [s + '_PC' + (str(name).rsplit(',', 1)[0]).strip(
            '(') for s in gl261.columns]
        gl261_list.append(gl261)
naive_df = pd.concat(naive_list, axis=1)
gl261_df = pd.concat(gl261_list, axis=1)

PC_dict = {}
corr_dict = {}
sig_corr_dict = {}
names = ['n', 'g']
cutoff = 0.7
for (status, name) in zip([naive_df, gl261_df], names):
    for timepoint in sample_names_ext['time_point'].unique():
        cols = [c for c in list(status) if c.startswith(str(timepoint))]
        spec_df = '%s%s' % (str(name), str(timepoint))
        PC_dict[spec_df] = status[cols]
        # PC_dict[spec_df].columns = ['_'.join(x.split('_', 3)[1:4:2])
        #                             for x in PC_dict[spec_df]]
        corr_dict[spec_df] = PC_dict[spec_df].corr(
            method='spearman', min_periods=1)
        corr_dict[spec_df].values[
            np.tril_indices_from(corr_dict[spec_df])] = np.nan
        indices = np.where(abs(corr_dict[spec_df]) > cutoff)
        sig_corr_dict[spec_df] = [(corr_dict[spec_df].index[x],
                                  corr_dict[spec_df].columns[y]) for
                                  x, y in zip(*indices)]
sig_corr_dict_order = ('n7', 'n14', 'n30', 'g7', 'g14', 'g30')
sig_corr_dict_list = sorted(
    sig_corr_dict.items(), key=lambda pair: sig_corr_dict_order.index(pair[0]))
print('Pearson correlations greater than ' + str(cutoff) + ':')
print()
print(sig_corr_dict_list)
print()

# get sample PCA ranks for naive and gl261 datasets
print('Calculating sample PCA ranks')
print()
naive_PC_ranks_list = []
naive_PC_ranks = naive_df.rank()
for column in naive_PC_ranks:
    sample = naive_PC_ranks[column].reindex([naive_PC_ranks[column].values])
    sample = sample.to_frame()
    sample_name = list(sample.columns)
    sample_name = str(sample_name)[2:-2]
    sample[sample_name] = [i[0] for i in sample.index]
    sample.reset_index(drop=True, inplace=True)
    naive_PC_ranks_list.append(sample)
naive_PC_ranks = pd.concat(naive_PC_ranks_list, axis=1)

gl261_PC_ranks_list = []
gl261_PC_ranks = gl261_df.rank()
for column in gl261_PC_ranks:
    sample = gl261_PC_ranks[column].reindex([gl261_PC_ranks[column].values])
    sample = sample.sort_values()
    sample = sample.to_frame()
    sample_name = list(sample.columns)
    sample_name = str(sample_name)[2:-2]
    sample[sample_name] = [i[0] for i in sample.index]
    sample.reset_index(drop=True, inplace=True)
    gl261_PC_ranks_list.append(sample)
gl261_PC_ranks = pd.concat(gl261_PC_ranks_list, axis=1)

# isolate PCA ranks for samples showing significant correlation in PC space
print('Isolating sample PCA ranks with significant Spearman correlation')
print()
sig_list = []
for key, value in sig_corr_dict.items():
    for pair in value:
        for s in pair:
            sig_list.append(s)
sig_list = list(set(sig_list))
naive_rank_slice = [x for x in naive_PC_ranks.columns if x in sig_list]
gl261_rank_slice = [x for x in gl261_PC_ranks.columns if x in sig_list]
sig_naive_ranks = naive_PC_ranks[naive_rank_slice]
sig_gl261_ranks = gl261_PC_ranks[gl261_rank_slice]

n_ranks = {}
g_ranks = {}
for key, value in sig_corr_dict.items():
    for pair in value:
        for sample in pair:
            if 'naive' in sample:
                n_ranks[pair] = [sig_naive_ranks[pair[0]].values.astype(int),
                                 sig_naive_ranks[pair[1]].values.astype(int)]
            elif 'gl261' in sample:
                g_ranks[pair] = [sig_gl261_ranks[pair[0]].values.astype(int),
                                 sig_gl261_ranks[pair[1]].values.astype(int)]
n_ranks_df = pd.DataFrame.from_dict(n_ranks, orient='index')
n_ranks_df.columns = ['x', 'y']

g_ranks_df = pd.DataFrame.from_dict(g_ranks, orient='index')
g_ranks_df.columns = ['x', 'y']

# find timepoint-specific antipodes
print('Finding gl261 timepoint-specific antipodes')
print()
g_antipodes = {}
g_tp_list = []
g_index = {}
for col in gl261_PC_ranks.columns:
    g_tp_list.append(col.rsplit('_', 3)[0])
    g_tp_set = set(g_tp_list)
    for tp in g_tp_set:
        tp_slice = [x for x in gl261_PC_ranks.columns if x.startswith(tp)]
        g_anti_input = gl261_PC_ranks[tp_slice].T
        g_index[tp] = g_anti_input
        g_tupe_list = []
        for (j, k) in zip(g_anti_input[0], g_anti_input[7]):
            g_tupe_list.append((j, k))
        g_antipodes[tp] = Counter(tuple(sorted(tup)) for tup in g_tupe_list)
g_antipodes = collections.OrderedDict(natsorted(g_antipodes.items()))

print('Finding naive timepoint-specific antipodes')
print()
n_antipodes = {}
n_tp_list = []
n_index = {}
for col in naive_PC_ranks.columns:
    n_tp_list.append(col.rsplit('_', 3)[0])
    n_tp_set = set(n_tp_list)
    for tp in n_tp_set:
        tp_slice = [x for x in naive_PC_ranks.columns if x.startswith(tp)]
        n_anti_input = naive_PC_ranks[tp_slice].T
        n_index[tp] = n_anti_input
        n_tupe_list = []
        for (j, k) in zip(n_anti_input[0], n_anti_input[7]):
            n_tupe_list.append((j, k))
        n_antipodes[tp] = Counter(tuple(sorted(tup)) for tup in n_tupe_list)
n_antipodes = collections.OrderedDict(natsorted(n_antipodes.items()))

# plot heatmaps of antipodal combinations per timepoint
print('Plotting gl261 timepoint-specific antipodes')
print()
antipode_savdir = os.path.join(pca_savedir, 'antipode_plots')
os.makedirs(antipode_savdir)
os.chdir(antipode_savdir)

g_antipodes_df = pd.DataFrame.from_dict(g_antipodes)
g_antipodes_df = g_antipodes_df.fillna(value=0)
g_antipodes_df.reset_index(inplace=True)
g_antipodes_df.rename(
    columns={'level_0': 'x', 'level_1': 'y'}, inplace=True)
for col in g_antipodes_df.columns:
    if col in ['x', 'y']:
        pass
    else:
        data = g_antipodes_df[['x', 'y', col]]
        data = data.pivot_table(index=data['x'], columns=data['y'])
        data = data.fillna(value=0)
        col_index = data.columns.droplevel(0)
        data.columns = col_index
        col_list = list(data.columns)
        col_list = ['m' + str(int(x)) for x in col_list]
        data.columns = col_list
        row_list = list(data.index)
        row_list = ['m' + str(int(x)) for x in row_list]
        data.index = row_list
        ax = sns.heatmap(data, square=True, annot=True, vmin=0, vmax=7)
        plt.savefig(os.path.join(
            antipode_savdir, 'gl261_timepoint_' + str(col) + '.pdf'))
        plt.close('all')

print('Plotting naive timepoint-specific antipodes')
print()
n_antipodes_df = pd.DataFrame.from_dict(n_antipodes)
n_antipodes_df = n_antipodes_df.fillna(value=0)
n_antipodes_df.reset_index(inplace=True)
n_antipodes_df.rename(columns={'level_0': 'x', 'level_1': 'y'}, inplace=True)
for col in n_antipodes_df.columns:
    if col in ['x', 'y']:
        pass
    else:
        data = n_antipodes_df[['x', 'y', col]]
        data = data.pivot_table(index=data['x'], columns=data['y'])
        data = data.fillna(value=0)
        col_index = data.columns.droplevel(0)
        data.columns = col_index
        col_list = list(data.columns)
        col_list = ['m' + str(int(x)) for x in col_list]
        data.columns = col_list
        row_list = list(data.index)
        row_list = ['m' + str(int(x)) for x in row_list]
        data.index = row_list
        ax = sns.heatmap(data, square=True, annot=True, vmin=0, vmax=7)
        plt.savefig(os.path.join(
            antipode_savdir, 'naive_timepoint_' + str(col) + '.pdf'))
        plt.close('all')

# get a list of all the Boolean vector names
cell_type_list = pca7.iloc[:, 4:].columns.get_values().tolist()

# get a list of all the sample names
sample_list = pca7.iloc[:, :4].values.tolist()

# assign each Boolean vector to its major hematopoietic lineage

# generate loadings plot save folder
loadingsplot_savdir = os.path.join(pca_savedir, 'loadings_plots')
os.makedirs(loadingsplot_savdir)
os.chdir(loadingsplot_savdir)

# # plot PC1 vs. PC2 vs. PC3 loadings plot
# for i, j in enumerate(list(itertools.combinations(pc_list, 3))):
#     print('Saving ' + 'PC' + str(j[0]+1) + ' vs ' + 'PC' + str(j[1]+1)
#           + ' vs ' + 'PC' + str(j[2]+1) + ' 3D loadings plot')
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.rcParams['legend.fontsize'] = 12
#     for z, (cell_type, value) in enumerate(zip(cell_type_list, matrix_w)):
#         if cell_type in mature_myeloid_list:
#             marker = 'o'
#             markersize = 6.0
#             alpha = 1.0
#             markerfacecolor = 'k'
#             markeredgecolor = 'None'
#             markeredgewidth = 0.75
#         elif cell_type in immature_myeloid_list:
#             marker = 'o'
#             markersize = 6.0
#             alpha = 1.0
#             markerfacecolor = 'None'
#             markeredgecolor = 'k'
#             markeredgewidth = 0.75
#         elif cell_type in mature_lymphoid_list:
#             marker = 'o'
#             markersize = 6
#             alpha = 1.0
#             markerfacecolor = 'deeppink'
#             markeredgecolor = 'None'
#             markeredgewidth = 0.75
#         elif cell_type in immature_lymphoid_list:
#             marker = 'o'
#             markersize = 6.0
#             alpha = 1.0
#             markerfacecolor = 'None'
#             markeredgecolor = 'deeppink'
#             markeredgewidth = 0.75
#         elif cell_type in other_list:
#             marker = 'o'
#             markersize = 6.5
#             alpha = 1.0
#             markerfacecolor = 'orange'
#             markeredgecolor = 'gray'
#             markeredgewidth = 0.3
#         ax.plot(np.asarray([value[j[0]]]), np.asarray([value[j[1]]]),
#                 np.asarray([value[j[2]]]),
#                 marker=marker, markersize=markersize,
#                 markerfacecolor=markerfacecolor,
#                 markeredgecolor=markeredgecolor,
#                 markeredgewidth=markeredgewidth,
#                 alpha=alpha, label=cell_type)
#     ax.set_xlabel('PC%d' % (j[0]+1))
#     ax.set_ylabel('PC%d' % (j[1]+1))
#     ax.set_zlabel('PC%d' % (j[2]+1))
#     plt.title(
#         'Transformed variables with class labels from stepwise computation')
#     plt.savefig(os.path.join(
#         loadingsplot_savdir, 'PC' + str(j[0]+1) + '_vs_' + 'PC' + str(j[1]+1)
#         + '_vs_' + 'PC' + str(j[2]+1) + '.pdf'))
#     # ax.legend(loc='upper right')
#     # for angle in range(0, 360):
#     #     ax.view_init(30, angle)
#     #     plt.draw()
#     #     plt.pause(.001)
#     plt.close('all')
# print()

# plot all 2D loadings plot combinations
major_classes = []
for longer in cell_type_list:
    if any(
      substring in longer for substring in list(class_lut.keys())) is True:
        x = ''.join(
            [sub for sub in list(class_lut.keys()) if sub in longer])
        major_classes.append(x)

for i, cell in enumerate(major_classes):
    if cell == 'BNK':
        major_classes[i] = 'NK'
    elif cell == 'BCD8aposT':
        major_classes[i] = 'CD8aposT'
    elif cell == 'BMac':
        major_classes[i] = 'Mac'
    elif cell == 'BISPT':
        major_classes[i] = 'ISPT'

sns.set(style='whitegrid')
for i, j in enumerate(list(itertools.combinations(pc_list, 2))):
    tup = (i, j)
    x_index = tup[1][0]
    y_index = tup[1][1]

    print('Saving ' + 'PC' + str(x_index+1) + ' vs. ' +
          'PC' + str(y_index+1) + ' 2D loadings plot.')

    x_data = {}
    y_data = {}
    for z, (cell_type, clss, value) in enumerate(
      zip(cell_type_list, major_classes, matrix_w)):

        if cell_type != 'unspecified':

            cell_type_update = cell_type.replace('neg', '$^-$').replace(
                'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')

            marker = 'o'
            markersize = 8.0
            alpha = 1.0
            markerfacecolor = class_lut[clss]
            markeredgecolor = 'None'
            markeredgewidth = 1.5

            x_data[cell_type] = value[x_index]
            y_data[cell_type] = value[y_index]

            plt.plot(np.asarray(value[x_index]), np.asarray(value[y_index]),
                     marker=marker, markersize=markersize,
                     markerfacecolor=markerfacecolor,
                     markeredgecolor=markeredgecolor,
                     markeredgewidth=markeredgewidth,
                     alpha=alpha, label=cell_type)

            plt.annotate(cell_type_update, xy=(np.asarray(value[x_index]),
                         np.asarray(value[y_index])), xytext=(2, 2), size=7,
                         textcoords='offset points', ha='left', va='bottom',
                         weight='bold')

    plt.xlabel('PC%d' % (x_index+1), fontweight='bold')
    plt.ylabel('PC%d' % (y_index+1), fontweight='bold')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(
        axis='both', which='minor', length=3.0,
        color='k', direction='in', top='on',
        bottom='on', right='on', left='on')

    ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.axhline(0, ls='dashed', color='gray', linewidth=0.75)
    plt.axvline(0, ls='dashed', color='gray', linewidth=0.75)

    plt.xlim([-0.3, 0.3])
    plt.ylim([-0.4, 0.4])

    plt.savefig(os.path.join(
        loadingsplot_savdir, 'PC' + str(x_index+1) + '_v_' + 'PC' +
                             str(y_index+1) + '.pdf'))
    plt.close('all')
print()

# generate PC score bar graph save folder
scorebarplot_savdir = os.path.join(pca_savedir, 'score_bar_plot')
os.makedirs(scorebarplot_savdir)
os.chdir(scorebarplot_savdir)

# plot PC score bar graph
print('Plotting PC score bar graph')
print()
N = 30
ind = np.arange(N)
width = 0.15
opacity = 0.4
error_config = {'ecolor': '0.3'}
fig, ax = plt.subplots()
color_map = {'PC1': 'r', 'PC2': 'darkcyan', 'PC3': 'g', 'PC4': 'm', 'PC5': 'y'}

means_PC = []
stds_PC = []
PC_color = []
for i, x in PC_data.groupby(['PC']):
    x = x.sort_values(['time_point', 'tissue', 'status'],
                      ascending=[True, True, False])
    means_PC.append(x.loc[:, 'PC_score_mean'])
    stds_PC.append(x.loc[:, 'PC_score_std'])
    PC_color.append('PC' + str(int(x.PC.unique())))

rects = []
for i in range(len(means_PC)):
    color = color_map.get(PC_color[i])
    rects.append(ax.bar(ind + width*i, means_PC[i], width, color=color,
                 alpha=opacity, yerr=stds_PC[i], error_kw=error_config))

ax.set_ylabel('scores')

ax.set_ylim(-10.0, 10.0)

major_ticks = np.arange(-10.0, 10.0, 2.5)
ax.set_yticks(major_ticks)

minor_ticks = np.arange(-10.0, 10.0, 0.5)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='minor', alpha=0.3)
ax.grid(which='major', alpha=0.7)

ax.set_title('scores by sample and principle component')
ax.set_xticks(ind + (width*2))
ax.set_xticklabels(('n_bld_7', 'g_bld_7', 'n_bm_7', 'g_bm_7', 'n_cln_7',
                    'g_cln_7', 'n_spl_7', 'g_spl_7', 'n_thy_7', 'g_thy_7',
                    'n_bld_14', 'g_bld_14', 'n_bm_14', 'g_bm_14', 'n_cln_14',
                    'g_cln_14', 'n_spl_14', 'g_spl_14', 'n_thy_14', 'g_thy_14',
                    'n_bld_30', 'g_bld_30', 'n_bm_30', 'g_bm_30', 'n_cln_30',
                    'g_cln_30', 'n_spl_30', 'g_spl_30', 'n_thy_30',
                    'g_thy_30'), rotation=90, ha='center')

plt.axhline(0, linestyle='dashed', linewidth=0.5, color='gray')

ax.legend(
    (rects[0], rects[1], rects[2], rects[3], rects[4]),
    ('PC1', 'PC2', 'PC3', 'PC4', 'PC5'))

plt.savefig(os.path.join(scorebarplot_savdir, 'PC_score_bar_plot' + '.pdf'))
plt.close('all')

# generate score plot save folder
scoreplot_savdir = os.path.join(pca_savedir, 'score_plots')
os.makedirs(scoreplot_savdir)
os.chdir(scoreplot_savdir)

# plot all 3D score plot combinations
for i, j in enumerate(list(itertools.combinations(pc_list, 3))):

    print('Saving ' + 'PC' + str(j[0]+1) + ' vs ' + 'PC' + str(j[1]+1)
          + ' vs ' + 'PC' + str(j[2]+1) + ' 3D score plot')

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')

    plt.rcParams['legend.fontsize'] = 12

    for z, (sample, value) in enumerate(zip(sample_list, PCs)):

            if 'naive' == sample[2]:
                marker = time_style[sample[0]]
                markersize = size_style[(sample[2], marker)]
                markerfacecolor = color_dict[sample[1]]
                markeredgecolor = 'None'
                markeredgewidth = 0.75

            elif 'gl261' == sample[2]:
                marker = time_style[sample[0]]
                markersize = size_style[(sample[2], marker)]
                markerfacecolor = 'None'
                markeredgecolor = color_dict[sample[1]]
                markeredgewidth = 0.75

            plt.plot(np.asarray([value[j[0]]]), np.asarray([value[j[1]]]),
                     np.asarray([value[j[2]]]),
                     marker=marker, markersize=markersize,
                     markerfacecolor=markerfacecolor,
                     markeredgecolor=markeredgecolor,
                     markeredgewidth=markeredgewidth,
                     alpha=alpha, label=sample)

    ax.set_xlabel('PC%d' % (j[0]+1))
    ax.set_ylabel('PC%d' % (j[1]+1))
    ax.set_zlabel('PC%d' % (j[2]+1))

    ax = plt.gca()

    ax.set_facecolor('white')

    ax.w_xaxis.set_pane_color((.98, .98, .98, .98))
    ax.w_yaxis.set_pane_color((.98, .98, .98, .98))
    ax.w_zaxis.set_pane_color((.98, .98, .98, .98))

    mpl.rcParams['grid.color'] = 'lightgrey'

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 18
    ax.zaxis.labelpad = 10

    xlbl = ax.get_xlabel()
    ax.set_xlabel(xlbl, fontweight='bold')

    ylbl = ax.get_ylabel()
    ax.set_ylabel(ylbl, fontweight='bold')

    zlbl = ax.get_zlabel()
    ax.set_zlabel(zlbl, fontweight='bold')

    legend_list1 = []
    for key, value in color_dict.items():

        line = mlines.Line2D(
            [], [], color=value, linestyle='-', linewidth=5.0, label=key)
        legend_list1.append(line)

    legend1_text_properties = {'size': 8, 'weight': 'bold'}
    legend1 = plt.legend(handles=legend_list1, prop=legend1_text_properties,
                         bbox_to_anchor=(1.145, 1.02))
    ax.add_artist(legend1)

    legend2_list = []
    for key2, value2 in status_dict.items():

        if key2 == 'naive':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['7'], markersize=8,
                markerfacecolor='k', markeredgecolor='None',
                markeredgewidth=0.75, label=key2 + ', ' + '7dpi')
            legend2_list.append(line)

        elif key2 == 'gl261':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['7'], markersize=7,
                markerfacecolor='None', markeredgecolor='k',
                markeredgewidth=0.75, label=key2 + ', ' + '7dpi')
            legend2_list.append(line)

    legend2_text_properties = {'size': 8, 'weight': 'bold'}
    legend2 = plt.legend(handles=legend2_list, prop=legend2_text_properties,
                         loc=3, bbox_to_anchor=(0.985, 0.795))
    ax.add_artist(legend2)

    legend3_list = []
    for key2, value2 in status_dict.items():

        if key2 == 'naive':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['14'], markersize=8,
                markerfacecolor='k', markeredgecolor='None',
                markeredgewidth=0.75, label=key2 + ', ' + '14dpi')
            legend3_list.append(line)

        elif key2 == 'gl261':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['14'], markersize=7,
                markerfacecolor='None', markeredgecolor='k',
                markeredgewidth=0.73, label=key2 + ', ' + '14dpi')
            legend3_list.append(line)

    legend3_text_properties = {'size': 8, 'weight': 'bold'}
    legend3 = plt.legend(handles=legend3_list, prop=legend3_text_properties,
                         loc=3, bbox_to_anchor=(0.985, 0.74))
    ax.add_artist(legend3)

    legend4_list = []
    for key2, value2 in status_dict.items():

        if key2 == 'naive':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['30'], markersize=8,
                markerfacecolor='k', markeredgecolor='None',
                markeredgewidth=0.75, label=key2 + ', ' + '30dpi')
            legend4_list.append(line)

        elif key2 == 'gl261':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['30'], markersize=7,
                markerfacecolor='None', markeredgecolor='k',
                markeredgewidth=0.75, label=key2 + ', ' + '30dpi')
            legend4_list.append(line)

    legend4_text_properties = {'size': 8, 'weight': 'bold'}
    legend4 = plt.legend(handles=legend4_list, prop=legend4_text_properties,
                         loc=3, bbox_to_anchor=(0.985, 0.685))

    plt.savefig(os.path.join(
        scoreplot_savdir, 'PC' + str(j[0]+1) + '_vs_' + 'PC' + str(j[1]+1)
        + '_vs_' + 'PC' + str(j[2]+1) + '.pdf'), bbox_extra_artists=(legend4,),
        bbox_inches='tight')

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

    plt.close('all')
print()


# plot all 2D score plot combinations
sns.set(style='whitegrid')
for i, j in enumerate(list(itertools.combinations(pc_list, 2))):

    print(
        'Saving ' + 'PC' + str(j[0]+1) + ' vs ' +
        'PC' + str(j[1]+1) + ' 2D score plot')

    tp_list = []
    status_list = []

    x_scores = {}
    y_scores = {}
    for e, (sample, value1) in enumerate(zip(sample_list, PCs)):

        if 'naive' == sample[2]:
            marker = time_style[sample[0]]
            markersize = size_style[(sample[2], marker)]
            markerfacecolor = color_dict[sample[1]]
            markeredgecolor = 'None'
            markeredgewidth = 0.75

        elif 'gl261' == sample[2]:
            marker = time_style[sample[0]]
            markersize = size_style[(sample[2], marker)]
            markerfacecolor = 'None'
            markeredgecolor = color_dict[sample[1]]
            markeredgewidth = 0.75

        plt.plot(np.asarray([value1[j[0]]]), np.asarray([value1[j[1]]]),
                 marker=marker, markersize=markersize,
                 markerfacecolor=markerfacecolor,
                 markeredgecolor=markeredgecolor,
                 markeredgewidth=markeredgewidth,
                 alpha=alpha, label=sample, zorder=4)

        plt.annotate(sample[3], xy=(np.asarray([value1[j[0]]]),
                     np.asarray([value1[j[1]]])),
                     xytext=(2, 2), size=7, textcoords='offset points',
                     ha='left', va='bottom', weight='bold')

        tp_list.append(sample[0])
        status_list.append(sample[2])

        x_scores[e] = value1[j[0]]
        y_scores[e] = value1[j[1]]

        x_score_min = min(x_scores.values())
        x_score_max = max(x_scores.values())

        y_score_min = min(y_scores.values())
        y_score_max = max(y_scores.values())

    x_loadings = {}
    y_loadings = {}
    for e, (cell_type, clss, value2) in enumerate(
      zip(cell_type_list, major_classes, matrix_w)):

        marker = '|'
        markersize = 5.0
        alpha = 1.0
        markerfacecolor = 'None'
        markeredgecolor = class_lut[clss]
        markeredgewidth = 1.0

        x_loadings[cell_type] = value2[j[0]]
        y_loadings[cell_type] = value2[j[1]]

        x_loading_min = min(x_loadings.values())
        x_loading_max = max(x_loadings.values())

        y_loading_min = min(y_loadings.values())
        y_loading_max = max(y_loadings.values())

        x_loadings_range = x_loading_max + abs(x_loading_min)
        y_loadings_range = y_loading_max + abs(y_loading_min)

        x_loadings_conversion_dict = {}
        for key, val in x_loadings.items():
            percent = ((-val) + x_loading_max)/(-x_loadings_range) + 1

            x_loadings_conversion_dict[val] = percent

        y_loadings_conversion_dict = {}
        for key, val in y_loadings.items():
            percent = ((-val) + y_loading_max)/(-y_loadings_range) + 1

            y_loadings_conversion_dict[val] = percent

        # plt.plot(x_score_max - (
        #     (x_score_max * 2) *
        #     (1 - x_loadings_conversion_dict[value2[j[0]]])), 0,
        #     marker=marker, markersize=markersize,
        #     markerfacecolor=markerfacecolor,
        #     markeredgecolor=markeredgecolor,
        #     markeredgewidth=markeredgewidth,
        #     alpha=alpha, label=cell_type, zorder=3)

        marker = '_'
        markersize = 5.0
        alpha = 1.0
        markerfacecolor = class_lut[clss]
        markeredgecolor = class_lut[clss]
        markeredgewidth = 1.0

        # plt.plot(0, y_score_max - (
        #     (y_score_max * 2) *
        #     (1 - y_loadings_conversion_dict[value2[j[1]]])),
        #     marker=marker,
        #     markersize=markersize, markerfacecolor=markerfacecolor,
        #     markeredgecolor=markeredgecolor,
        #     markeredgewidth=markeredgewidth,
        #     alpha=alpha, label=cell_type, zorder=3)

    sorted_x = sorted(x_loadings.items(), key=operator.itemgetter(1))
    sorted_y = sorted(y_loadings.items(), key=operator.itemgetter(1))

    ax = plt.gca()

    x_lim = ax.get_xlim()
    x_lim = [abs(i) for i in list(x_lim)]
    x_lim = max(x_lim)

    y_lim = ax.get_ylim()
    y_lim = [abs(i) for i in list(y_lim)]
    y_lim = max(y_lim)

    plt.xlim([-x_lim, x_lim])
    plt.ylim([-y_lim, y_lim])

    plt.xlabel('PC%d' % (j[0]+1), fontweight='bold')
    plt.ylabel('PC%d' % (j[1]+1), fontweight='bold')

    plt.axhline(0, ls='dashed', color='gray', linewidth=0.75)
    plt.axvline(0, ls='dashed', color='gray', linewidth=0.75)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(
        axis='both', which='minor', length=3.0,
        color='k', direction='in', top='on',
        bottom='on', right='on', left='on')

    ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    legend_list1 = []
    for key, value in color_dict.items():

        line = mlines.Line2D(
            [], [], color=value, linestyle='-', linewidth=5.0, label=key)
        legend_list1.append(line)

    legend1_text_properties = {'size': 8, 'weight': 'bold'}
    legend1 = plt.legend(handles=legend_list1, prop=legend1_text_properties,
                         bbox_to_anchor=(1.145, 1.02))
    ax.add_artist(legend1)

    legend2_list = []
    for key2, value2 in status_dict.items():

        if key2 == 'naive':
            line = mlines.Line2D(
                [], [], color='dimgrey', linestyle='None',
                marker=marker_dict['7'], markersize=8,
                markerfacecolor='k', markeredgecolor='None',
                markeredgewidth=0.75, label=key2 + ', ' + '7dpi')
            legend2_list.append(line)

        elif key2 == 'gl261':
            line = mlines.Line2D(
                [], [], color='dimgrey', linestyle='None',
                marker=marker_dict['7'], markersize=7,
                markerfacecolor='None', markeredgecolor='k',
                markeredgewidth=0.75, label=key2 + ', ' + '7dpi')
            legend2_list.append(line)

    legend2_text_properties = {'size': 8, 'weight': 'bold'}
    legend2 = plt.legend(handles=legend2_list, prop=legend2_text_properties,
                         loc=3, bbox_to_anchor=(0.985, 0.695))
    ax.add_artist(legend2)

    legend3_list = []
    for key2, value2 in status_dict.items():

        if key2 == 'naive':
            line = mlines.Line2D(
                [], [], color='dimgrey', linestyle='None',
                marker=marker_dict['14'], markersize=8,
                markerfacecolor='k', markeredgecolor='None',
                markeredgewidth=0.75, label=key2 + ', ' + '14dpi')
            legend3_list.append(line)

        elif key2 == 'gl261':
            line = mlines.Line2D(
                [], [], color='dimgrey', linestyle='None',
                marker=marker_dict['14'], markersize=7,
                markerfacecolor='None', markeredgecolor='k',
                markeredgewidth=0.73, label=key2 + ', ' + '14dpi')
            legend3_list.append(line)

    legend3_text_properties = {'size': 8, 'weight': 'bold'}
    legend3 = plt.legend(handles=legend3_list, prop=legend3_text_properties,
                         loc=3, bbox_to_anchor=(0.985, 0.615))
    ax.add_artist(legend3)

    legend4_list = []
    for key2, value2 in status_dict.items():

        if key2 == 'naive':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['30'], markersize=8,
                markerfacecolor='k', markeredgecolor='None',
                markeredgewidth=0.75, label=key2 + ', ' + '30dpi')
            legend4_list.append(line)

        elif key2 == 'gl261':
            line = mlines.Line2D(
                [], [], color='k', linestyle='None',
                marker=marker_dict['30'], markersize=7,
                markerfacecolor='None', markeredgecolor='k',
                markeredgewidth=0.75, label=key2 + ', ' + '30dpi')
            legend4_list.append(line)

    legend4_text_properties = {'size': 8, 'weight': 'bold'}
    legend4 = plt.legend(handles=legend4_list, prop=legend4_text_properties,
                         loc=3, bbox_to_anchor=(0.985, 0.535))
    ax.add_artist(legend4)

    legend_list5 = []
    for key, value in class_lut.items():

        line = mlines.Line2D(
            [], [], color=value, linestyle='-', linewidth=5.0,
            label=key.replace('neg', '$^-$').replace('pos', '$^+$').replace(
                'CD8a', 'CD8' + u'\u03B1'))
        legend_list5.append(line)

    legend5_text_properties = {'size': 8, 'weight': 'bold'}
    legend5 = plt.legend(handles=legend_list5, prop=legend5_text_properties,
                         bbox_to_anchor=(1.0, .54))

    plt.savefig(os.path.join(
        scoreplot_savdir, 'PC' + str(j[0]+1) + '_vs_' +
        'PC' + str(j[1]+1) + '.pdf'),
        bbox_extra_artists=(legend1, legend2, legend3, legend4, legend5),
        bbox_inches='tight')
    plt.close('all')
print()

print('Postbot_pca analysis completed in ' + str(datetime.now() - startTime))
