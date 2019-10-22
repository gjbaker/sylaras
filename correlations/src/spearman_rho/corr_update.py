# RATIONALE for using Spearman's rho correlation
# Pearson's r assumes:
#  - continuous data
#  - linear relationship between the two variables
#  - no significant outliers
#  - normally distributed data

# Spearman's p assumes:
#  - at least ordinal data (i.e. able to be rank ordered,
#    < 15 points on the scale)
#  - a monotonic relationship between the two variables

# SET GLOBAL CONFIGURATIONS
# import pdb; pdb.set_trace()
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr
import seaborn as sns
from datetime import datetime
import matplotlib.gridspec as gridspec
from collections import defaultdict
import collections
from collections import OrderedDict
import networkx as nx
from shutil import copy2
import copy
import math
import re
import pickle
import shelve
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.stats import binom
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import itertools
from sklearn import metrics
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator
import pprint
from matplotlib.ticker import FuncFormatter


# set map matplotlib color codes to the default seaborn palette
sns.set_color_codes()


# allow for printing statements in bold
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# display adjustments
pd.set_option('display.width', None)
pd.options.display.max_rows = 500
pd.options.display.max_columns = 33

# script call error message
if len(sys.argv) != 2:
    print("Usage: correlations.py <path_to_project>")
    sys.exit()

# project path for script call
project_path = sys.argv[1]

# project path error message
if not os.path.exists(project_path):
    print('Project path does not exist')
    sys.exit()

# script run timer
startTime = datetime.now()

# use this project path when copy/pasting code into ipython
project_path = '/Users/gjbaker/projects/gbm_immunosuppression'

orthant_dir = os.path.join(project_path, 'postbot', 'data',
                           'logicle_e20', 'orthant')

# path to pickles from postbot analysis
orthant_pickle_dir = os.path.join(orthant_dir, 'pickled_global_vars')

# path to correlation analysis save directory
save_dir = os.path.join(project_path, 'correlations', 'data',
                        'spearman_correlation_analysis', 'orthant')

# path to correlation analysis pickle directory
if os.path.isdir(save_dir + '/pickled_global_vars') is True:
    correlation_pickle_dir = os.path.join(save_dir, 'pickled_global_vars')

else:

    correlation_pickle_dir = os.path.join(save_dir, 'pickled_global_vars')
    os.makedirs(correlation_pickle_dir)

# get required postbot pickles
os.chdir(orthant_pickle_dir)
pi_color_dict = open('color_dict.pickle', 'rb')
color_dict = pickle.load(pi_color_dict)

pi_vector_classification = open('vector_classification.pickle', 'rb')
vector_classification = pickle.load(pi_vector_classification)

# update vector_classification dictionary with unspecified celltype
vector_classification['unspecified'] = {
    'lineage': 'unspecified', 'signature': []}

vector_classification = collections.OrderedDict(
    sorted(vector_classification.items()))

# make lineage dict and lut
lineage_dict = {}
for key1, value1 in vector_classification.items():
    value1 = collections.OrderedDict(
        sorted(value1.items()))
    for key2, value2 in value1.items():
        if key2 == 'lineage':
            lineage_dict[key1] = value2

lineage_lut = {}
for key, value in lineage_dict.items():
    if value == 'lymphoid':
        lineage_lut[key] = 'g'
    elif value == 'myeloid':
        lineage_lut[key] = 'b'
    elif value == 'other':
        lineage_lut[key] = 'w'
    elif value == 'unspecified':
        lineage_lut[key] = 'w'

# make class lut
class_lut = {'B': '#F87660', 'CD4T': '#DE8C00',
             'CD8T': '#7CAE00', 'DC': '#B79F00', 'NK': '#00B4F0',
             'PMN': '#F564E3', 'Eo': 'orange', 'Mono': '#619CFF',
             'Mac': '#C77CFF', 'DPT': '#FF64B0', 'ISPT': '#00BFC4',
             'DNT': '#00BA38', 'Precursor': '#00C08B',
             'LTi': 'grey', 'unspecified': 'black'}
class_list = list(class_lut.keys())

os.chdir(correlation_pickle_dir)
po_class_lut = open('class_lut.pickle', 'wb')
pickle.dump(class_lut, po_class_lut)
po_class_lut.close()

# make arm dict and lut
arm_dict = {'B': 'adaptive', 'CD4T': 'adaptive', 'CD8T': 'adaptive',
            'DC': 'innate', 'NK': 'innate', 'Eo': 'innate',
            'PMN': 'innate', 'Mono': 'innate', 'Mac': 'innate',
            'DPT': 'adaptive', 'ISPT': 'adaptive', 'DNT': 'innate',
            'Precursor': 'other', 'LTi': 'innate',
            'unspecified': 'other'}

arm_lut = {'B': 'y', 'CD4T': 'y', 'CD8T': 'y', 'DC': 'm', 'NK': 'm',
           'PMN': 'm', 'Eo': 'm', 'Mono': 'm', 'Mac': 'm', 'DPT': 'y',
           'DNT': 'm', 'ISPT': 'y', 'Precursor': 'w', 'LTi': 'w',
           'unspecified': 'w'}


# make banner to introduce each RUNNING MODULE
def banner(MODULE_title):

    print('=' * 70)
    print(MODULE_title)
    print('=' * 70)


class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;"><td style="background-color: {0}; border: 0;"><code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>'
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html


def get_linkage_colors(dend, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(dend['color_list'], dend['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [dend[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes


# ------------------------------------------------------------------------------
# RUN SCRIPT

# extract replicate data from the dashboard dictionary, place into a dataframe
def get_dataframe():
    banner('RUNNING MODULE: get_dataframe')

    os.chdir(orthant_pickle_dir)
    dashboard_shlf = shelve.open('dashboard.shelve', writeback=True)

    print(color.CYAN + 'Extracting replicate data from the'
          ' dashboard dictionary.' + color.END)

    corr_dict = {}
    for celltype in dashboard_shlf.keys():
        corr_dict[celltype] = []
        for value in dashboard_shlf[celltype]:
            if str(value).startswith(
              tuple(dashboard_shlf[celltype]['sig_dif_all'].index)):
                tissue_name = value.split('_', 1)[0]
                condition_list = [i + ', ' + tissue_name for i in
                                  dashboard_shlf[celltype]['x_final_rep_data']]
                condition_list_update = []
                for j in condition_list:
                    y = list(j.split(','))
                    y = [x.strip(' ') for x in y]
                    condition_list_update.append(y)
                e = list(zip(condition_list_update,
                             dashboard_shlf[celltype][value]))
                corr_dict[celltype].append(e)
    print()

    # generate seperate lists of data from corr dict
    print(color.CYAN + 'Adding replicate data to correlation'
          ' DataFrame.' + color.END)

    cell_type_list = []
    status_list = []
    time_point_list = []
    replicate_list = []
    tissue_list = []
    percentage_list = []
    for celltype in corr_dict.keys():
        for value in corr_dict[celltype]:
            for tup in value:
                cell_type_list.append(celltype)
                status_list.append(tup[0][0])
                time_point_list.append(tup[0][1])
                replicate_list.append(tup[0][2])
                tissue_list.append(tup[0][3])
                percentage_list.append(tup[1])
    print()

    # add seperate data lists to respective columns of a single dataframe
    df = {}
    frame = pd.DataFrame()
    frame['cell_type'] = cell_type_list
    frame['status'] = status_list
    frame['time_point'] = time_point_list
    frame['replicate'] = replicate_list
    frame['tissue'] = tissue_list
    frame['percentage'] = percentage_list

    # add a column that contains all info on condition
    frame['status_tp_rep'] = (frame['status'].map(str) +
                              ', ' + frame['time_point'] + ', ' +
                              frame['replicate'])

    frame['celltype_tissue'] = (frame['cell_type'].map(str) +
                                '_' + frame['tissue'])
    frame.name = 'experimental_data'

    df['experimental_data'] = frame

    df['experimental_data'] = df['experimental_data'][
        df['experimental_data']['cell_type'] != 'unspecified']

    os.chdir(correlation_pickle_dir)
    po_df = open('df.pickle', 'wb')
    pickle.dump(df, po_df)
    po_df.close()


get_dataframe()


# generate a pivot table of the replicate data
def get_pivot_table(df):
    banner('RUNNING MODULE: get_pivot_table')

    os.chdir(orthant_pickle_dir)
    dashboard_shlf = shelve.open('dashboard.shelve', writeback=True)

    if df == 'experimental':

        print(color.CYAN + 'Converting experimental dataframe into pivot'
              ' table.' + color.END)
        print()

        os.chdir(correlation_pickle_dir)
        pi_df = open('df.pickle', 'rb')
        df = pickle.load(pi_df)

        unfiltered_pivot = {}
        for key, value in df.items():
            table = value.pivot_table(
                index='status_tp_rep', columns='celltype_tissue',
                values='percentage')

            idx = dashboard_shlf['B']['x_final_rep_data']
            table = table.reindex(index=idx)
            del table.columns.name
            table.name = key

            unfiltered_pivot[key] = table

        os.chdir(correlation_pickle_dir)
        po_unfiltered_pivot = open('unfiltered_pivot.pickle', 'wb')
        pickle.dump(unfiltered_pivot, po_unfiltered_pivot)
        po_unfiltered_pivot.close()

    elif df == 'shuffles':
        print(color.CYAN + 'Converting shuffled dataframes into pivot'
              ' tables.' + color.END)
        print()

        os.chdir(correlation_pickle_dir)
        pi_df_shuffles = open('df_shuffles.pickle', 'rb')
        df_shuffles = pickle.load(pi_df_shuffles)

        key1_list = []
        for key1, value1 in df_shuffles.items():
            key1_list.append(key1)

        unfiltered_pivot_shuffles = {}
        for key, value in df_shuffles.items():
            table = value.pivot_table(
                index='status_tp_rep', columns='celltype_tissue',
                values='percentage')

            idx = dashboard_shlf['B']['x_final_rep_data']
            table = table.reindex(index=idx)
            del table.columns.name
            table.name = key

            unfiltered_pivot_shuffles[key] = table

        os.chdir(correlation_pickle_dir)
        po_unfiltered_pivot_shuffles = open(
            'unfiltered_pivot_shuffles.pickle', 'wb')
        pickle.dump(unfiltered_pivot_shuffles, po_unfiltered_pivot_shuffles)
        po_unfiltered_pivot_shuffles.close()


get_pivot_table('experimental')


# split the pivot table into correlation groups of interest
def get_pivot_subsets(pivot_table):

    banner('RUNNING MODULE: get_pivot_subsets')

    os.chdir(correlation_pickle_dir)

    if pivot_table == 'unfiltered_pivot':

        pi_unfiltered_pivot = open('unfiltered_pivot.pickle', 'rb')
        unfiltered_pivot = pickle.load(pi_unfiltered_pivot)

        print(color.CYAN + 'Getting pivot table subsets from the experimental'
              ' data.' + color.END)
        print()

        unfiltered_pivot_subsets = {}
        for key, value in unfiltered_pivot.items():

            pivot_subsets = {}

            all_conds = value
            all_conds.name = 'all_conds_unfiltered'

            value = all_conds.copy()
            value.reset_index(drop=False, inplace=True)
            meta_data = pd.DataFrame(
                value.status_tp_rep.str.split(',', 2).tolist(),
                columns=['status', 'time_point', 'replicate'])
            value = pd.concat([value, meta_data], axis=1)

            for name, group in value.groupby(['status']):
                group = group.copy()
                group.set_index('status_tp_rep', drop=True, inplace=True)
                group.drop(
                    ['status', 'time_point', 'replicate'],
                    axis=1, inplace=True)
                group.name = ''.join(
                    group.index[0].split(',', 3)[0:1])
                pivot_subsets[group.name + '_unfiltered'] = group

            for name, group in value.groupby(['status', 'time_point']):
                group = group.copy()
                group.set_index('status_tp_rep', drop=True, inplace=True)
                group.drop(
                    ['status', 'time_point', 'replicate'],
                    axis=1, inplace=True)
                group.name = ''.join(
                    group.index[0].split(',', 3)[0:2]).replace(' ', '_')
                pivot_subsets[group.name + '_unfiltered'] = group

            pivot_subsets[all_conds.name] = all_conds

            unfiltered_pivot_subsets[key] = pivot_subsets

        os.chdir(correlation_pickle_dir)
        po_unfiltered_pivot_subsets = open(
            'unfiltered_pivot_subsets.pickle', 'wb')
        pickle.dump(unfiltered_pivot_subsets, po_unfiltered_pivot_subsets)
        po_unfiltered_pivot_subsets.close()

    elif pivot_table == 'unfiltered_pivot_shuffles':

        pi_unfiltered_pivot_shuffles = open(
            'unfiltered_pivot_shuffles.pickle', 'rb')
        unfiltered_pivot_shuffles = pickle.load(pi_unfiltered_pivot_shuffles)

        print(color.CYAN + 'Getting pivot table subsets from the ' +
              str(len(unfiltered_pivot_shuffles.keys())) + '-runs of'
              ' shuffled experimental data.' + color.END)
        print()

        unfiltered_pivot_subsets_shuffles = {}
        for key, value in unfiltered_pivot_shuffles.items():

            pivot_subsets = {}

            all_conds = value
            all_conds.name = 'all_conds_unfiltered'

            value = all_conds.copy()
            value.reset_index(drop=False, inplace=True)
            meta_data = pd.DataFrame(
                value.status_tp_rep.str.split(',', 2).tolist(),
                columns=['status', 'time_point', 'replicate'])
            value = pd.concat([value, meta_data], axis=1)

            for name, group in value.groupby(['status']):
                group = group.copy()
                group.set_index('status_tp_rep', drop=True, inplace=True)
                group.drop(
                    ['status', 'time_point', 'replicate'],
                    axis=1, inplace=True)
                group.name = ''.join(
                    group.index[0].split(',', 3)[0:1])
                pivot_subsets[group.name + '_unfiltered'] = group

            for name, group in value.groupby(['status', 'time_point']):
                group = group.copy()
                group.set_index('status_tp_rep', drop=True, inplace=True)
                group.drop(
                    ['status', 'time_point', 'replicate'],
                    axis=1, inplace=True)
                group.name = ''.join(
                    group.index[0].split(',', 3)[0:2]).replace(' ', '_')
                pivot_subsets[group.name + '_unfiltered'] = group

            pivot_subsets[all_conds.name] = all_conds

            unfiltered_pivot_subsets_shuffles[key] = pivot_subsets

        os.chdir(correlation_pickle_dir)
        po_unfiltered_pivot_subsets_shuffles = open(
            'unfiltered_pivot_subsets_shuffles.pickle', 'wb')
        pickle.dump(unfiltered_pivot_subsets_shuffles,
                    po_unfiltered_pivot_subsets_shuffles)
        po_unfiltered_pivot_subsets_shuffles.close()


get_pivot_subsets('unfiltered_pivot')


# per tissue-specific immunophenotype (columns), set all mouse replicate subset
# percentage values to zero if at least one falls below percentage cutoff
def percentage_filter(percentage_cutoff):
    banner('RUNNING MODULE: percentage_filter')

    os.chdir(correlation_pickle_dir)
    pi_unfiltered_pivot_subsets = open('unfiltered_pivot_subsets.pickle', 'rb')
    unfiltered_pivot_subsets = pickle.load(pi_unfiltered_pivot_subsets)

    print(color.CYAN + 'Setting all celltype_tissue combination values'
          ' to zero if at least one value is < ' +
          str(percentage_cutoff) + '.' + color.END)

    filtered_pivot_subsets = copy.deepcopy(unfiltered_pivot_subsets)

    for key1, value1 in filtered_pivot_subsets.items():
        for key2, value2 in value1.items():
            new_dict = {
                re.sub('\_unfiltered$', '', k) + '_filtered':
                v for k, v in value1.items()}
            for key3, value3 in new_dict.items():
                for col in value3:
                    col_list = []
                    for i in value3[col]:
                        col_list.append(i)
                    if min(col_list) < percentage_cutoff:
                        value3[col] = 0.0
                    else:
                        value3[col] = value3[col]

        filtered_pivot_subsets[key1] = new_dict
    print()

    os.chdir(correlation_pickle_dir)
    po_filtered_pivot_subsets = open(
        'filtered_pivot_subsets.pickle', 'wb')
    pickle.dump(filtered_pivot_subsets, po_filtered_pivot_subsets)
    po_filtered_pivot_subsets.close()


percentage_filter(0.1)


def spearman_rho(pivot_subsets):

    banner('RUNNING MODULE: spearman_rho')

    os.chdir(correlation_pickle_dir)

    pi_filtered_pivot_subsets = open('filtered_pivot_subsets.pickle', 'rb')
    filtered_pivot_subsets = pickle.load(pi_filtered_pivot_subsets)

    key1_list = []
    key2_list = []
    for key1, value1 in filtered_pivot_subsets.items():
        key1_list.append(key1)
        for key2, value2 in value1.items():
            key2_list.append(key2)

    if 'experimental_data' in key1_list:

        print(color.CYAN + 'Spearman_rho input is filtered'
              ' experimental data, running Spearman correlation.'
              + color.END)

        filtered_subset_corrs = {}
        filtered_subset_pvals = {}
        for key1, value1 in filtered_pivot_subsets.items():

            rho_conditions = {}
            pvals_conditions = {}

            for key2, value2 in value1.items():

                print('Running Spearman correlation on the ' + key2 +
                      ' condition subset of ' + key1 + '.')

                rho, pval = spearmanr(value2)

                rho_df = pd.DataFrame(
                    data=rho, index=value2.columns, columns=value2.columns)

                pvals_df = pd.DataFrame(
                    data=pval, index=value2.columns, columns=value2.columns)

                rho_conditions[key2] = rho_df
                pvals_conditions[key2] = pvals_df

            filtered_subset_corrs[key1] = rho_conditions
            filtered_subset_pvals[key1] = pvals_conditions
            print()

        os.chdir(correlation_pickle_dir)

        po_filtered_subset_corrs = open(
            'filtered_subset_corrs.pickle', 'wb')
        pickle.dump(filtered_subset_corrs, po_filtered_subset_corrs)
        po_filtered_subset_corrs.close()

        po_filtered_subset_pvals = open(
            'filtered_subset_pvals.pickle', 'wb')
        pickle.dump(filtered_subset_pvals, po_filtered_subset_pvals)
        po_filtered_subset_pvals.close()


spearman_rho('filtered_pivot_subsets')


def sig_corrs(correlation_matrix_type):

    banner('RUNNING MODULE: sig_corrs')

    os.chdir(correlation_pickle_dir)

    pi_filtered_subset_corrs = open(
        'filtered_subset_corrs.pickle', 'rb')
    filtered_subset_corrs = pickle.load(pi_filtered_subset_corrs)

    pi_filtered_subset_pvals = open(
        'filtered_subset_pvals.pickle', 'rb')
    filtered_subset_pvals = pickle.load(
        pi_filtered_subset_pvals)

    print(color.CYAN + 'Finding significant correlations in the'
          ' filtered data.' + color.END)

    filtered_sig_corr_dict = {}
    for (key1x, value1x), (key1s, value1s) in zip(
      filtered_subset_corrs.items(),
      filtered_subset_pvals.items()):

        for (key2x, value2x), (key2s, value2s) in zip(
          value1x.items(), value1s.items()):

            print('Getting the ' + key2x +
                  ' significant correlations.')

            if key2x == key2s:

                condition_x = value2x.copy()
                condition_s = value2s.copy()

                condition_x.replace(to_replace=np.nan, value=0.0, inplace=True)
                condition_s.replace(to_replace=np.nan, value=1.0, inplace=True)

                m, n = condition_x.shape
                condition_x[:] = np.where(
                    np.arange(m)[:, None] < np.arange(n), np.nan, condition_x)
                for s in np.arange(len(condition_x)):
                    condition_x.iloc[s, s] = np.nan

                m, n = condition_s.shape
                condition_s[:] = np.where(
                    np.arange(m)[:, None] < np.arange(n), np.nan, condition_s)
                for s in np.arange(len(condition_s)):
                    condition_s.iloc[s, s] = np.nan

                condition_x_final = condition_x.unstack().dropna()
                condition_x_final.name = 'rho'

                condition_s_final = condition_s.unstack().dropna()
                condition_s_final.name = 'two_tailed_pval'

                uncorrected_pvals = pd.concat(
                    [condition_x_final, condition_s_final], axis=1)

                # implement false discovery rate (FDR) p-value correction
                stats = importr('stats')
                p_adjust = stats.p_adjust(
                    FloatVector(uncorrected_pvals['two_tailed_pval'].tolist()),
                    method='BH')

                total_corrected_qvals = uncorrected_pvals.copy()
                total_corrected_qvals['two_tailed_qval'] = p_adjust
                total_corrected_qvals.drop(
                    'two_tailed_pval', axis=1, inplace=True)

                sig_corrected_qvals = total_corrected_qvals[
                    total_corrected_qvals['two_tailed_qval'] <= 0.05]

                filtered_sig_corr_dict[key2x] = sig_corrected_qvals
    print()

    for key, value in filtered_sig_corr_dict.items():
        print(key + ' ===================================================')
        print(value)
        print()

    os.chdir(correlation_pickle_dir)
    po_filtered_sig_corr_dict = open(
        'filtered_sig_corr_dict.pickle', 'wb')
    pickle.dump(filtered_sig_corr_dict,
                po_filtered_sig_corr_dict)
    po_filtered_sig_corr_dict.close()


sig_corrs('filtered_subset_corrs')


def plot_clustermap(correlation_matrix, data_type, file_path):
    banner('RUNNING MODULE: plot_clustermap')

    if os.path.isdir(file_path + '/clustermaps') is False:

        print(color.CYAN + 'Dendrogram directory does not exist,'
              ' creating the directory and saving coefficient dendrograms.'
              + color.END)
        print()

        clustermaps = os.path.join(file_path, 'clustermaps')
        os.makedirs(clustermaps)

    else:

        print(color.CYAN + 'Dendrogram directory already exist,'
              ' saving coefficient dendrograms.' + color.END)

        clustermaps = os.path.join(file_path, 'clustermaps')

    if data_type == 'experimental':

        pi_name = 'pi' + correlation_matrix

        os.chdir(correlation_pickle_dir)
        pi_name = open(correlation_matrix + '.pickle', 'rb')
        correlation_matrix_dict = pickle.load(
            pi_name)

        if os.path.isdir(
          clustermaps + '/experimental_data') is False:

            experimental_data_cmaps = os.path.join(
                clustermaps, 'experimental_data')
            os.makedirs(experimental_data_cmaps)

        else:
            experimental_data_cmaps = os.path.join(
                clustermaps, 'experimental_data')

        if os.path.isdir(
          clustermaps + '/experimental_data_dropped') is False:

            experimental_data_dropped_cmaps = os.path.join(
                clustermaps, 'experimental_data_dropped')
            os.makedirs(experimental_data_dropped_cmaps)

        else:
            experimental_data_dropped_cmaps = os.path.join(
                clustermaps, 'experimental_data_dropped')

        if os.path.isdir(
          clustermaps + '/experimental_data/combo') is True:

            pass

        correlation_matrix_dict_copy = copy.deepcopy(
            correlation_matrix_dict)

        for key1, value1 in correlation_matrix_dict_copy.items():

            if 'unfiltered' in correlation_matrix:
                if 'dropped' in correlation_matrix:
                    dict_name1 = 'unfiltered_exp_cdfp_drp'
                    dict_name2 = {}
                else:
                    dict_name1 = 'unfiltered_exp_cdfp'
                    dict_name2 = {}

            else:
                if 'dropped' in correlation_matrix:
                    dict_name1 = 'filtered_exp_cdfp_drp'
                    dict_name2 = {}
                else:
                    dict_name1 = 'filtered_exp_cdfp'
                    dict_name2 = {}

            for key2, value2 in value1.items():

                if 'dropped' in correlation_matrix:
                    key_dir = os.path.join(
                        experimental_data_dropped_cmaps, key2)
                    os.makedirs(key_dir)

                    print('Saving the ' + key2 +
                          ' experimental dropped clustermap.')

                else:
                    key_dir = os.path.join(
                        experimental_data_cmaps, key2)
                    os.makedirs(key_dir)

                    print('Saving the ' + key2 +
                          ' experimental clustermap.')

                    value2.replace(to_replace=np.nan, value=0.0, inplace=True)

                # get row index components lists
                idx_full = value2.index.tolist()
                idx_cell = [i.split('_', 2)[0] for i in idx_full]
                idx_tissue = [i.split('_', 2)[1] for i in idx_full]

                # generate class index from full celltype index
                idx_major_classes = []
                for longer in idx_cell:
                    if any(substring in longer for
                           substring in class_list) is True:
                        x = ''.join(
                            [sub for sub in class_list if sub in longer])
                        idx_major_classes.append(x)

                for i, cell in enumerate(idx_major_classes):
                    if cell == 'BNK':
                        idx_major_classes[i] = 'NK'
                    elif cell == 'BCD8T':
                        idx_major_classes[i] = 'CD8T'
                    elif cell == 'BMac':
                        idx_major_classes[i] = 'Mac'
                    elif cell == 'BISPT':
                        idx_major_classes[i] = 'ISPT'

                # get different row colors
                classes = pd.Series(idx_major_classes)
                classes.name = 'class'
                row_colors_classes = classes.map(class_lut)
                row_colors_classes.index = idx_full

                arm = pd.Series(idx_major_classes)
                arm.name = 'arm'
                row_colors_arm = arm.map(arm_lut)
                row_colors_arm.index = idx_full

                lineage = pd.Series(idx_cell)
                lineage.name = 'lineage'
                row_colors_lineage = lineage.map(lineage_lut)
                row_colors_lineage.index = idx_full

                tissue = pd.Series(idx_tissue)
                tissue.name = 'tissue'
                row_colors_tissue = tissue.map(color_dict)
                row_colors_tissue.index = idx_full

                # combine all row color types
                row_colors_combo = pd.DataFrame()
                row_colors_combo['arm'] = row_colors_arm
                row_colors_combo['lineage'] = row_colors_lineage
                row_colors_combo['tissue'] = row_colors_tissue
                row_colors_combo['class'] = row_colors_classes

                e = sns.clustermap(value2, xticklabels=1, yticklabels=1)
                plt.close('all')

                # get row linkage
                linkage_row = e.dendrogram_row.linkage

                # assign the default dendrogram
                # distance threshold = 0.7*max(Z[:, 2])
                threshold = 4.676  # 0.6*max(linkage_row[:, 2])

                # get cluster assignments and
                # generate mapped_clusters dataframe
                clusters = fcluster(
                    linkage_row, threshold, criterion='distance')

                mapped_clusters = pd.DataFrame(
                    {'condition': value2.index.tolist(),
                     'cluster': clusters})

                mapped_clusters['class'] = idx_major_classes
                mapped_clusters['tissue'] = tissue
                mapped_clusters['arm'] = mapped_clusters['class'].map(
                    arm_dict)
                mapped_clusters['lineage'] = lineage.map(lineage_dict)

                # rearrange column order in row_colors_combo
                row_colors_combo.columns = [
                    'arm', 'lineage', 'tissue', 'class']

                # generate cluster_lut whose length is
                # proportional to the number of identified clusters
                c_list = ['r', 'g', 'b', 'y', 'm', 'c']
                c_list = c_list * int(
                    (math.ceil(max(clusters)/len(c_list))))
                idx_list = list(range(len(c_list)))
                idx_list = [i+1 for i in idx_list]
                cluster_lut_col = dict(zip(idx_list, c_list))

                # get cluster row color and add to row_colors_combo
                cluster = pd.Series(mapped_clusters['cluster'])
                cluster.name = 'cluster'
                row_cluster_colors = cluster.map(cluster_lut_col)

                row_full_update = [i.replace('neg', '$^-$').replace(
                    'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1') for i in idx_full]

                row_cluster_colors.index = row_full_update

                row_cluster_colors_dict = row_cluster_colors.to_dict()

                row_colors_combo['cluster'] = row_cluster_colors

                # plot dendrogram
                dend = dendrogram(
                    linkage_row, color_threshold=threshold,
                    above_threshold_color='grey',
                    leaf_font_size=7, labels=value2.index)
                plt.close('all')

                leaf_colors = get_linkage_colors(dend)

                leaf_colors_long = {}
                for c in leaf_colors.keys():
                    next_color = dict(
                        zip(leaf_colors[c], [c] * len(leaf_colors[c])))
                    leaf_colors_long.update(next_color)

                # change default color scheme
                for key, value in leaf_colors_long.items():
                    if value == 'r':
                        leaf_colors_long[key] = 'g'
                    elif value == 'g':
                        leaf_colors_long[key] = 'r'
                    elif value == 'c':
                        leaf_colors_long[key] = 'b'
                    elif value == 'm':
                        leaf_colors_long[key] = 'y'
                    elif value == 'y':
                        leaf_colors_long[key] = 'm'
                    elif value == 'k':
                        leaf_colors_long[key] = 'c'

                leaf_indices = [
                    list(x) for x in zip(dend['ivl'], dend['leaves'])]
                leaf_indices_frame = pd.DataFrame(leaf_indices)
                leaf_indices_frame.rename(
                    columns={0: 'leaf', 1: 'index'}, inplace=True)

                leaf_colors_frame = pd.DataFrame(
                    list(leaf_colors_long.items()))
                leaf_colors_frame.rename(
                    columns={0: 'leaf', 1: 'color'}, inplace=True)

                color_index = leaf_indices_frame.merge(
                    leaf_colors_frame, how='inner', on='leaf')

                c = color_index['color'].tolist()
                index = color_index['index'].tolist()
                leaf = ['leaf'] * len(index)

                prefix_frame = pd.DataFrame(
                    {'prefix': leaf, 'index': index})
                prefix_frame['index'] = prefix_frame['index'].astype(str)
                prefix_frame['prefix_index'] = prefix_frame[
                    'prefix'].map(str) + '_' + prefix_frame['index']

                link_color_index = prefix_frame['prefix_index'].tolist()
                link_color_dict = dict(zip(link_color_index, c))

                dflt_color = 'grey'
                link_colors = {}
                for i, i12 in enumerate(linkage_row[:, :2].astype(int)):
                    c1, c2 = (link_colors[x] if x > len(linkage_row) else
                              link_color_dict['leaf_%d' % x]
                              for x in i12)
                    link_colors[i + 1 +
                                len(linkage_row)
                                ] = c1 if c1 == c2 else dflt_color

                dend_labels = [
                    i.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    for i in value2.index.tolist()]

                dend_plot = dendrogram(
                    linkage_row, color_threshold=None,
                    link_color_func=lambda x: link_colors[x],
                    leaf_font_size=2, labels=dend_labels)

                dend_plot = plt.gcf()
                ax = plt.gca()
                ax.set_facecolor('white')
                ax.grid(color='grey', linestyle='-',
                        linewidth=0.5, alpha=0.5)
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)

                new_key_list = []
                value_list = []
                for key, value in leaf_colors_long.items():

                    new_key = key.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')
                    new_key_list.append(new_key)
                    value_list.append(value)

                leaf_colors_long_newkeys = dict(
                    zip(new_key_list, value_list))

                for item in ax.get_xticklabels():
                    name = item.get_text()
                    item.set_color(leaf_colors_long_newkeys[name])

                plt.tight_layout()
                dend_plot.savefig(
                    os.path.join(key_dir, 'dendrogram' + '.pdf'))

                value2 = value2.copy()

                xlabels = [i.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1') for i in value2.columns]
                value2.columns = xlabels

                ylabels = [i.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1') for i in value2.index]
                value2.index = ylabels

                # if status included
                if idx_full[0].count('_') == 2:

                    idx_status = [i.split('_', 3)[2] for i in idx_full]
                    status = pd.Series(idx_status)
                    status.name = 'status'
                    status_dict = {'naive': 'k', 'gl261': 'gray'}
                    row_colors_status = status.map(status_dict)
                    row_colors_status.index = idx_full

                    row_colors_combo = pd.DataFrame()
                    row_colors_combo['status'] = row_colors_status

                    g = sns.clustermap(
                        value2, method='average', metric='euclidean',
                        row_colors=row_cluster_colors,
                        cmap='cividis', center=0.0, xticklabels=1,
                        yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    g.ax_heatmap.set_xlabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.ax_heatmap.set_ylabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.savefig(os.path.join(key_dir, 'clustermap' + '.pdf'))
                    plt.close('all')

                else:

                    g = sns.clustermap(
                        value2, method='average', metric='euclidean',
                        row_colors=row_cluster_colors,
                        cmap='cividis', center=0.0, xticklabels=1,
                        yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    g.ax_heatmap.set_xlabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.ax_heatmap.set_ylabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.savefig(os.path.join(key_dir, 'clustermap' + '.pdf'))
                    plt.close('all')

                # binom.cdf(k, n, p)
                cdf_dict = {}

                for col in mapped_clusters[
                  ['class', 'tissue', 'arm', 'lineage']]:

                    p = mapped_clusters.groupby(col).size()/len(
                        mapped_clusters)
                    n = mapped_clusters.groupby(['cluster']).size()
                    k = mapped_clusters.pivot_table(
                        index='cluster', columns=col, values='condition',
                        fill_value=0, aggfunc='count').unstack()

                    for i, cluster in zip(range(len(n)), n):
                        clust_idx = n[n == cluster].index[0]
                        for j, idx, percent in zip(
                          range(len(p)), p.index, p):
                            x_list = []
                            y_list = []
                            pval_list = []
                            for x in range(cluster+1):
                                x_list.append(x)
                                y_list.append(
                                    binom.cdf(
                                        k=(x), n=cluster, p=percent))
                                pval_list.append(
                                    1-binom.cdf(
                                        k=(x), n=cluster, p=percent))

                            data = pd.DataFrame(
                                OrderedDict({'cluster': clust_idx, 'ID': idx,
                                            'k': x_list, 'F(k)': y_list,
                                             '1-F(k)': pval_list}))

                            cdf_dict[
                                'k=' + str(k[idx][clust_idx]), 'n=' +
                                str(n[clust_idx]), 'p=' + str(p[idx]),
                                idx, 'cluster_' +
                                     str(clust_idx), col] = data

                # get cluster enrichments
                clss = []
                cluster_clss = []
                cdfp_clss = []

                tis = []
                cluster_tis = []
                cdfp_tis = []

                lin = []
                cluster_lin = []
                cdfp_lin = []

                ar = []
                cluster_ar = []
                cdfp_ar = []

                for key, value in cdf_dict.items():

                    if key[5] == 'class':
                        # print(key)
                        clss.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_clss.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_clss.append(pval)

                    elif key[5] == 'tissue':
                        # print(key)
                        tis.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_tis.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_tis.append(pval)

                    elif key[5] == 'lineage':
                        # print(key)
                        lin.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_lin.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_lin.append(pval)

                    elif key[5] == 'arm':
                        # print(key)
                        ar.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_ar.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_ar.append(pval)

                cdfp_dict_inner = {}
                for name, u in zip(
                    ['class', 'tissue', 'lineage', 'arm'], [
                        [clss, cluster_clss, cdfp_clss],
                        [tis, cluster_tis, cdfp_tis],
                        [lin, cluster_lin, cdfp_lin],
                        [ar, cluster_ar, cdfp_ar]]):

                    cdfp_frame = pd.DataFrame(u)
                    cdfp_frame = cdfp_frame.T
                    cdfp_frame.rename(
                        columns={0: 'classifier', 1: 'cluster', 2: 'pval'},
                        inplace=True)

                    cdfp_frame_threshold = cdfp_frame[
                        cdfp_frame['pval'] <= 0.05].sort_values(by='pval')

                    cdfp_dict_inner[name] = cdfp_frame_threshold
                dict_name2[key2] = cdfp_dict_inner

                os.chdir(key_dir)
                po_mapped_clusters = open(
                    'mapped_clusters.pickle', 'wb')
                pickle.dump(mapped_clusters, po_mapped_clusters)
                po_mapped_clusters.close()

                po_cdfp_dict_inner = open(
                    'cdfp_dict_inner.pickle', 'wb')
                pickle.dump(cdfp_dict_inner, po_cdfp_dict_inner)
                po_cdfp_dict_inner.close()

                pprint.pprint(cdfp_dict_inner, width=1)
                print()
            print()

        if 'dropped' in correlation_matrix:

            name = correlation_matrix + '_final'
            po_name = 'po_' + correlation_matrix + '_final'

            os.chdir(correlation_pickle_dir)
            po_name = open(name + '.pickle', 'wb')
            name = correlation_matrix_dict_copy
            pickle.dump(name, po_name)
            po_name.close()

        else:
            name = correlation_matrix + '_zeroed'
            po_name = 'po_' + correlation_matrix + '_zeroed'

            os.chdir(correlation_pickle_dir)
            po_name = open(name + '.pickle', 'wb')
            name = correlation_matrix_dict_copy
            pickle.dump(name, po_name)
            po_name.close()

        po_dict_name = 'po_' + dict_name1

        po_dict_name = open(dict_name1 + '.pickle', 'wb')
        dict_name1 = dict_name2
        pickle.dump(dict_name1, po_dict_name)
        po_dict_name.close()

    elif data_type == 'combo':

        pi_name = 'pi' + correlation_matrix

        os.chdir(correlation_pickle_dir)
        pi_name = open(correlation_matrix + '.pickle', 'rb')
        correlation_matrix_dict = pickle.load(
            pi_name)

        if os.path.isdir(
          clustermaps + '/experimental_data') is False:

            experimental_data_cmaps = os.path.join(
                clustermaps, 'experimental_data')
            os.makedirs(experimental_data_cmaps)

        else:
            experimental_data_cmaps = os.path.join(
                clustermaps, 'experimental_data')

        if os.path.isdir(
          clustermaps + '/experimental_data_dropped') is False:

            experimental_data_dropped_cmaps = os.path.join(
                clustermaps, 'experimental_data_dropped')
            os.makedirs(experimental_data_dropped_cmaps)

        else:
            experimental_data_dropped_cmaps = os.path.join(
                clustermaps, 'experimental_data_dropped')

        correlation_matrix_dict_copy = copy.deepcopy(
            correlation_matrix_dict)

        for key1, value1 in correlation_matrix_dict_copy.items():

            if 'unfiltered' in correlation_matrix:
                if 'dropped' in correlation_matrix:
                    dict_name1 = 'unfiltered_exp_cdfp_drp_cmbo'
                    dict_name2 = {}
                else:
                    dict_name1 = 'unfiltered_exp_cdfp_cmbo'
                    dict_name2 = {}

            else:
                if 'dropped' in correlation_matrix:
                    dict_name1 = 'filtered_exp_cdfp_drp_cmbo'
                    dict_name2 = {}
                else:
                    dict_name1 = 'filtered_exp_cdfp_cmbo'
                    dict_name2 = {}

            for key2, value2 in value1.items():

                if 'dropped' in correlation_matrix:
                    key_dir = os.path.join(
                        experimental_data_dropped_cmaps, key2)
                    os.makedirs(key_dir)

                    print('Saving the ' + key2 +
                          ' experimental dropped combo clustermap.')

                else:
                    key_dir = os.path.join(
                        experimental_data_cmaps, key2)
                    os.makedirs(key_dir)

                    print('Saving the ' + key2 +
                          ' experimental combo clustermap.')

                value2.replace(to_replace=np.nan, value=0.0, inplace=True)

                # get row index components lists
                idx_full = value2.index.tolist()
                idx_cell = [i.split('_', 2)[0] for i in idx_full]
                idx_tissue = [i.split('_', 2)[1] for i in idx_full]

                # generate class index from full celltype index
                idx_major_classes = []
                for longer in idx_cell:
                    if any(substring in longer for
                           substring in class_list) is True:
                        x = ''.join(
                            [sub for sub in class_list if sub in longer])
                        idx_major_classes.append(x)

                for i, cell in enumerate(idx_major_classes):
                    if cell == 'BNK':
                        idx_major_classes[i] = 'NK'
                    elif cell == 'BCD8T':
                        idx_major_classes[i] = 'CD8T'
                    elif cell == 'BMac':
                        idx_major_classes[i] = 'Mac'
                    elif cell == 'BISPT':
                        idx_major_classes[i] = 'ISPT'

                # get different row colors
                classes = pd.Series(idx_major_classes)
                classes.name = 'class'
                row_colors_classes = classes.map(class_lut)
                row_colors_classes.index = idx_full

                arm = pd.Series(idx_major_classes)
                arm.name = 'arm'
                row_colors_arm = arm.map(arm_lut)
                row_colors_arm.index = idx_full

                lineage = pd.Series(idx_cell)
                lineage.name = 'lineage'
                row_colors_lineage = lineage.map(lineage_lut)
                row_colors_lineage.index = idx_full

                tissue = pd.Series(idx_tissue)
                tissue.name = 'tissue'
                row_colors_tissue = tissue.map(color_dict)
                row_colors_tissue.index = idx_full

                # combine all row color types
                row_colors_combo = pd.DataFrame()
                row_colors_combo['arm'] = row_colors_arm
                row_colors_combo['lineage'] = row_colors_lineage
                row_colors_combo['tissue'] = row_colors_tissue
                row_colors_combo['class'] = row_colors_classes

                e = sns.clustermap(value2)
                plt.close('all')

                # get row linkage
                linkage_row = e.dendrogram_row.linkage

                # assign the default dendrogram
                # distance threshold = 0.7*max(Z[:, 2])
                threshold = 4.676  # 0.6*max(linkage_row[:, 2])

                # get cluster assignments and
                # generate mapped_clusters dataframe
                clusters = fcluster(
                    linkage_row, threshold, criterion='distance')

                mapped_clusters = pd.DataFrame(
                    {'condition': value2.index.tolist(),
                     'cluster': clusters})

                mapped_clusters['class'] = idx_major_classes
                mapped_clusters['tissue'] = tissue
                mapped_clusters['arm'] = mapped_clusters['class'].map(
                    arm_dict)
                mapped_clusters['lineage'] = lineage.map(lineage_dict)

                # rearrange column order in row_colors_combo
                row_colors_combo.columns = [
                    'arm', 'lineage', 'tissue', 'class']

                # generate cluster_lut whose length is
                # proportional to the number of identified clusters
                c_list = ['r', 'g', 'b', 'y', 'm', 'c']
                c_list = c_list * int(
                    (math.ceil(max(clusters)/len(c_list))))
                idx_list = list(range(len(c_list)))
                idx_list = [i+1 for i in idx_list]
                cluster_lut_col = dict(zip(idx_list, c_list))

                # get cluster row color and add to row_colors_combo
                cluster = pd.Series(mapped_clusters['cluster'])
                cluster.name = 'cluster'
                row_cluster_colors = cluster.map(cluster_lut_col)

                row_full_update = [i.replace('neg', '$^-$').replace(
                    'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1') for i in idx_full]

                row_cluster_colors.index = row_full_update

                row_cluster_colors_dict = row_cluster_colors.to_dict()

                row_colors_combo['cluster'] = row_cluster_colors

                # plot dendrogram
                dend = dendrogram(
                    linkage_row, color_threshold=threshold,
                    above_threshold_color='grey',
                    leaf_font_size=7, labels=value2.index)
                plt.close('all')

                leaf_colors = get_linkage_colors(dend)

                leaf_colors_long = {}
                for c in leaf_colors.keys():
                    next_color = dict(
                        zip(leaf_colors[c], [c] * len(leaf_colors[c])))
                    leaf_colors_long.update(next_color)

                # change default color scheme
                for key, value in leaf_colors_long.items():
                    if value == 'r':
                        leaf_colors_long[key] = 'g'
                    elif value == 'g':
                        leaf_colors_long[key] = 'r'
                    elif value == 'c':
                        leaf_colors_long[key] = 'b'
                    elif value == 'm':
                        leaf_colors_long[key] = 'y'
                    elif value == 'y':
                        leaf_colors_long[key] = 'm'
                    elif value == 'k':
                        leaf_colors_long[key] = 'c'

                leaf_indices = [
                    list(x) for x in zip(dend['ivl'], dend['leaves'])]
                leaf_indices_frame = pd.DataFrame(leaf_indices)
                leaf_indices_frame.rename(
                    columns={0: 'leaf', 1: 'index'}, inplace=True)

                leaf_colors_frame = pd.DataFrame(
                    list(leaf_colors_long.items()))
                leaf_colors_frame.rename(
                    columns={0: 'leaf', 1: 'color'}, inplace=True)

                color_index = leaf_indices_frame.merge(
                    leaf_colors_frame, how='inner', on='leaf')

                c = color_index['color'].tolist()
                index = color_index['index'].tolist()
                leaf = ['leaf'] * len(index)

                prefix_frame = pd.DataFrame(
                    {'prefix': leaf, 'index': index})
                prefix_frame['index'] = prefix_frame['index'].astype(str)
                prefix_frame['prefix_index'] = prefix_frame[
                    'prefix'].map(str) + '_' + prefix_frame['index']

                link_color_index = prefix_frame['prefix_index'].tolist()
                link_color_dict = dict(zip(link_color_index, c))

                dflt_color = 'grey'
                link_colors = {}
                for i, i12 in enumerate(linkage_row[:, :2].astype(int)):
                    c1, c2 = (link_colors[x] if x > len(linkage_row) else
                              link_color_dict['leaf_%d' % x]
                              for x in i12)
                    link_colors[i + 1 +
                                len(linkage_row)
                                ] = c1 if c1 == c2 else dflt_color

                dend_labels = [
                    i.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    for i in value2.index.tolist()]

                dend_plot = dendrogram(
                    linkage_row, color_threshold=None,
                    link_color_func=lambda x: link_colors[x],
                    leaf_font_size=2, labels=dend_labels)

                dend_plot = plt.gcf()
                ax = plt.gca()
                ax.set_facecolor('white')
                ax.grid(color='grey', linestyle='-',
                        linewidth=0.5, alpha=0.5)
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)

                new_key_list = []
                value_list = []
                for key, value in leaf_colors_long.items():

                    new_key = key.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')
                    new_key_list.append(new_key)
                    value_list.append(value)

                leaf_colors_long_newkeys = dict(
                    zip(new_key_list, value_list))

                for item in ax.get_xticklabels():
                    name = item.get_text()
                    item.set_color(leaf_colors_long_newkeys[name])

                plt.tight_layout()
                dend_plot.savefig(
                    os.path.join(key_dir, 'dendrogram' + '.pdf'))

                value2 = value2.copy()

                xlabels = [i.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1') for i in value2.columns]
                value2.columns = xlabels

                ylabels = [i.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1') for i in value2.index]
                value2.index = ylabels

                # if status included
                if idx_full[0].count('_') == 2:

                    idx_status = [i.split('_', 3)[2] for i in idx_full]
                    status = pd.Series(idx_status)
                    status.name = 'status'
                    status_dict = {'naive': 'k', 'gl261': 'gray'}
                    row_colors_status = status.map(status_dict)
                    row_colors_status.index = idx_full

                    row_colors_combo = pd.DataFrame()
                    row_colors_combo['status'] = row_colors_status

                    g = sns.clustermap(
                        value2, method='average', metric='euclidean',
                        row_colors=row_cluster_colors,
                        cmap='cividis', center=0.0, xticklabels=1,
                        yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    g.ax_heatmap.set_xlabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.ax_heatmap.set_ylabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.savefig(os.path.join(key_dir, 'clustermap' + '.pdf'))
                    plt.close('all')

                else:

                    g = sns.clustermap(
                        value2, method='average', metric='euclidean',
                        row_colors=row_cluster_colors,
                        cmap='cividis', center=0.0, xticklabels=1,
                        yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    g.ax_heatmap.set_xlabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.ax_heatmap.set_ylabel(
                        'immunophenotype' + '$_i$' + ' in ' +
                        'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.savefig(os.path.join(key_dir, 'clustermap' + '.pdf'))
                    plt.close('all')

                # binom.cdf(k, n, p)
                cdf_dict = {}

                for col in mapped_clusters[
                  ['class', 'tissue', 'arm', 'lineage']]:

                    p = mapped_clusters.groupby(col).size()/len(
                        mapped_clusters)
                    n = mapped_clusters.groupby(['cluster']).size()
                    k = mapped_clusters.pivot_table(
                        index='cluster', columns=col, values='condition',
                        fill_value=0, aggfunc='count').unstack()

                    for i, cluster in zip(range(len(n)), n):
                        clust_idx = n[n == cluster].index[0]
                        for j, idx, percent in zip(
                          range(len(p)), p.index, p):
                            x_list = []
                            y_list = []
                            pval_list = []
                            for x in range(cluster+1):
                                x_list.append(x)
                                y_list.append(
                                    binom.cdf(
                                        k=(x), n=cluster, p=percent))
                                pval_list.append(
                                    1-binom.cdf(
                                        k=(x), n=cluster, p=percent))

                            data = pd.DataFrame(
                                OrderedDict({'cluster': clust_idx, 'ID': idx,
                                            'k': x_list, 'F(k)': y_list,
                                             '1-F(k)': pval_list}))

                            cdf_dict[
                                'k=' + str(k[idx][clust_idx]), 'n=' +
                                str(n[clust_idx]), 'p=' + str(p[idx]),
                                idx, 'cluster_' +
                                     str(clust_idx), col] = data

                # get cluster enrichments
                clss = []
                cluster_clss = []
                cdfp_clss = []

                tis = []
                cluster_tis = []
                cdfp_tis = []

                lin = []
                cluster_lin = []
                cdfp_lin = []

                ar = []
                cluster_ar = []
                cdfp_ar = []

                for key, value in cdf_dict.items():

                    if key[5] == 'class':
                        # print(key)
                        clss.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_clss.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_clss.append(pval)

                    elif key[5] == 'tissue':
                        # print(key)
                        tis.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_tis.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_tis.append(pval)

                    elif key[5] == 'lineage':
                        # print(key)
                        lin.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_lin.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_lin.append(pval)

                    elif key[5] == 'arm':
                        # print(key)
                        ar.append(key[3])
                        clus = int(key[4].split('_', 2)[1])
                        cluster_ar.append(clus)
                        k = int(key[0].split('=', 2)[1])
                        n = int(key[1].split('=', 2)[1])
                        p = float(key[2].split('=', 2)[1])
                        pval = 1 - binom.cdf(k, n, p)
                        cdfp_ar.append(pval)

                cdfp_dict_inner = {}
                for name, u in zip(
                    ['class', 'tissue', 'lineage', 'arm'], [
                        [clss, cluster_clss, cdfp_clss],
                        [tis, cluster_tis, cdfp_tis],
                        [lin, cluster_lin, cdfp_lin],
                        [ar, cluster_ar, cdfp_ar]]):

                    cdfp_frame = pd.DataFrame(u)
                    cdfp_frame = cdfp_frame.T
                    cdfp_frame.rename(
                        columns={0: 'classifier', 1: 'cluster', 2: 'pval'},
                        inplace=True)

                    cdfp_frame_threshold = cdfp_frame[
                        cdfp_frame['pval'] <= 0.05].sort_values(by='pval')

                    cdfp_dict_inner[name] = cdfp_frame_threshold
                dict_name2[key2] = cdfp_dict_inner

                os.chdir(key_dir)
                po_mapped_clusters = open(
                    'mapped_clusters.pickle', 'wb')
                pickle.dump(mapped_clusters, po_mapped_clusters)
                po_mapped_clusters.close()

                po_cdfp_dict_inner = open(
                    'cdfp_dict_inner.pickle', 'wb')
                pickle.dump(cdfp_dict_inner, po_cdfp_dict_inner)
                po_cdfp_dict_inner.close()

                pprint.pprint(cdfp_dict_inner, width=1)
                print()
            print()

        if 'dropped' in correlation_matrix:

            name = correlation_matrix + '_final'
            po_name = 'po_' + correlation_matrix + '_final'

            os.chdir(correlation_pickle_dir)
            po_name = open(name + '.pickle', 'wb')
            name = correlation_matrix_dict_copy
            pickle.dump(name, po_name)
            po_name.close()

        else:
            name = correlation_matrix + '_zeroed'
            po_name = 'po_' + correlation_matrix + '_zeroed'

            os.chdir(correlation_pickle_dir)
            po_name = open(name + '.pickle', 'wb')
            name = correlation_matrix_dict_copy
            pickle.dump(name, po_name)
            po_name.close()

        po_dict_name = 'po_' + dict_name1

        po_dict_name = open(dict_name1 + '.pickle', 'wb')
        dict_name1 = dict_name2
        pickle.dump(dict_name1, po_dict_name)
        po_dict_name.close()


plot_clustermap('filtered_subset_corrs', 'experimental', save_dir)


# drop tissue-specific immunophenotypes (columns) whose row values were set to
# 0.0 after correlation threw NaN for correlation pairs whose
# percentage values were initially set to zero because at least one fell below
# the percentage cutoff
def drop_non_correlations(zeroed_correlation_matrix):
    banner('RUNNING MODULE: drop_non_correlations')

    if zeroed_correlation_matrix == 'unfiltered_subset_corrs_zeroed':

        os.chdir(correlation_pickle_dir)
        pi_unfiltered_subset_corrs_zeroed = open(
            'unfiltered_subset_corrs_zeroed.pickle', 'rb')
        unfiltered_subset_corrs_zeroed = pickle.load(
            pi_unfiltered_subset_corrs_zeroed)

        unfiltered_subset_corrs_dropped = copy.deepcopy(
            unfiltered_subset_corrs_zeroed)

        key1_list = []
        for key1, value1 in unfiltered_subset_corrs_dropped.items():
            key1_list.append(key1)

        kwd = 'experimental_data'
        if any(kwd in x for x in key1_list) is True:

            print(color.CYAN + 'Dropping all complete zero correlations from'
                  ' the unfiltered ' + key1 + '.' + color.END)

            unfiltered_subset_corrs_dropped[
                key1 + '_dropped'] = unfiltered_subset_corrs_dropped.pop(key1)

            for key1, value1 in unfiltered_subset_corrs_dropped.items():
                for key2, value2 in value1.items():
                    for (idx, row), col in zip(value2.iterrows(), value2):

                        if all(v == 0.0 for v in row):
                            value2.drop(idx, axis=0, inplace=True)

                        if all(v == 0.0 for v in value2[col].values):
                            value2.drop(col, axis=1, inplace=True)
            print()

            os.chdir(correlation_pickle_dir)
            po_unfiltered_subset_corrs_dropped = open(
                'unfiltered_subset_corrs_dropped.pickle', 'wb')
            pickle.dump(unfiltered_subset_corrs_dropped,
                        po_unfiltered_subset_corrs_dropped)
            po_unfiltered_subset_corrs_dropped.close()

    elif zeroed_correlation_matrix == 'filtered_subset_corrs_zeroed':

        os.chdir(correlation_pickle_dir)
        pi_filtered_subset_corrs_zeroed = open(
            'filtered_subset_corrs_zeroed.pickle', 'rb')
        filtered_subset_corrs_zeroed = pickle.load(
            pi_filtered_subset_corrs_zeroed)

        filtered_subset_corrs_dropped = copy.deepcopy(
            filtered_subset_corrs_zeroed)

        key1_list = []
        for key1, value1 in filtered_subset_corrs_dropped.items():
            key1_list.append(key1)

        kwd = 'experimental_data'
        if any(kwd in x for x in key1_list) is True:

            print(color.CYAN + 'Dropping all complete zero correlations from'
                  ' the filtered ' + key1 + '.' + color.END)

            filtered_subset_corrs_dropped[
                key1 + '_dropped'] = filtered_subset_corrs_dropped.pop(key1)

            for key1, value1 in filtered_subset_corrs_dropped.items():
                for key2, value2 in value1.items():
                    for (idx, row), col in zip(value2.iterrows(), value2):

                        if all(v == 0.0 for v in row):
                            value2.drop(idx, axis=0, inplace=True)

                        if all(v == 0.0 for v in value2[col].values):
                            value2.drop(col, axis=1, inplace=True)
            print()

            os.chdir(correlation_pickle_dir)
            po_filtered_subset_corrs_dropped = open(
                'filtered_subset_corrs_dropped.pickle', 'wb')
            pickle.dump(filtered_subset_corrs_dropped,
                        po_filtered_subset_corrs_dropped)
            po_filtered_subset_corrs_dropped.close()

    elif zeroed_correlation_matrix == 'unfiltered_subset_corrs_combo_zeroed':

        os.chdir(correlation_pickle_dir)
        pi_unfiltered_subset_corrs_combo_zeroed = open(
            'unfiltered_subset_corrs_combo_zeroed.pickle', 'rb')
        unfiltered_subset_corrs_combo_zeroed = pickle.load(
            pi_unfiltered_subset_corrs_combo_zeroed)

        unfiltered_subset_corrs_combo_dropped = copy.deepcopy(
            unfiltered_subset_corrs_combo_zeroed)

        key1_list = []
        for key1, value1 in unfiltered_subset_corrs_combo_dropped.items():
            key1_list.append(key1)

        kwd = 'experimental_data'
        if any(kwd in x for x in key1_list) is True:

            print(color.CYAN + 'Dropping all complete zero correlations from'
                  ' the unfiltered ' + key1 + '.' + color.END)

            unfiltered_subset_corrs_combo_dropped[
                key1 + '_dropped'] = unfiltered_subset_corrs_combo_dropped.pop(
                    key1)

            for key1, value1 in unfiltered_subset_corrs_combo_dropped.items():
                for key2, value2 in value1.items():
                    for (idx, row), col in zip(value2.iterrows(), value2):

                        if all(v == 0.0 for v in row):
                            value2.drop(idx, axis=0, inplace=True)

                        if all(v == 0.0 for v in value2[col].values):
                            value2.drop(col, axis=1, inplace=True)
            print()

            os.chdir(correlation_pickle_dir)
            po_unfiltered_subset_corrs_combo_dropped = open(
                'unfiltered_subset_corrs_combo_dropped.pickle', 'wb')
            pickle.dump(unfiltered_subset_corrs_combo_dropped,
                        po_unfiltered_subset_corrs_combo_dropped)
            po_unfiltered_subset_corrs_combo_dropped.close()


drop_non_correlations('filtered_subset_corrs_zeroed')


plot_clustermap('filtered_subset_corrs_dropped', 'experimental', save_dir)


def silhouette_analysis(correlation_matrix, num_clust, cutoff, condition, file_path):

    banner('RUNNING MODULE: silhouette_analysis')

    print(condition)

    clustermaps = os.path.join(file_path, 'clustermaps')

    if 'dropped' in correlation_matrix:
        silhouette_dir = os.path.join(
            clustermaps, 'experimental_data_dropped', condition, 'silhouette')
        os.makedirs(silhouette_dir)

    else:
        silhouette_dir = os.path.join(
            clustermaps, 'experimental_data', condition, 'silhouette')
        os.makedirs(silhouette_dir)

    pi_name = 'pi' + correlation_matrix

    os.chdir(correlation_pickle_dir)
    pi_name = open(correlation_matrix + '.pickle', 'rb')
    correlation_matrix = pickle.load(
        pi_name)

    # print(color.CYAN + 'Performing silhouette analysis on the ' + condition +
    #       ' condition.' + color.END)

    copy_of_dict = copy.deepcopy(correlation_matrix)

    for key1, value1 in copy_of_dict.items():

        for key2, value2 in value1.items():
            if key2 == condition:

                dist_matrix = squareform(pdist(value2, metric='euclidean'))

                e = sns.clustermap(
                    value2, method='average', metric='euclidean',
                    xticklabels=1, yticklabels=1)
                plt.close('all')

                linkage_row = e.dendrogram_row.linkage

                range_n_clusters = []
                for threshold in np.arange(0, max(linkage_row[:, 2]), .001):

                    # print(key2, 'linkage threshold=' + str(threshold))

                    cluster_labels = fcluster(
                        linkage_row, threshold, criterion='distance')
                    n_clusters = len(set(cluster_labels))
                    tup = (threshold, n_clusters)
                    range_n_clusters.append(tup)

                range_n_clusters = [i for i in range_n_clusters
                                    if i[1] <= num_clust]

                range_n_clusters = sorted(range_n_clusters)

                cluster_set = set([x[1] for x in range_n_clusters])

                trunc_thresholds = []
                for v in cluster_set:
                    element = max([i for i in range_n_clusters if i[1] == v])
                    trunc_thresholds.append(element)
                trunc_thresholds = sorted(trunc_thresholds, reverse=False)

                sns.set(style='whitegrid')
                fig, axarr = plt.subplots(
                    2, 4, sharex=False, sharey=False, figsize=(10, 8))

                fig.subplots_adjust(left=0.05, right=0.95, bottom=0.06,
                                    top=0.92, hspace=0.3, wspace=0.4)

                fig.suptitle('silhouette analysis on hierarchical '
                             'agglomerative clustering', y=0.98, weight='bold')

                coordinates = list(itertools.product(range(4), repeat=2))
                dif = len(coordinates) - len(trunc_thresholds)

                if dif > 0:
                    coordinates = coordinates[:-dif]
                else:
                    pass

                neg_dict = {}
                for tup, n_clusters in itertools.zip_longest(
                  coordinates, trunc_thresholds):

                    sns.set(style='whitegrid')

                    axarr[tup[0], tup[1]].set_xlim([-1, 1])

                    # The (n_clusters+1) * 10 is for inserting blank space
                    # between silhouette plots of individual clusters
                    # to demarcate them clearly.
                    axarr[tup[0], tup[1]].set_ylim(
                        [0, len(dist_matrix) + (n_clusters[1] + 1) * 10])

                    cluster_labels = fcluster(
                        linkage_row, n_clusters[0], criterion='distance')
                    # print(n_clusters[1], cluster_labels)

                    silhouette_avg = metrics.silhouette_score(
                        dist_matrix, cluster_labels, metric='precomputed')
                    # print('For n_clusters =', n_clusters[1],
                    #       'The average silhouette_score is :', silhouette_avg)

                    sample_silhouette_values = metrics.silhouette_samples(
                        dist_matrix, cluster_labels)

                    series = pd.Series(
                        sample_silhouette_values, index=cluster_labels)
                    neg_list = []
                    for i in series.index.unique():
                        clust_data = series[series.index == i]

                        percent_neg = len(
                            clust_data[
                                clust_data < 0.0]) / \
                            len(clust_data)
                        neg_list.append(percent_neg)

                    neg_dict[n_clusters[0]] = neg_list

                    y_lower = 10
                    for i in range(n_clusters[1]):

                        ith_clust_silhouette_values = sample_silhouette_values[
                            cluster_labels == i+1]

                        ith_clust_silhouette_values.sort()

                        size_cluster_i = ith_clust_silhouette_values.shape[0]

                        y_upper = y_lower + size_cluster_i

                        color = cm.nipy_spectral(float(i)/n_clusters[1])

                        axarr[tup[0], tup[1]].fill_betweenx(
                            np.arange(y_lower, y_upper), 0,
                            ith_clust_silhouette_values, facecolor=color,
                            edgecolor=color, alpha=0.7)

                        axarr[tup[0], tup[1]].text(
                            -1.02, y_lower + 0.5 * size_cluster_i, str(i+1),
                            ha='right', color=color)

                        y_lower = y_upper + 10

                    axarr[tup[0], tup[1]].set_title(
                        'threshold = ' + str(n_clusters[0]) +
                        ', num_clust = ' + str(n_clusters[1]),
                        size=7, weight='bold')

                    axarr[tup[0], tup[1]].set_xlabel(
                        'silhouette coefficient',
                        size=8, weight='bold')
                    axarr[tup[0], tup[1]].set_ylabel(
                        'clusters', labelpad=20, size=12, weight='bold')

                    axarr[tup[0], tup[1]].axvline(
                        x=silhouette_avg, color='k',
                        linestyle="--", linewidth=0.75)
                    axarr[tup[0], tup[1]].text(
                        silhouette_avg, 0, 'ave: ' +
                        str(round(silhouette_avg, 2)),
                        rotation=0)

                    axarr[tup[0], tup[1]].axvline(
                        x=0, color='dimgrey', linestyle="-", linewidth=0.75)

                    axarr[tup[0], tup[1]].xaxis.set_minor_locator(
                        AutoMinorLocator())
                    axarr[tup[0], tup[1]].tick_params(
                        axis='x', which='minor', length=2.5,
                        color='k', direction='in')
                    axarr[tup[0], tup[1]].set_yticks([])
                    axarr[tup[0], tup[1]].grid(
                        color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
                    axarr[tup[0], tup[1]].xaxis.grid(False)

                plt.savefig(os.path.join(
                    silhouette_dir, 'silhouette_scores' + '.pdf'))
                plt.close('all')

                #######

                # get row index components lists
                idx_full = value2.index.tolist()
                idx_cell = [i.split('_', 2)[0] for i in idx_full]
                idx_tissue = [i.split('_', 2)[1] for i in idx_full]

                # generate class index from full celltype index
                idx_major_classes = []
                for longer in idx_cell:
                    if any(substring in longer for
                           substring in class_list) is True:
                        x = ''.join(
                            [sub for sub in class_list if sub in longer])
                        idx_major_classes.append(x)

                for i, cell in enumerate(idx_major_classes):
                    if cell == 'BNK':
                        idx_major_classes[i] = 'NK'
                    elif cell == 'BCD8T':
                        idx_major_classes[i] = 'CD8T'
                    elif cell == 'BMac':
                        idx_major_classes[i] = 'Mac'
                    elif cell == 'BISPT':
                        idx_major_classes[i] = 'ISPT'

                # get different row colors
                classes = pd.Series(idx_major_classes)
                classes.name = 'class'
                row_colors_classes = classes.map(class_lut)
                row_colors_classes.index = idx_full

                arm = pd.Series(idx_major_classes)
                arm.name = 'arm'
                row_colors_arm = arm.map(arm_lut)
                row_colors_arm.index = idx_full

                lineage = pd.Series(idx_cell)
                lineage.name = 'lineage'
                row_colors_lineage = lineage.map(lineage_lut)
                row_colors_lineage.index = idx_full

                tissue = pd.Series(idx_tissue)
                tissue.name = 'tissue'
                row_colors_tissue = tissue.map(color_dict)
                row_colors_tissue.index = idx_full

                # combine all row color types
                row_colors_combo = pd.DataFrame()
                row_colors_combo['arm'] = row_colors_arm
                row_colors_combo['lineage'] = row_colors_lineage
                row_colors_combo['tissue'] = row_colors_tissue
                row_colors_combo['class'] = row_colors_classes

                e = sns.clustermap(
                    value2, method='average', metric='euclidean',
                    xticklabels=1, yticklabels=1)
                plt.close('all')

                # set clustermap linkage threshold such that the fraction of
                # each cluster with negative silhouette values is < cutoff
                threshold = sorted(
                    {k for k, v in neg_dict.items() if max(v) < cutoff})[0]

                # get cluster assignments and
                # generate mapped_clusters dataframe
                clusters = fcluster(
                    linkage_row, threshold, criterion='distance')

                mapped_clusters = pd.DataFrame(
                    {'condition': value2.index.tolist(),
                     'cluster': clusters})

                mapped_clusters['class'] = idx_major_classes
                mapped_clusters['tissue'] = tissue
                mapped_clusters['arm'] = mapped_clusters['class'].map(
                    arm_dict)
                mapped_clusters['lineage'] = lineage.map(lineage_dict)

                # rearrange column order in row_colors_combo
                row_colors_combo.columns = [
                    'arm', 'lineage', 'tissue', 'class']

                # generate cluster_lut whose length is
                # proportional to the number of identified clusters
                c_list = ['r', 'g', 'b', 'y', 'm', 'c']
                c_list = c_list * int(
                    (math.ceil(max(clusters)/len(c_list))))
                idx_list = list(range(len(c_list)))
                idx_list = [i+1 for i in idx_list]
                cluster_lut_col = dict(zip(idx_list, c_list))

                # get cluster row color and add to row_colors_combo
                cluster = pd.Series(mapped_clusters['cluster'])
                cluster.name = 'cluster'
                row_cluster_colors = cluster.map(cluster_lut_col)

                row_full_update = [i.replace('neg', '$^-$').replace(
                    'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1') for i in idx_full]

                row_cluster_colors.index = row_full_update

                row_cluster_colors_dict = row_cluster_colors.to_dict()

                row_colors_combo['cluster'] = row_cluster_colors

                # plot dendrogram
                dend = dendrogram(
                    linkage_row, color_threshold=threshold,
                    above_threshold_color='grey',
                    leaf_font_size=7, labels=value2.index)
                plt.close('all')

                leaf_colors = get_linkage_colors(dend)

                leaf_colors_long = {}
                for c in leaf_colors.keys():
                    next_color = dict(
                        zip(leaf_colors[c], [c] * len(leaf_colors[c])))
                    leaf_colors_long.update(next_color)

                # change default color scheme
                for key, value in leaf_colors_long.items():
                    if value == 'r':
                        leaf_colors_long[key] = 'g'
                    elif value == 'g':
                        leaf_colors_long[key] = 'r'
                    elif value == 'c':
                        leaf_colors_long[key] = 'b'
                    elif value == 'm':
                        leaf_colors_long[key] = 'y'
                    elif value == 'y':
                        leaf_colors_long[key] = 'm'
                    elif value == 'k':
                        leaf_colors_long[key] = 'c'

                leaf_indices = [
                    list(x) for x in zip(dend['ivl'], dend['leaves'])]
                leaf_indices_frame = pd.DataFrame(leaf_indices)
                leaf_indices_frame.rename(
                    columns={0: 'leaf', 1: 'index'}, inplace=True)

                leaf_colors_frame = pd.DataFrame(
                    list(leaf_colors_long.items()))
                leaf_colors_frame.rename(
                    columns={0: 'leaf', 1: 'color'}, inplace=True)

                color_index = leaf_indices_frame.merge(
                    leaf_colors_frame, how='inner', on='leaf')

                c = color_index['color'].tolist()
                index = color_index['index'].tolist()
                leaf = ['leaf'] * len(index)

                prefix_frame = pd.DataFrame(
                    {'prefix': leaf, 'index': index})
                prefix_frame['index'] = prefix_frame['index'].astype(str)
                prefix_frame['prefix_index'] = prefix_frame[
                    'prefix'].map(str) + '_' + prefix_frame['index']

                link_color_index = prefix_frame['prefix_index'].tolist()
                link_color_dict = dict(zip(link_color_index, c))

                dflt_color = 'grey'
                link_colors = {}
                for i, i12 in enumerate(linkage_row[:, :2].astype(int)):
                    c1, c2 = (link_colors[x] if x > len(linkage_row) else
                              link_color_dict['leaf_%d' % x]
                              for x in i12)
                    link_colors[i + 1 +
                                len(linkage_row)
                                ] = c1 if c1 == c2 else dflt_color

                dend_labels = [
                    i.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    for i in value2.index.tolist()]

                dend_plot = dendrogram(
                    linkage_row, color_threshold=None,
                    link_color_func=lambda x: link_colors[x],
                    leaf_font_size=2, labels=dend_labels)

                dend_plot = plt.gcf()
                ax = plt.gca()
                ax.set_facecolor('white')
                ax.grid(color='grey', linestyle='-',
                        linewidth=0.5, alpha=0.5)
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)

                new_key_list = []
                value_list = []
                for key, value in leaf_colors_long.items():

                    new_key = key.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')
                    new_key_list.append(new_key)
                    value_list.append(value)

                leaf_colors_long_newkeys = dict(
                    zip(new_key_list, value_list))

                for item in ax.get_xticklabels():
                    name = item.get_text()
                    item.set_color(leaf_colors_long_newkeys[name])

                plt.tight_layout()
                dend_plot.savefig(
                    os.path.join(silhouette_dir, 'dendrogram' + '.pdf'))

                value2 = value2.copy()

                xlabels = [i.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1') for i in value2.columns]
                value2.columns = xlabels

                ylabels = [i.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1') for i in value2.index]
                value2.index = ylabels

                # if status included
                if idx_full[0].count('_') == 2:

                    idx_status = [i.split('_', 3)[2] for i in idx_full]
                    status = pd.Series(idx_status)
                    status.name = 'status'
                    status_dict = {'naive': 'k', 'gl261': 'gray'}
                    row_colors_status = status.map(status_dict)
                    row_colors_status.index = idx_full

                    row_colors_combo = pd.DataFrame()
                    row_colors_combo['status'] = row_colors_status

                    g = sns.clustermap(
                        value2, method='average', metric='euclidean',
                        row_colors=row_cluster_colors,
                        cmap='coolwarm', center=0.0, xticklabels=1,
                        yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    # g.ax_heatmap.set_xlabel(
                    #     'immunophenotype' + '$_i$' + ' in ' +
                    #     'tissue' + '$_j$', fontsize=15, weight='bold')
                    #
                    # g.ax_heatmap.set_ylabel(
                    #     'immunophenotype' + '$_i$' + ' in ' +
                    #     'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.savefig(
                        os.path.join(
                            silhouette_dir, 'silo_clustermap' + '.pdf'))
                    plt.close('all')

                else:

                    g = sns.clustermap(
                        value2, method='average', metric='euclidean',
                        row_colors=row_cluster_colors,
                        cmap='coolwarm', center=0.0, xticklabels=1,
                        yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(2)
                        name = item.get_text()
                        item.set_color(row_cluster_colors_dict[name])

                    # g.ax_heatmap.set_xlabel(
                    #     'immunophenotype' + '$_i$' + ' in ' +
                    #     'tissue' + '$_j$', fontsize=15, weight='bold')
                    #
                    # g.ax_heatmap.set_ylabel(
                    #     'immunophenotype' + '$_i$' + ' in ' +
                    #     'tissue' + '$_j$', fontsize=15, weight='bold')

                    g.savefig(
                        os.path.join(
                            silhouette_dir, 'silo_clustermap' + '.pdf'))
                    plt.close('all')

                    # binom.cdf(k, n, p)
                    cdf_dict = {}

                    for col in mapped_clusters[
                      ['class', 'tissue', 'arm', 'lineage']]:

                        p = mapped_clusters.groupby(col).size()/len(
                            mapped_clusters)
                        n = mapped_clusters.groupby(['cluster']).size()
                        k = mapped_clusters.pivot_table(
                            index='cluster', columns=col, values='condition',
                            fill_value=0, aggfunc='count').unstack()

                        for i, cluster in zip(range(len(n)), n):
                            clust_idx = n[n == cluster].index[0]
                            for j, idx, percent in zip(
                              range(len(p)), p.index, p):
                                x_list = []
                                y_list = []
                                pval_list = []
                                for x in range(cluster+1):
                                    x_list.append(x)
                                    y_list.append(
                                        binom.cdf(
                                            k=x, n=cluster, p=percent))
                                    pval_list.append(
                                        1-binom.cdf(
                                            k=x, n=cluster, p=percent))

                                data = pd.DataFrame(
                                    OrderedDict({'cluster': clust_idx,
                                                 'ID': idx, 'k': x_list,
                                                 'F(k)': y_list,
                                                 '1-F(k)': pval_list}))

                                cdf_dict[
                                    'k=' + str(k[idx][clust_idx]), 'n=' +
                                    str(n[clust_idx]), 'p=' + str(p[idx]),
                                    idx, 'cluster_' +
                                         str(clust_idx), col] = data

                    # get cluster enrichments
                    clss = []
                    cluster_clss = []
                    cdfp_clss = []

                    tis = []
                    cluster_tis = []
                    cdfp_tis = []

                    lin = []
                    cluster_lin = []
                    cdfp_lin = []

                    ar = []
                    cluster_ar = []
                    cdfp_ar = []

                    for key, value in cdf_dict.items():

                        if key[5] == 'class':
                            # print(key)
                            clss.append(key[3])
                            clus = int(key[4].split('_', 2)[1])
                            cluster_clss.append(clus)
                            k = int(key[0].split('=', 2)[1])
                            n = int(key[1].split('=', 2)[1])
                            p = float(key[2].split('=', 2)[1])
                            pval = 1 - binom.cdf(k, n, p)
                            cdfp_clss.append(pval)

                        elif key[5] == 'tissue':
                            # print(key)
                            tis.append(key[3])
                            clus = int(key[4].split('_', 2)[1])
                            cluster_tis.append(clus)
                            k = int(key[0].split('=', 2)[1])
                            n = int(key[1].split('=', 2)[1])
                            p = float(key[2].split('=', 2)[1])
                            pval = 1 - binom.cdf(k, n, p)
                            cdfp_tis.append(pval)

                        elif key[5] == 'lineage':
                            # print(key)
                            lin.append(key[3])
                            clus = int(key[4].split('_', 2)[1])
                            cluster_lin.append(clus)
                            k = int(key[0].split('=', 2)[1])
                            n = int(key[1].split('=', 2)[1])
                            p = float(key[2].split('=', 2)[1])
                            pval = 1 - binom.cdf(k, n, p)
                            cdfp_lin.append(pval)

                        elif key[5] == 'arm':
                            # print(key)
                            ar.append(key[3])
                            clus = int(key[4].split('_', 2)[1])
                            cluster_ar.append(clus)
                            k = int(key[0].split('=', 2)[1])
                            n = int(key[1].split('=', 2)[1])
                            p = float(key[2].split('=', 2)[1])
                            pval = 1 - binom.cdf(k, n, p)
                            cdfp_ar.append(pval)

                    cdfp_dict_inner = {}
                    for name, u in zip(
                        ['class', 'tissue', 'lineage', 'arm'], [
                            [clss, cluster_clss, cdfp_clss],
                            [tis, cluster_tis, cdfp_tis],
                            [lin, cluster_lin, cdfp_lin],
                            [ar, cluster_ar, cdfp_ar]]):

                        cdfp_frame = pd.DataFrame(u)
                        cdfp_frame = cdfp_frame.T
                        cdfp_frame.rename(
                            columns={0: 'classifier', 1: 'cluster', 2: 'pval'},
                            inplace=True)

                        cdfp_frame_threshold = cdfp_frame[
                            cdfp_frame['pval'] <= 0.05].sort_values(by='pval')

                        cdfp_dict_inner[name] = cdfp_frame_threshold

                    os.chdir(silhouette_dir)
                    po_mapped_clusters = open(
                        'mapped_clusters.pickle', 'wb')
                    pickle.dump(mapped_clusters, po_mapped_clusters)
                    po_mapped_clusters.close()

                    po_cdfp_dict_inner = open(
                        'cdfp_dict_inner.pickle', 'wb')
                    pickle.dump(cdfp_dict_inner, po_cdfp_dict_inner)
                    po_cdfp_dict_inner.close()

                    pprint.pprint(cdfp_dict_inner, width=1)
                    print()
                print()


for condition in ['naive_filtered', 'gl261_filtered']:
    silhouette_analysis(
        'filtered_subset_corrs_dropped', 9, 0.6, condition, save_dir)


# def cluster_alignment(correlation_matrix):
#     banner('RUNNING MODULE: cluster_alignment')
#
#     pi_name = 'pi' + correlation_matrix
#
#     os.chdir(correlation_pickle_dir)
#     pi_name = open(correlation_matrix + '.pickle', 'rb')
#     correlation_matrix_dict = pickle.load(
#         pi_name)
#
#     if os.path.isdir(
#       save_dir + 'cluster_alignment') is False:
#
#         cluster_alignment = os.path.join(
#             save_dir, 'cluster_alignment')
#         os.makedirs(cluster_alignment)
#
#     else:
#         cluster_alignment = os.path.join(
#             save_dir, 'cluster_alignment')
#
#     cluster_subsets = {}
#     for key1, value1 in correlation_matrix_dict.items():
#
#         mapped_clusters_dict = {}
#         row_colors_cluster_dict = {}
#         axis_dict = {}
#         leaf_colors_long_dict = {}
#         matrix_dict = {}
#
#         for key2, value2 in value1.items():
#             if key2 in ['naive_filtered', 'gl261_filtered']:
#
#                 status = key2.split('_', 2)[0]
#
#                 print(status)
#
#                 value2_copy = value2.copy()
#
#                 value2_copy.replace(to_replace=np.nan, value=0.0, inplace=True)
#
#                 # get row index components lists
#                 idx_full = value2_copy.index.tolist()
#                 idx_cell = [i.split('_', 2)[0] for i in idx_full]
#                 idx_tissue = [i.split('_', 2)[1] for i in idx_full]
#
#                 # generate class index from idx_cell
#                 idx_class = []
#                 for celltype in idx_cell:
#                     if any(substring in celltype for
#                            substring in class_list) is True:
#                         x = ''.join(
#                             [sub for sub in class_list if sub in celltype])
#                         idx_class.append(x)
#
#                 # correct for B cell class
#                 for i, clss in enumerate(idx_class):
#                     if clss == 'BNK':
#                         idx_class[i] = 'NK'
#                     elif clss == 'BCD8T':
#                         idx_class[i] = 'CD8T'
#                     elif clss == 'BMac':
#                         idx_class[i] = 'Mac'
#                     elif clss == 'BISPT':
#                         idx_class[i] = 'ISPT'
#
#                 # get class row colors
#                 classes = pd.Series(idx_class)
#                 classes.name = 'class'
#                 row_colors_classes = classes.map(class_lut)
#                 row_colors_classes.index = idx_full
#
#                 # get arm row colors
#                 arm = pd.Series(idx_class)
#                 arm.name = 'arm'
#                 row_colors_arm = arm.map(arm_lut)
#                 row_colors_arm.index = idx_full
#
#                 # get lineage row colors
#                 lineage = pd.Series(idx_cell)
#                 lineage.name = 'lineage'
#                 row_colors_lineage = lineage.map(lineage_lut)
#                 row_colors_lineage.index = idx_full
#
#                 # get tissue row colors
#                 tissue = pd.Series(idx_tissue)
#                 tissue.name = 'tissue'
#                 row_colors_tissue = tissue.map(color_dict)
#                 row_colors_tissue.index = idx_full
#
#                 # combine all row color types
#                 row_colors_combo = pd.DataFrame()
#                 row_colors_combo['arm'] = row_colors_arm
#                 row_colors_combo['lineage'] = row_colors_lineage
#                 row_colors_combo['tissue'] = row_colors_tissue
#                 row_colors_combo['class'] = row_colors_classes
#
#                 e = sns.clustermap(value2_copy, xticklabels=1, yticklabels=1)
#                 plt.close('all')
#
#                 # get row linkage
#                 linkage_row = e.dendrogram_row.linkage
#
#                 # assign the default dendrogram
#                 # distance threshold = 0.7*max(Z[:, 2])
#                 threshold = 0.7*max(linkage_row[:, 2])
#
#                 # plot dendrogram
#                 dend = dendrogram(
#                     linkage_row, color_threshold=threshold,
#                     above_threshold_color='grey',
#                     leaf_font_size=7, labels=value2_copy.index)
#                 plt.close('all')
#
#                 leaf_colors = get_linkage_colors(dend)
#
#                 leaf_colors_long = {}
#                 for c in leaf_colors.keys():
#                     next_color = dict(
#                         zip(leaf_colors[c], [c] * len(leaf_colors[c])))
#                     leaf_colors_long.update(next_color)
#
#                 # change default leaf color scheme
#                 for key, value in leaf_colors_long.items():
#                     if value == 'r':
#                         leaf_colors_long[key] = 'g'
#                     elif value == 'g':
#                         leaf_colors_long[key] = 'r'
#                     elif value == 'c':
#                         leaf_colors_long[key] = 'b'
#                     elif value == 'm':
#                         leaf_colors_long[key] = 'y'
#                     elif value == 'y':
#                         leaf_colors_long[key] = 'm'
#                     elif value == 'k':
#                         leaf_colors_long[key] = 'c'
#
#                 leaf_colors_long_dict[status] = leaf_colors_long
#
#                 leaf_indices = [
#                     list(x) for x in zip(dend['ivl'], dend['leaves'])]
#                 leaf_indices_frame = pd.DataFrame(leaf_indices,
#                                                   columns=['leaf', 'index'])
#
#                 leaf_colors_frame = pd.DataFrame(
#                     list(leaf_colors_long.items()), columns=['leaf', 'colors'])
#
#                 color_index = leaf_indices_frame.merge(
#                     leaf_colors_frame, how='inner', on='leaf')
#
#                 c = color_index['colors'].tolist()
#                 index = color_index['index'].tolist()
#                 leaf = ['leaf'] * len(index)
#
#                 prefix_frame = pd.DataFrame({'prefix': leaf, 'index': index})
#                 prefix_frame['index'] = prefix_frame['index'].astype(str)
#                 prefix_frame['prefix_index'] = prefix_frame[
#                     'prefix'].map(str) + '_' + prefix_frame['index']
#
#                 link_color_index = prefix_frame['prefix_index'].tolist()
#                 link_color_dict = dict(zip(link_color_index, c))
#
#                 dflt_color = 'grey'
#                 link_colors = {}
#                 for i, i12 in enumerate(linkage_row[:, :2].astype(int)):
#                     c1, c2 = (link_colors[x] if x > len(linkage_row) else
#                               link_color_dict['leaf_%d' % x]
#                               for x in i12)
#                     link_colors[
#                         i+1+len(linkage_row)] = c1 if c1 == c2 else dflt_color
#
#                 # get cluster assignments and
#                 # generate mapped_clusters dataframe
#                 clusters = fcluster(
#                     linkage_row, threshold, criterion='distance')
#
#                 mapped_clusters = pd.DataFrame(
#                     {'condition': value2_copy.index.tolist(),
#                      'cluster': clusters})
#
#                 mapped_clusters['class'] = idx_class
#                 mapped_clusters['tissue'] = tissue
#                 mapped_clusters['arm'] = mapped_clusters['class'].map(arm_dict)
#                 mapped_clusters['lineage'] = lineage.map(lineage_dict)
#
#                 mapped_clusters_dict[status] = mapped_clusters
#
#                 # generate cluster_lut whose length is
#                 # proportional to the number of identified clusters
#                 c_list = ['r', 'g', 'b', 'y', 'm', 'c']
#                 c_list = c_list * int((math.ceil(max(clusters)/len(c_list))))
#                 idx_list = list(range(len(c_list)))
#                 idx_list = [i+1 for i in idx_list]
#                 cluster_lut = dict(zip(idx_list, c_list))
#
#                 # get cluster row colors
#                 cluster = pd.Series(mapped_clusters['cluster'])
#                 cluster.name = 'cluster'
#                 row_colors_cluster = cluster.map(cluster_lut)
#                 row_colors_cluster.index = idx_full
#
#                 row_colors_cluster_dict[status] = row_colors_cluster
#
#                 g = sns.clustermap(
#                     value2_copy, row_colors=row_colors_cluster,
#                     cmap='cividis', center=0, xticklabels=1, yticklabels=1)
#
#                 new_key_list = []
#                 value_list = []
#
#                 for key, value in leaf_colors_long.items():
#
#                     new_key = key.replace(
#                         'neg', '$^-$').replace('pos', '$^+$').replace(
#                         'CD8a', 'CD8' + u'\u03B1')
#
#                     new_key_list.append(new_key)
#                     value_list.append(value)
#
#                 leaf_colors_long_newkeys = dict(
#                     zip(new_key_list, value_list))
#
#                 xlabels = [
#                     item.get_text() for item in
#                     g.ax_heatmap.get_xticklabels()]
#
#                 xlabels_update = [xlabel.replace(
#                     'neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]
#
#                 g.ax_heatmap.set_xticklabels(xlabels_update)
#
#                 for item in g.ax_heatmap.get_xticklabels():
#                     item.set_rotation(90)
#                     item.set_size(2)
#                     name = item.get_text()
#                     item.set_color(leaf_colors_long_newkeys[name])
#
#                 ylabels = [
#                     item.get_text() for item in
#                     g.ax_heatmap.get_yticklabels()]
#
#                 ylabels_update = [ylabel.replace(
#                     'neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1') for ylabel in ylabels]
#
#                 g.ax_heatmap.set_yticklabels(ylabels_update)
#
#                 for item in g.ax_heatmap.get_yticklabels():
#                     item.set_rotation(0)
#                     item.set_size(2)
#                     name = item.get_text()
#                     item.set_color(leaf_colors_long_newkeys[name])
#
#                 g.ax_heatmap.set_xlabel(
#                     'immunophenotype' + '$_i$' + ' in ' +
#                     'tissue' + '$_j$', fontsize=15, weight='bold')
#
#                 g.ax_heatmap.set_ylabel(
#                     'immunophenotype' + '$_i$' + ' in ' +
#                     'tissue' + '$_j$', fontsize=15, weight='bold')
#
#                 axis_dict[status] = g
#
#                 value2_copy_copy = value2_copy.copy()
#
#                 sorted_value2_copy_copy = value2_copy_copy.reindex(
#                     index=xlabels, columns=xlabels)
#
#                 matrix_dict[status] = sorted_value2_copy_copy
#
#                 g.savefig(os.path.join(cluster_alignment, status + '.pdf'))
#                 plt.close('all')
#
#     # enforce opposite clustering
#     for key, value in axis_dict.items():
#
#         # naive clustering enforced on gl261 data
#         if key == 'naive':
#
#             naive_ax = value
#
#             naive_idx = [
#                 item.get_text() for item in
#                 naive_ax.ax_heatmap.get_xticklabels()]
#
#             value3 = correlation_matrix_dict[
#                 'experimental_data']['gl261_filtered'].copy()
#
#             value3.replace(to_replace=np.nan, value=0.0, inplace=True)
#
#             value3.index = [
#                 i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1') for i in value3.index]
#
#             value3.columns = [
#                 i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1') for i in value3.columns]
#
#             row_colors_cluster_dict['gl261'].index = [
#                 i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1')
#                 for i in row_colors_cluster_dict['gl261'].index]
#
#             sorted_value3 = value3.reindex(
#                 index=naive_idx, columns=naive_idx)
#
#             sorted_value3_copy = sorted_value3.copy()
#
#             matrix_dict[key + '_on_gl261'] = sorted_value3_copy
#
#             matrix_dict[key + '_on_gl261'].index = [
#                 i.replace('$^-$', 'neg').replace('$^+$', 'pos').replace(
#                     'CD8' + u'\u03B1', 'CD8a')
#                 for i in matrix_dict[key + '_on_gl261'].index]
#
#             matrix_dict[key + '_on_gl261'].columns = [
#                 i.replace('$^-$', 'neg').replace('$^+$', 'pos').replace(
#                     'CD8' + u'\u03B1', 'CD8a')
#                 for i in matrix_dict[key + '_on_gl261'].columns]
#
#             ng = sns.clustermap(
#                 sorted_value3, row_cluster=False, col_cluster=False,
#                 row_colors=row_colors_cluster_dict['gl261'],
#                 cmap='cividis', center=0, xticklabels=1, yticklabels=1)
#
#             new_key_list = []
#             value_list = []
#             for key, value in leaf_colors_long_dict['naive'].items():
#
#                 new_key = key.replace(
#                     'neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1')
#                 new_key_list.append(new_key)
#                 value_list.append(value)
#
#             leaf_colors_long_newkeys = dict(
#                 zip(new_key_list, value_list))
#
#             for item in ng.ax_heatmap.get_xticklabels():
#                 item.set_rotation(90)
#                 item.set_size(2)
#                 name = item.get_text()
#                 item.set_color(leaf_colors_long_newkeys[name])
#
#             for item in ng.ax_heatmap.get_yticklabels():
#                 item.set_rotation(0)
#                 item.set_size(2)
#                 name = item.get_text()
#                 item.set_color(leaf_colors_long_newkeys[name])
#
#             ng.ax_heatmap.set_xlabel(
#                 'immunophenotype' + '$_i$' + ' in ' +
#                 'tissue' + '$_j$', fontsize=15, weight='bold')
#
#             ng.ax_heatmap.set_ylabel(
#                 'immunophenotype' + '$_i$' + ' in ' +
#                 'tissue' + '$_j$', fontsize=15, weight='bold')
#
#             ng.savefig(os.path.join(
#                 cluster_alignment, 'naive_on_gl261' + '.pdf'))
#             plt.close('all')
#
#         # gl261 clustering enforced on naive data
#         elif key == 'gl261':
#
#             gl261_ax = value
#
#             gl261_idx = [
#                 item.get_text() for item in
#                 gl261_ax.ax_heatmap.get_xticklabels()]
#
#             value3 = correlation_matrix_dict[
#                 'experimental_data']['naive_filtered'].copy()
#
#             value3.replace(to_replace=np.nan, value=0.0, inplace=True)
#
#             value3.index = [
#                 i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1') for i in value3.index]
#
#             value3.columns = [
#                 i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1') for i in value3.columns]
#
#             row_colors_cluster_dict['naive'].index = [
#                 i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1')
#                 for i in row_colors_cluster_dict['naive'].index]
#
#             sorted_value3 = value3.reindex(
#                 index=gl261_idx, columns=gl261_idx)
#
#             sorted_value3_copy = sorted_value3.copy()
#
#             matrix_dict[key + '_on_naive'] = sorted_value3_copy
#
#             matrix_dict[key + '_on_naive'].index = [
#                 i.replace('$^-$', 'neg').replace('$^+$', 'pos').replace(
#                     'CD8' + u'\u03B1', 'CD8a')
#                 for i in matrix_dict[key + '_on_naive'].index]
#
#             matrix_dict[key + '_on_naive'].columns = [
#                 i.replace('$^-$', 'neg').replace('$^+$', 'pos').replace(
#                     'CD8' + u'\u03B1', 'CD8a')
#                 for i in matrix_dict[key + '_on_naive'].columns]
#
#             gn = sns.clustermap(
#                 sorted_value3, row_cluster=False, col_cluster=False,
#                 row_colors=row_colors_cluster_dict['naive'], cmap='cividis',
#                 xticklabels=1, yticklabels=1)
#
#             new_key_list = []
#             value_list = []
#             for key, value in leaf_colors_long_dict['gl261'].items():
#
#                 new_key = key.replace(
#                     'neg', '$^-$').replace('pos', '$^+$').replace(
#                     'CD8a', 'CD8' + u'\u03B1')
#                 new_key_list.append(new_key)
#                 value_list.append(value)
#
#             leaf_colors_long_newkeys = dict(
#                 zip(new_key_list, value_list))
#
#             for item in gn.ax_heatmap.get_xticklabels():
#                 item.set_rotation(90)
#                 item.set_size(2)
#                 name = item.get_text()
#                 item.set_color(leaf_colors_long_newkeys[name])
#
#             for item in gn.ax_heatmap.get_yticklabels():
#                 item.set_rotation(0)
#                 item.set_size(2)
#                 name = item.get_text()
#                 item.set_color(leaf_colors_long_newkeys[name])
#
#             gn.ax_heatmap.set_xlabel(
#                 'immunophenotype' + '$_i$' + ' in ' +
#                 'tissue' + '$_j$', fontsize=15, weight='bold')
#
#             gn.ax_heatmap.set_ylabel(
#                 'immunophenotype' + '$_i$' + ' in ' +
#                 'tissue' + '$_j$', fontsize=15, weight='bold')
#
#             gn.savefig(os.path.join(
#                 cluster_alignment, 'gl261_on_naive' + '.pdf'))
#             plt.close('all')
#
#     # get alignment matrix
#     cluster_subsets = {}
#     for key, value in mapped_clusters_dict.items():
#         for cluster in set(value['cluster']):
#             subset = mapped_clusters_dict[key][
#                 mapped_clusters_dict[key]['cluster'] == cluster]
#             cluster_subsets[key[0] + str(cluster)] = subset
#
#     naive_conds = [i for i in cluster_subsets.keys() if 'n' in i]
#     gl261_conds = [i for i in cluster_subsets.keys() if 'g' in i]
#
#     matrix = pd.DataFrame(columns=['naive_cond', 'gl261_cond', 'value'])
#     for i in naive_conds:
#         for j in gl261_conds:
#             value = len(
#                 cluster_subsets[i].merge(
#                     cluster_subsets[j], how='inner',
#                     on='condition')) / len(cluster_subsets[i])
#
#             matrix = matrix.append(
#                 {'naive': i, 'gl261': j, 'value': value},
#                 ignore_index=True)
#
#     matrix_pivot = matrix.pivot_table(
#         index='naive', columns='gl261', values='value')
#
#     fig, ax = plt.subplots(figsize=(5, 5))
#
#     fmt = lambda x, pos: '{:.1%}'.format(x)
#
#     sns.heatmap(matrix_pivot, square=True, linewidth=1.5,
#                 annot=True, fmt='.1%', cmap="YlGnBu",
#                 cbar_kws={'format': FuncFormatter(fmt)})
#
#     rg_series = pd.Series(gl261_idx)
#     rg_series_map = rg_series.map(row_colors_cluster_dict['gl261'])
#     color_list_g = sorted(rg_series_map.unique())
#     cluster_list_g = list(range(1, len(color_list_g) + 1))
#     cluster_color_dict_g = dict(zip(cluster_list_g, color_list_g))
#
#     xlabels = [
#         item.get_text() for item in ax.get_xticklabels()]
#
#     xlabels_update = [
#         xlabel + '(' + cluster_color_dict_g[int(xlabel[1])] + ')'
#         for xlabel in xlabels]
#
#     ax.set_xticklabels(xlabels_update)
#
#     rn_series = pd.Series(naive_idx)
#     rn_series_map = rn_series.map(row_colors_cluster_dict['naive'])
#     color_list_n = sorted(rn_series_map.unique())
#     cluster_list_n = list(range(1, len(color_list_n) + 1))
#     cluster_color_dict_n = dict(zip(cluster_list_n, color_list_n))
#
#     ylabels = [
#         item.get_text() for item in ax.get_yticklabels()]
#
#     ylabels_update = [
#         ylabel + '(' + cluster_color_dict_n[int(ylabel[1])] + ')'
#         for ylabel in ylabels]
#
#     ax.set_yticklabels(ylabels_update)
#
#     for item in ax.get_xticklabels():
#         item.set_rotation(0)
#         item.set_size(10)
#         item.set_weight('bold')
#         name = item.get_text()
#         if 'r' in name:
#             item.set_color('r')
#         elif 'b' in name:
#             item.set_color('b')
#         elif 'g' in name:
#             item.set_color('g')
#         elif 'y' in name:
#             item.set_color('y')
#         elif 'm' in name:
#             item.set_color('m')
#         elif 'c' in name:
#             item.set_color('c')
#
#     for item in ax.get_yticklabels():
#         item.set_rotation(0)
#         item.set_size(10)
#         item.set_weight('bold')
#         name = item.get_text()
#         if 'r' in name:
#             item.set_color('r')
#         elif 'b' in name:
#             item.set_color('b')
#         elif 'g' in name:
#             item.set_color('g')
#         elif 'y' in name:
#             item.set_color('y')
#         elif 'm' in name:
#             item.set_color('m')
#         elif 'c' in name:
#             item.set_color('c')
#
#     ax.set_xlabel('gl261 clusters', weight='bold', labelpad=10)
#     ax.set_ylabel('naive clusters', weight='bold', labelpad=10)
#
#     plt.savefig(os.path.join(cluster_alignment, 'alignment_matrix' + '.pdf'))
#     plt.close('all')
#
#     # filter naive clustering on gl261 data
#     naive_on_gl261_filtered = matrix_dict['naive_on_gl261'].copy()
#
#     for row1, row2 in zip(matrix_dict['naive'].iterrows(),
#                           matrix_dict['naive_on_gl261'].iterrows()):
#         print(row1[0], row2[0])
#         for idx, (i, j) in enumerate(zip(row1[1], row2[1])):
#             print(i, j)
#             if not (i-0.7) <= j <= (i+0.7):
#                 pass
#             else:
#                 naive_on_gl261_filtered.set_value(
#                     row2[1].index[idx], row2[0], 0)
#
#     naive_on_gl261_filtered.index = [
#         i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#             'CD8a', 'CD8' + u'\u03B1') for i in naive_on_gl261_filtered.index]
#
#     naive_on_gl261_filtered.columns = [
#         i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#             'CD8a', 'CD8' + u'\u03B1')
#         for i in naive_on_gl261_filtered.columns]
#
#     gn = sns.clustermap(
#         naive_on_gl261_filtered, row_cluster=False, col_cluster=False,
#         row_colors=row_colors_cluster_dict['gl261'],
#         xticklabels=1, yticklabels=1)
#
#     new_key_list = []
#     value_list = []
#     for key, value in leaf_colors_long_dict['naive'].items():
#
#         new_key = key.replace(
#             'neg', '$^-$').replace('pos', '$^+$').replace(
#             'CD8a', 'CD8' + u'\u03B1')
#         new_key_list.append(new_key)
#         value_list.append(value)
#
#     leaf_colors_long_newkeys = dict(
#         zip(new_key_list, value_list))
#
#     for item in gn.ax_heatmap.get_xticklabels():
#         item.set_rotation(90)
#         item.set_size(2)
#         name = item.get_text()
#         item.set_color(leaf_colors_long_newkeys[name])
#
#     for item in gn.ax_heatmap.get_yticklabels():
#         item.set_rotation(0)
#         item.set_size(2)
#         name = item.get_text()
#         item.set_color(leaf_colors_long_newkeys[name])
#
#     gn.ax_heatmap.set_xlabel(
#         'immunophenotype' + '$_i$' + ' in ' +
#         'tissue' + '$_j$', fontsize=15, weight='bold')
#
#     gn.ax_heatmap.set_ylabel(
#         'immunophenotype' + '$_i$' + ' in ' +
#         'tissue' + '$_j$', fontsize=15, weight='bold')
#
#     gn.savefig(os.path.join(
#         cluster_alignment, 'naive_on_gl261_filtered' + '.pdf'))
#     plt.close('all')
#
#     # get misclustered gl261 conditions
#     naive_on_gl261_filtered_binarized = naive_on_gl261_filtered.copy()
#     for row in naive_on_gl261_filtered_binarized.iterrows():
#         for idx, i in enumerate(row[1]):
#             if abs(i) > 0.0:
#                 naive_on_gl261_filtered_binarized.set_value(
#                     row[1].index[idx], row[0], 1)
#             else:
#                 pass
#
#     counts = naive_on_gl261_filtered_binarized.astype(bool).sum(axis=0)
#     counts.drop([i for i in counts.index if 'unspecified' in i], inplace=True)
#     counts1 = counts.sort_values(ascending=False)
#     counts2 = pd.DataFrame(counts1)
#     counts2.rename(columns={0: 'count'}, inplace=True)
#
#     sns.set(style='whitegrid')
#     fig, ax = plt.subplots()
#     sns.barplot(x=counts2['count'], y=counts2.index, palette='Greys_r')
#
#     for item in ax.get_xticklabels():
#         item.set_weight('bold')
#
#     for item in ax.get_yticklabels():
#         item.set_rotation(0)
#         item.set_size(1)
#         item.set_weight('bold')
#         name = item.get_text()
#         item.set_color(leaf_colors_long_newkeys[name])
#     ax.grid(color='grey', linestyle='--',
#             linewidth=0.5, alpha=0.5)
#
#     for spine in ax.spines.values():
#         spine.set_edgecolor('k')
#
#     ax.xaxis.grid(True)
#     ax.yaxis.grid(False)
#     ax.xaxis.set_minor_locator(
#         AutoMinorLocator())
#     ax.tick_params(
#         axis='x', which='minor', length=2.5,
#         color='k', direction='in')
#     ax.set_xlabel('count', size=15, weight='bold')
#     plt.savefig(os.path.join(
#         cluster_alignment, 'naive_on_gl261_filtered_counts' + '.pdf'))
#     plt.close('all')
#
#     # get jointplot
#     rows = list(range(len(naive_on_gl261_filtered_binarized.index)))
#     rows = [i + 1 for i in rows]
#     rows_rev = []
#     for i in reversed(rows):
#         rows_rev.append(i)
#     naive_on_gl261_filtered_binarized.index = rows_rev
#
#     cols = list(range(len(naive_on_gl261_filtered_binarized.columns)))
#     cols = [i + 1 for i in cols]
#     naive_on_gl261_filtered_binarized.columns = cols
#
#     jointplot_input = naive_on_gl261_filtered_binarized.unstack()[
#         naive_on_gl261_filtered_binarized.unstack() == 1.0].reset_index()
#
#     sns.set(style='whitegrid')
#     fig, ax = plt.subplots()
#     g = sns.jointplot('level_0', 'level_1', data=jointplot_input,
#                       color='k', size=10,
#                       marginal_kws=dict(bins=170, rug=False),
#                       annot_kws=dict(stat='r'),
#                       s=10, edgecolor='None', linewidth=1)
#
#     plt.savefig(os.path.join(
#         cluster_alignment, 'naive_on_gl261_filtered_joint' + '.pdf'))
#     plt.close('all')
#
#     # filter gl261 clustering on naive data
#     gl261_on_naive_filtered = matrix_dict['gl261_on_naive'].copy()
#     for row1, row2 in zip(matrix_dict['gl261'].iterrows(),
#                           matrix_dict['gl261_on_naive'].iterrows()):
#         print(row1[0], row2[0])
#         for idx, (i, j) in enumerate(zip(row1[1], row2[1])):
#             print(i, j)
#             if not (i-0.7) <= j <= (i+0.7):
#                 pass
#             else:
#                 gl261_on_naive_filtered.set_value(
#                     row2[1].index[idx], row2[0], 0)
#
#     gl261_on_naive_filtered.index = [
#         i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#             'CD8a', 'CD8' + u'\u03B1') for i in gl261_on_naive_filtered.index]
#
#     gl261_on_naive_filtered.columns = [
#         i.replace('neg', '$^-$').replace('pos', '$^+$').replace(
#             'CD8a', 'CD8' + u'\u03B1')
#         for i in gl261_on_naive_filtered.columns]
#
#     gn = sns.clustermap(
#         gl261_on_naive_filtered, row_cluster=False, col_cluster=False,
#         row_colors=row_colors_cluster_dict['naive'],
#         xticklabels=1, yticklabels=1)
#
#     new_key_list = []
#     value_list = []
#     for key, value in leaf_colors_long_dict['gl261'].items():
#
#         new_key = key.replace(
#             'neg', '$^-$').replace('pos', '$^+$').replace(
#             'CD8a', 'CD8' + u'\u03B1')
#         new_key_list.append(new_key)
#         value_list.append(value)
#
#     leaf_colors_long_newkeys = dict(
#         zip(new_key_list, value_list))
#
#     for item in gn.ax_heatmap.get_xticklabels():
#         item.set_rotation(90)
#         item.set_size(2)
#         name = item.get_text()
#         item.set_color(leaf_colors_long_newkeys[name])
#
#     for item in gn.ax_heatmap.get_yticklabels():
#         item.set_rotation(0)
#         item.set_size(2)
#         name = item.get_text()
#         item.set_color(leaf_colors_long_newkeys[name])
#
#     gn.ax_heatmap.set_xlabel(
#         'immunophenotype' + '$_i$' + ' in ' +
#         'tissue' + '$_j$', fontsize=15, weight='bold')
#
#     gn.ax_heatmap.set_ylabel(
#         'immunophenotype' + '$_i$' + ' in ' +
#         'tissue' + '$_j$', fontsize=15, weight='bold')
#
#     gn.savefig(os.path.join(
#         cluster_alignment, 'gl261_on_naive_filtered' + '.pdf'))
#     plt.close('all')
#
#     # get misclustered naive conditions
#     gl261_on_naive_filtered_binarized = gl261_on_naive_filtered.copy()
#     for row in gl261_on_naive_filtered_binarized.iterrows():
#         for idx, i in enumerate(row[1]):
#             if abs(i) > 0.0:
#                 gl261_on_naive_filtered_binarized.set_value(
#                     row[1].index[idx], row[0], 1)
#             else:
#                 pass
#
#     counts = gl261_on_naive_filtered_binarized.astype(bool).sum(axis=0)
#     counts.drop([i for i in counts.index if 'unspecified' in i], inplace=True)
#     counts = counts.sort_values(ascending=False)
#     counts = pd.DataFrame(counts)
#     counts.rename(columns={0: 'count'}, inplace=True)
#
#     sns.set(style='whitegrid')
#     fig, ax = plt.subplots()
#     sns.barplot(x=counts['count'], y=counts.index, palette='Greys_r')
#
#     for item in ax.get_xticklabels():
#         item.set_weight('bold')
#
#     for item in ax.get_yticklabels():
#         item.set_rotation(0)
#         item.set_size(1)
#         item.set_weight('bold')
#         name = item.get_text()
#         item.set_color(leaf_colors_long_newkeys[name])
#     ax.grid(color='grey', linestyle='--',
#             linewidth=0.5, alpha=0.5)
#
#     for spine in ax.spines.values():
#         spine.set_edgecolor('k')
#
#     ax.xaxis.grid(True)
#     ax.yaxis.grid(False)
#     ax.xaxis.set_minor_locator(
#         AutoMinorLocator())
#     ax.tick_params(
#         axis='x', which='minor', length=2.5,
#         color='k', direction='in')
#     ax.set_xlabel('count', size=15, weight='bold')
#     plt.savefig(os.path.join(
#         cluster_alignment, 'gl261_on_naive_filtered_counts' + '.pdf'))
#     plt.close('all')
#
#
# cluster_alignment('filtered_subset_corrs')


# generate tables of significant correlation pairs
# (unfiltered and filtered)
def sig_corr_tables(sig_corr_dict):
    banner('RUNNING MODULE: sig_corr_tables')

    os.chdir(correlation_pickle_dir)

    pi_filtered_sig_corr_dict = open(
        'filtered_sig_corr_dict.pickle', 'rb')
    filtered_sig_corr_dict = pickle.load(
        pi_filtered_sig_corr_dict)

    print(color.CYAN + 'Saving all significant correlation tables for'
          ' the filtered data.' + color.END)

    tables = os.path.join(save_dir, 'significant_corrs_tables')

    if os.path.isdir(tables) is True:
        pass

    else:
        os.makedirs(tables)

    os.chdir(tables)
    filtered_sig = {}
    for key, value in filtered_sig_corr_dict.items():

        if not value.empty:
            if os.path.isdir(key) is True:
                pass

            else:
                os.mkdir(key)

            value_df = pd.DataFrame(value).reset_index(drop=False)
            value_df.columns = ['x', 'y', u'\u03C1', 'two_tailed_qval']
            value_df[u'\u03C1'] = value_df[u'\u03C1'].round(decimals=4)
            value_df['x'] = value_df['x'].str.replace('pos', u'\u207A')
            value_df['x'] = value_df['x'].str.replace('neg', u'\u207B')
            value_df['x'] = value_df['x'].str.replace(
                'CD8a', 'CD8' + u'\u03B1')
            value_df['y'] = value_df['y'].str.replace('pos', u'\u207A')
            value_df['y'] = value_df['y'].str.replace('neg', u'\u207B')
            value_df['y'] = value_df['y'].str.replace(
                'CD8a', 'CD8' + u'\u03B1')

            os.chdir(key)

            value_df.to_csv(
                key + '.csv', index=False, encoding='utf-8-sig')

            os.chdir(tables)

        if value.empty:
            if os.path.isdir(key) is True:
                pass

            else:
                os.mkdir(key)

            value_df = pd.DataFrame(
                index=[0],
                columns=['x', 'y', u'\u03C1', 'two_tailed_qval'])
            value_df['x'] = 'empty'
            value_df['y'] = 'empty'
            value_df[u'\u03C1'] = 'empty'

            os.chdir(key)

            value_df.to_csv(
                key + '.csv', index=False, encoding='utf-8-sig')

            os.chdir(tables)

        filtered_sig[key] = value_df

    os.chdir(correlation_pickle_dir)
    po_filtered_sig = open(
        'filtered_sig.pickle', 'wb')
    pickle.dump(filtered_sig,
                po_filtered_sig)
    po_filtered_sig.close()


sig_corr_tables('filtered_sig_corr_dict')


# generate tables of significant correlations shared between or
# unique to a given status
# (unfiltered and filtered)
def table_dif():
    banner('RUNNING MODULE: table_dif')

    os.chdir(correlation_pickle_dir)

    pi_filtered_sig_corr_dict = open(
        'filtered_sig_corr_dict.pickle', 'rb')
    filtered_sig_corr_dict = pickle.load(
        pi_filtered_sig_corr_dict)

    pi_filtered_subset_corrs = open('filtered_subset_corrs.pickle', 'rb')
    filtered_subset_corrs = pickle.load(pi_filtered_subset_corrs)

    print(color.CYAN + 'Splitting significant correlation tables by'
          ' status.' + color.END)

    tables = os.path.join(save_dir, 'significant_corrs_tables')

    def table_dif_inner(series_dict):

        # Sort the keys explicitly as dict ordering is not consistent.
        gl261_key, naive_key = sorted(series_dict.keys())
        name = gl261_key + ' vs. ' + naive_key

        if (series_dict[gl261_key].empty & series_dict[naive_key].empty):
            merge = pd.DataFrame(index=[0], columns=[0])
            merge[0] = 'both conditions were empty.'
            merge.name = name

            return merge.name, merge

        else:

            gl261_series_frame = pd.DataFrame(series_dict[gl261_key])
            gl261_series_frame.rename(columns={0: u'\u03C1'}, inplace=True)

            naive_series_frame = pd.DataFrame(series_dict[naive_key])
            naive_series_frame.rename(columns={0: u'\u03C1'}, inplace=True)

            merge = gl261_series_frame.merge(
                naive_series_frame, how='outer', left_index=True,
                right_index=True, suffixes=('_gl261', '_naive'),
                indicator=True)

            merge['index'] = merge.index
            merge.drop_duplicates()
            merge.drop('index', axis=1, inplace=True)
            merge.name = name

            return merge.name, merge

    for name, d in zip(['filtered_sig_corr_dict'],
                       [filtered_sig_corr_dict]):

        d = collections.OrderedDict(sorted(d.items()))

        dict_7_fil = {}
        dict_14_fil = {}
        dict_30_fil = {}
        dict_status_fil = {}

        for key, value in d.items():
            if 'gl261' in key:
                gl261_series = value
                if '7' in key:
                    dict_7_fil[key] = gl261_series
                elif '14' in key:
                    dict_14_fil[key] = gl261_series
                elif '30' in key:
                    dict_30_fil[key] = gl261_series
                else:
                    dict_status_fil[key] = gl261_series
            elif 'naive' in key:
                naive_series = value
                if '7' in key:
                    dict_7_fil[key] = naive_series
                elif '14' in key:
                    dict_14_fil[key] = naive_series
                elif '30' in key:
                    dict_30_fil[key] = naive_series
                else:
                    dict_status_fil[key] = naive_series

    master_list = [dict_7_fil, dict_14_fil, dict_30_fil, dict_status_fil]

    merge_results = {}

    # filter out dict pairs both of Empty DataFrame
    for i in master_list:
        boo_list = []
        for k in i.keys():
            boo_list.append(i[k].empty)
        if not set(boo_list) == {True}:

            keys = list(i.keys())
            # data = list(i.values())

            gl261_key, naive_key = sorted(keys)

            gl261_pops = set(i[gl261_key].index)
            naive_pops = set(i[naive_key].index)

            name = gl261_key + ' vs. ' + naive_key

            populations = list(gl261_pops | naive_pops)

            if 'filtered' in keys[0]:
                corrs = filtered_subset_corrs['experimental_data']

            result = pd.DataFrame({
                '\u03C1_gl261': corrs[gl261_key].stack().loc[populations],
                '\u03C1_naive': corrs[naive_key].stack().loc[populations],
            })
            result.loc[list(gl261_pops), '_merge'] = 'left_only'
            result.loc[list(naive_pops), '_merge'] = 'right_only'
            result.loc[list(gl261_pops & naive_pops), '_merge'] = 'both'

            result = result.merge(
                pd.DataFrame(
                    filtered_sig_corr_dict[gl261_key]['two_tailed_qval']),
                how='left', left_index=True, right_index=True)

            result.rename(
                columns={'two_tailed_qval': 'gl261_qval'}, inplace=True)

            result = result.merge(
                pd.DataFrame(
                    filtered_sig_corr_dict[naive_key]['two_tailed_qval']),
                how='left', left_index=True, right_index=True)

            result.rename(
                columns={'two_tailed_qval': 'naive_qval'}, inplace=True)

            merge_results[name] = result.sort_index()

    group_sig_dict = {}
    for key, value in merge_results.items():

        if 'both conditions were empty.' in value.values:
            group_sig_dict[key] = value

        else:
            value.reset_index(inplace=True)
            value.rename(
                columns={'level_0': 'x', 'level_1': 'y'}, inplace=True)
            value['_merge'].replace(
                to_replace={
                    'left_only': 'gl261_only', 'right_only': 'naive_only'},
                inplace=True)

            for name, group in value.groupby(['_merge']):

                key_update = key + ' ' + '(' + name + ')'

                if 'naive_only' in key_update:
                    group_copy = group.copy()
                    group_copy.sort_values(
                        by=u'\u03C1' + '_naive', inplace=True)
                    group_copy[u'\u03C1' + '_naive'] = group_copy[
                        u'\u03C1' + '_naive'].round(decimals=4)
                    group_copy['x'] = group_copy['x'].str.replace(
                        'pos', u'\u207A')
                    group_copy['x'] = group_copy['x'].str.replace(
                        'neg', u'\u207B')
                    group_copy['x'] = group_copy['x'].str.replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'pos', u'\u207A')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'neg', u'\u207B')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    group_copy['product'] = (
                        group_copy[u'\u03C1' + '_gl261'] *
                        group_copy[u'\u03C1' + '_naive'])
                    group_copy.sort_values(
                        by='product', ascending=False, inplace=True)
                    for r in group_copy.iterrows():
                        if math.isnan(r[1][u'\u03C1' + '_gl261']):
                            group_copy[u'\u03C1' + '_gl261'].replace(
                                to_replace=np.nan, value='missing_pop.',
                                inplace=True)

                elif 'gl261_only' in key_update:
                    group_copy = group.copy()
                    group_copy.sort_values(
                        by=u'\u03C1' + '_gl261', inplace=True)
                    group_copy[u'\u03C1' + '_gl261'] = group_copy[
                        u'\u03C1' + '_gl261'].round(decimals=4)
                    group_copy['x'] = group_copy['x'].str.replace(
                        'pos', u'\u207A')
                    group_copy['x'] = group_copy['x'].str.replace(
                        'neg', u'\u207B')
                    group_copy['x'] = group_copy['x'].str.replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'pos', u'\u207A')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'neg', u'\u207B')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    group_copy['product'] = (
                        group_copy[u'\u03C1' + '_gl261'] *
                        group_copy[u'\u03C1' + '_naive'])
                    group_copy.sort_values(
                        by='product', ascending=False, inplace=True)
                    for r in group_copy.iterrows():
                        if math.isnan(r[1][u'\u03C1' + '_naive']):
                            group_copy[u'\u03C1' + '_naive'].replace(
                                to_replace=np.nan, value='missing_pop.',
                                inplace=True)

                elif 'both' in key_update:
                    group_copy = group.copy()
                    group_copy.sort_values(
                        by=u'\u03C1' + '_gl261', inplace=True)
                    group_copy[u'\u03C1' + '_gl261'] = group_copy[
                        u'\u03C1' + '_gl261'].round(decimals=4)
                    group_copy[u'\u03C1' + '_naive'] = group_copy[
                        u'\u03C1' + '_naive'].round(decimals=4)
                    group_copy['x'] = group_copy['x'].str.replace(
                        'pos', u'\u207A')
                    group_copy['x'] = group_copy['x'].str.replace(
                        'neg', u'\u207B')
                    group_copy['x'] = group_copy['x'].str.replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'pos', u'\u207A')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'neg', u'\u207B')
                    group_copy['y'] = group_copy['y'].str.replace(
                        'CD8a', 'CD8' + u'\u03B1')
                    group_copy['product'] = (
                        group_copy[u'\u03C1' + '_gl261'] *
                        group_copy[u'\u03C1' + '_naive'])
                    group_copy.sort_values(
                        by='product', ascending=False, inplace=True)

                group_sig_dict[key_update] = group_copy

                os.chdir(tables)

                if os.path.isdir(key_update) is True:
                    pass

                else:
                    os.mkdir(key_update)

                os.chdir(key_update)
                group_copy.to_csv(key_update + '.csv', index=False,
                                  encoding='utf-8-sig')
                os.chdir(tables)

    os.chdir(correlation_pickle_dir)
    po_group_sig_dict = open(
        'group_sig_dict.pickle', 'wb')
    pickle.dump(group_sig_dict, po_group_sig_dict)
    po_group_sig_dict.close()


table_dif()


# plot all significant correlations
def plot_correlations(pivot_subsets, sig_difs):

    banner('RUNNING MODULE: plot_correlations')

    pi_name1 = 'pi' + pivot_subsets

    os.chdir(correlation_pickle_dir)
    pi_name1 = open(pivot_subsets + '.pickle', 'rb')
    pivot_subsets_dict = pickle.load(
        pi_name1)

    pi_name2 = 'pi' + sig_difs

    pi_name2 = open(sig_difs + '.pickle', 'rb')
    sig_difs_dict = pickle.load(
        pi_name2)

    print('Plotting regression plots.')

    dict_map = {
        'gl261_7_unfiltered vs. naive_7_unfiltered (naive_only)':
            'naive_7_unfiltered',
        'gl261_7_unfiltered vs. naive_7_unfiltered (gl261_only)':
            'gl261_7_unfiltered',
        'gl261_7_unfiltered vs. naive_7_unfiltered (both)':
            'naive_7_unfiltered',

        'gl261_14_unfiltered vs. naive_14_unfiltered (naive_only)':
            'naive_14_unfiltered',
        'gl261_14_unfiltered vs. naive_14_unfiltered (gl261_only)':
            'gl261_14_unfiltered',
        'gl261_14_unfiltered vs. naive_14_unfiltered (both)':
            'naive_14_unfiltered',

        'gl261_30_unfiltered vs. naive_30_unfiltered (naive_only)':
            'naive_30_unfiltered',
        'gl261_30_unfiltered vs. naive_30_unfiltered (gl261_only)':
            'gl261_30_unfiltered',
        'gl261_30_unfiltered vs. naive_30_unfiltered (both)':
            'naive_30_unfiltered',

        'gl261_unfiltered vs. naive_unfiltered (naive_only)':
            'naive_unfiltered',
        'gl261_unfiltered vs. naive_unfiltered (gl261_only)':
            'gl261_unfiltered',
        'gl261_unfiltered vs. naive_unfiltered (both)':
            'naive_unfiltered',

        'gl261_7_filtered vs. naive_7_filtered (naive_only)':
            'naive_7_unfiltered',
        'gl261_7_filtered vs. naive_7_filtered (gl261_only)':
            'gl261_7_unfiltered',
        'gl261_7_filtered vs. naive_7_filtered (both)':
            'naive_7_unfiltered',

        'gl261_14_filtered vs. naive_14_filtered (naive_only)':
            'naive_14_unfiltered',
        'gl261_14_filtered vs. naive_14_filtered (gl261_only)':
            'gl261_14_unfiltered',
        'gl261_14_filtered vs. naive_14_filtered (both)':
            'naive_14_unfiltered',

        'gl261_30_filtered vs. naive_30_filtered (naive_only)':
            'naive_30_unfiltered',
        'gl261_30_filtered vs. naive_30_filtered (gl261_only)':
            'gl261_30_unfiltered',
        'gl261_30_filtered vs. naive_30_filtered (both)':
            'naive_30_unfiltered',

        'gl261_filtered vs. naive_filtered (naive_only)':
            'naive_unfiltered',
        'gl261_filtered vs. naive_filtered (gl261_only)':
            'gl261_unfiltered',
        'gl261_filtered vs. naive_filtered (both)':
            'naive_unfiltered'}

    for key, value in sig_difs_dict.items():

        if 'both conditions were empty.' not in value.values:

            scatter_plots = os.path.join(save_dir, 'scatter_plots')

            if os.path.isdir(scatter_plots) is True:
                pass

            else:
                os.makedirs(scatter_plots)

            os.chdir(scatter_plots)

            key_dir = os.path.join(scatter_plots, key)

            if os.path.isdir(key_dir) is True:
                pass

            else:
                os.makedirs(key_dir)

            linker = dict_map[key]

            value['x'] = value['x'].str.replace(
                u'\u207A', 'pos').str.replace(
                u'\u207B', 'neg').str.replace(
                'CD8' + u'\u03B1', 'CD8a')

            value['y'] = value['y'].str.replace(
                u'\u207A', 'pos').str.replace(
                u'\u207B', 'neg').str.replace(
                'CD8' + u'\u03B1', 'CD8a')

            for i in value.iterrows():

                if math.isnan(i[1]['product']) is False:

                    t_naive = []
                    t_gl261 = []

                    linker_dict = {'naive': 'gl261', 'gl261': 'naive'}
                    linker_type = linker.rsplit('_', 3)[0]
                    linker_alt = linker_dict[linker_type]

                    for t in [linker, linker.replace(linker_type, linker_alt)]:
                        if 'naive' in t:
                            cond1 = pivot_subsets_dict[
                                'experimental_data'][t][i[1][0]]
                            t_naive.append(cond1)

                            cond2 = pivot_subsets_dict[
                                'experimental_data'][t][i[1][1]]
                            t_naive.append(cond2)

                        elif 'gl261' in t:
                            cond1 = pivot_subsets_dict[
                                'experimental_data'][t][i[1][0]]
                            t_gl261.append(cond1)

                            cond2 = pivot_subsets_dict[
                                'experimental_data'][t][i[1][1]]
                            t_gl261.append(cond2)

                    data_dict = {}
                    for name, p in zip(
                      ['naive', 'gl261'], [t_naive, t_gl261]):
                        data = pd.concat(p, axis=1)
                        x_name = data.columns.tolist()[0]
                        y_name = data.columns.tolist()[1]

                        spearman_output = data.corr(
                            method='spearman',
                            min_periods=1).iloc[0:1, 1:2].values.round(4)
                        rho = str(spearman_output).strip('[]')

                        pearson_output, pval = pr(
                            list(data.iloc[:, 0]),
                            list(data.iloc[:, 1]))
                        r = str(round(pearson_output, 4))

                        data_dict[name] = [
                            data, x_name, y_name, rho, r]

                    sns.set(style='whitegrid')
                    fig = plt.figure(figsize=(16, 8))
                    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
                    ax1 = plt.subplot(gs[0])
                    ax1.grid(alpha=0.3, linestyle='--')
                    for side in ['top', 'bottom', 'left', 'right']:
                        ax1.spines[side].set_color('k')
                    ax2 = plt.subplot(gs[1])  # sharex=ax1, sharey=ax1
                    ax2.grid(alpha=0.3)
                    for side in ['top', 'bottom', 'left', 'right']:
                        ax2.spines[side].set_color('k')
                    ax1.set_title(
                        'naive', fontsize=19, color='b', fontweight='bold')
                    ax2.set_title(
                        'gl261', fontsize=19, color='g', fontweight='bold')

                    sns.regplot(
                        x=data_dict['naive'][1], y=data_dict['naive'][2],
                        data=data_dict['naive'][0], color='b', ax=ax1)

                    ax1.annotate(u'\u03C1' + ' = ' + data_dict['naive'][3] +
                                 '   ' + 'r' + ' = ' + data_dict['naive'][4],
                                 xy=(0, 0), xytext=(240, 450),
                                 xycoords='axes fraction',
                                 textcoords='offset points',
                                 size=15, weight='bold')

                    for label, x, y in zip(
                      data_dict['naive'][0].index,
                      data_dict['naive'][0][data_dict['naive'][1]],
                      data_dict['naive'][0][data_dict['naive'][2]]):
                        ax1.annotate(
                            'm' + label.split(',', 3)[2].replace(' ', '') +
                            '(d' + label.split(',', 3)[1].replace(' ', '')
                            + ')', xy=(x, y), xytext=(0, 0), size=9,
                            weight='bold', textcoords='offset points',
                            ha='right', va='bottom')

                    sns.regplot(
                        x=data_dict['gl261'][1], y=data_dict['gl261'][2],
                        data=data_dict['gl261'][0], color='g', ax=ax2)

                    ax2.set_xlabel('')
                    ax2.set_ylabel('')

                    ax2.annotate(u'\u03C1' + ' = ' + data_dict['gl261'][3] +
                                 '   ' + 'r' + ' = ' + data_dict['gl261'][4],
                                 xy=(0, 0), xytext=(240, 450),
                                 xycoords='axes fraction',
                                 textcoords='offset points',
                                 size=15, weight='bold', zorder=3)

                    for label, x, y in zip(
                      data_dict['gl261'][0].index,
                      data_dict['gl261'][0][data_dict['gl261'][1]],
                      data_dict['gl261'][0][data_dict['gl261'][2]]):
                        ax2.annotate(
                            'm' + label.split(',', 3)[2].replace(' ', '') +
                            '(d' + label.split(',', 3)[1].replace(' ', '')
                            + ')', xy=(x, y), xytext=(0, 0), size=9,
                            weight='bold', textcoords='offset points',
                            ha='right', va='bottom')

                    # x_values = np.append(
                    #     data_dict['naive'][0][
                    #         data_dict['naive'][1]].values,
                    #     data_dict['gl261'][0][
                    #         data_dict['gl261'][1]].values)

                    # y_values = np.append(
                    #     data_dict['naive'][0][
                    #         data_dict['naive'][2]].values,
                    #     data_dict['gl261'][0][
                    #         data_dict['gl261'][2]].values)

                    # n_axes = n_ax.axes

                    # n_axes.set_xlim(min(x_values), max(x_values))
                    # n_axes.set_ylim(min(y_values), max(y_values))

                    fig.canvas.draw()
                    for w in [ax1, ax2]:
                        for item in w.get_xticklabels():
                            item.set_rotation(0)
                            item.set_size(15)
                            item.set_fontweight('normal')
                            item.set_color('k')
                        for item in w.get_yticklabels():
                            item.set_size(15)
                            item.set_fontweight('normal')
                            item.set_color('k')

                        w.set_xticklabels([label.get_text() + '%' for
                                           label in w.get_xticklabels()])

                        w.set_yticklabels([label.get_text() + '%' for
                                           label in w.get_yticklabels()])

                    xlabel = ax1.get_xlabel()
                    ylabel = ax1.get_ylabel()
                    ax1.set_xlabel(xlabel.replace(
                        'neg', '$^-$').replace('pos', '$^+$'), fontsize=23)
                    ax1.set_ylabel(ylabel.replace(
                        'neg', '$^-$').replace('pos', '$^+$'), fontsize=23)

                    ax1.xaxis.set_label_coords(1.1, -0.08)
                    ax1.yaxis.set_label_coords(-0.15, 0.5)

                    if '7' in key:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (7dpi)',
                                    fontsize=20, fontweight='bold', y=0.98)
                    elif '14' in key:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (14dpi)',
                                    fontsize=20, fontweight='bold', y=0.98)
                    elif '30' in key:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (30dpi)',
                                    fontsize=20, fontweight='bold', y=0.98)
                    else:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (all time points)',
                                    fontsize=20, fontweight='bold', y=0.98)

                        w.grid(True)

                    plt.savefig(os.path.join(
                        key_dir, i[1][0] + ' vs. ' + i[1][1] + '.pdf'))
                    plt.close('all')

                else:
                    status_color = {'naive': 'b', 'gl261': 'g'}

                    cond1 = pivot_subsets_dict[
                        'experimental_data'][linker][i[1][0]]

                    cond2 = pivot_subsets_dict[
                        'experimental_data'][linker][i[1][1]]

                    data = pd.concat([cond1, cond2], axis=1)
                    x_name = data.columns.tolist()[0]
                    y_name = data.columns.tolist()[1]

                    spearman_output = data.corr(
                        method='spearman',
                        min_periods=1).iloc[0:1, 1:2].values.round(4)
                    rho = str(spearman_output).strip('[]')

                    pearson_output, pval = pr(
                        list(data.iloc[:, 0]),
                        list(data.iloc[:, 1]))
                    r = str(round(pearson_output, 4))

                    sns.set(style='whitegrid')
                    fig = plt.figure(figsize=(16, 8))
                    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
                    ax1 = plt.subplot(gs[0])
                    ax1.grid(alpha=0.3, linestyle='--')
                    for side in ['top', 'bottom', 'left', 'right']:
                        ax1.spines[side].set_color('k')
                    ax1.set_title(
                        linker.split('_', 2)[0], fontsize=19,
                        color=status_color[linker.split('_', 2)[0]],
                        fontweight='bold')

                    sns.regplot(
                        x=x_name, y=y_name, data=data,
                        color=status_color[linker.split('_', 2)[0]],
                        ax=ax1)

                    ax1.annotate(u'\u03C1' + ' = ' + rho + '   ' + 'r' +
                                 ' = ' + r,
                                 xy=(0, 0), xytext=(240, 450),
                                 xycoords='axes fraction',
                                 textcoords='offset points',
                                 size=15, weight='bold')

                    for label, x, y in zip(
                      data.index, data[x_name], data[y_name]):

                        ax1.annotate(
                            'm' + label.split(',', 3)[2].replace(' ', '') +
                            '(d' + label.split(',', 3)[1].replace(' ', '')
                            + ')', xy=(x, y), xytext=(0, 0), size=9,
                            weight='bold', textcoords='offset points',
                            ha='right', va='bottom')

                    for w in [ax1]:
                        for item in w.get_xticklabels():
                            item.set_rotation(0)
                            item.set_size(15)
                            item.set_fontweight('normal')
                            item.set_color('k')
                        for item in w.get_yticklabels():
                            item.set_size(15)
                            item.set_fontweight('normal')
                            item.set_color('k')

                        fig.canvas.draw()
                        w.set_xticklabels([label.get_text() + '%' for
                                           label in w.get_xticklabels()])

                        w.set_yticklabels([label.get_text() + '%' for
                                           label in w.get_yticklabels()])

                    xlabel = ax1.get_xlabel()
                    ylabel = ax1.get_ylabel()
                    ax1.set_xlabel(xlabel.replace(
                        'neg', '$^-$').replace('pos', '$^+$'), fontsize=23)
                    ax1.set_ylabel(ylabel.replace(
                        'neg', '$^-$').replace('pos', '$^+$'), fontsize=23)

                    ax1.xaxis.set_label_coords(0.5, -0.08)
                    ax1.yaxis.set_label_coords(-0.12, 0.5)

                    if '7' in key:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (7dpi)',
                                    fontsize=20, fontweight='bold',
                                    x=0.30, y=0.98)
                    elif '14' in key:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (14dpi)',
                                    fontsize=20, fontweight='bold',
                                    x=0.30, y=0.98)
                    elif '30' in key:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (30dpi)',
                                    fontsize=20, fontweight='bold',
                                    x=0.30, y=0.98)
                    else:
                        fig.suptitle(xlabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' vs. ' + ylabel.replace(
                                'neg', '$^-$').replace(
                                    'pos', '$^+$') + ' (all time points)',
                                    fontsize=20, fontweight='bold',
                                    x=0.30, y=0.98)

                        w.grid(True)

                    plt.savefig(os.path.join(
                        key_dir, i[1][0] + ' vs. ' + i[1][1] + '.pdf'))
                    plt.close('all')


plot_correlations('unfiltered_pivot_subsets', 'group_sig_dict')


# plot arbitrary correlations
def plot_select_correlations(pop1, pop2):

    scatter_plots = os.path.join(save_dir, 'scatter_plots')

    b220_dir = os.path.join(
        scatter_plots, pop1.split('_')[0] + ' vs. ' + pop2.split('_')[0])
    if not os.path.exists:
        os.makedirs(b220_dir)

    os.chdir(correlation_pickle_dir)
    pi_unfiltered_pivot_subsets = open(
        'unfiltered_pivot_subsets' + '.pickle', 'rb')
    pivot_subsets_dict = pickle.load(pi_unfiltered_pivot_subsets)

    for cond in pivot_subsets_dict['experimental_data'].keys():
        if 'naive' in cond:
            if cond not in ['naive_unfiltered']:

                cond_dir = os.path.join(b220_dir, cond)

                if not os.path.exists:
                    os.makedirs(cond_dir)

                t_naive = []
                t_gl261 = []

                linker_dict = {'naive': 'gl261', 'gl261': 'naive'}
                linker_type = cond.rsplit('_', 3)[0]
                linker_alt = linker_dict[linker_type]

                for status in [cond, cond.replace(linker_type, linker_alt)]:

                    if 'naive' in status:
                        cond1 = pivot_subsets_dict[
                            'experimental_data'][status][pop1]
                        t_naive.append(cond1)

                        cond2 = pivot_subsets_dict[
                            'experimental_data'][status][pop2]
                        t_naive.append(cond2)

                    elif 'gl261' in status:
                        cond1 = pivot_subsets_dict[
                            'experimental_data'][status][pop1]
                        t_gl261.append(cond1)

                        cond2 = pivot_subsets_dict[
                            'experimental_data'][status][pop2]
                        t_gl261.append(cond2)

                data_dict = {}
                for name, p in zip(
                  ['naive', 'gl261'], [t_naive, t_gl261]):
                    data = pd.concat(p, axis=1)
                    x_name = data.columns.tolist()[0]
                    y_name = data.columns.tolist()[1]

                    spearman_output = data.corr(
                        method='spearman',
                        min_periods=1).iloc[0:1, 1:2].values.round(4)
                    rho = str(spearman_output).strip('[]')

                    pearson_output, pval = pr(
                        list(data.iloc[:, 0]),
                        list(data.iloc[:, 1]))
                    r = str(round(pearson_output, 4))

                    data_dict[name] = [
                        data, x_name, y_name, rho, r]

                sns.set(style='whitegrid')
                fig = plt.figure(figsize=(16, 8))
                gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
                ax1 = plt.subplot(gs[0])
                ax1.grid(alpha=0.3, linestyle='--')
                for side in ['top', 'bottom', 'left', 'right']:
                    ax1.spines[side].set_color('k')
                ax2 = plt.subplot(gs[1])
                ax2.grid(alpha=0.3)
                for side in ['top', 'bottom', 'left', 'right']:
                    ax2.spines[side].set_color('k')
                ax1.set_title(
                    'naive', fontsize=19, color='b', fontweight='bold')
                ax2.set_title(
                    'gl261', fontsize=19, color='g', fontweight='bold')

                sns.regplot(
                    x=data_dict['naive'][1], y=data_dict['naive'][2],
                    data=data_dict['naive'][0], color='b', ax=ax1)

                ax1.annotate(u'\u03C1' + ' = ' + data_dict['naive'][3] +
                             '   ' + 'r' + ' = ' + data_dict['naive'][4],
                             xy=(0, 0), xytext=(240, 450),
                             xycoords='axes fraction',
                             textcoords='offset points',
                             size=15, weight='bold')

                for label, x, y in zip(
                  data_dict['naive'][0].index,
                  data_dict['naive'][0][data_dict['naive'][1]],
                  data_dict['naive'][0][data_dict['naive'][2]]):
                    ax1.annotate(
                        'm' + label.split(',', 3)[2].replace(' ', '') +
                        '(d' + label.split(',', 3)[1].replace(' ', '')
                        + ')', xy=(x, y), xytext=(0, 0), size=9,
                        weight='bold', textcoords='offset points',
                        ha='right', va='bottom')

                sns.regplot(
                    x=data_dict['gl261'][1], y=data_dict['gl261'][2],
                    data=data_dict['gl261'][0], color='g', ax=ax2)

                ax2.set_xlabel('')
                ax2.set_ylabel('')

                ax2.annotate(u'\u03C1' + ' = ' + data_dict['gl261'][3] +
                             '   ' + 'r' + ' = ' + data_dict['gl261'][4],
                             xy=(0, 0), xytext=(240, 450),
                             xycoords='axes fraction',
                             textcoords='offset points',
                             size=15, weight='bold', zorder=3)

                for label, x, y in zip(
                  data_dict['gl261'][0].index,
                  data_dict['gl261'][0][data_dict['gl261'][1]],
                  data_dict['gl261'][0][data_dict['gl261'][2]]):
                    ax2.annotate(
                        'm' + label.split(',', 3)[2].replace(' ', '') +
                        '(d' + label.split(',', 3)[1].replace(' ', '')
                        + ')', xy=(x, y), xytext=(0, 0), size=9,
                        weight='bold', textcoords='offset points',
                        ha='right', va='bottom')

                fig.canvas.draw()
                for w in [ax1, ax2]:
                    for item in w.get_xticklabels():
                        item.set_rotation(0)
                        item.set_size(15)
                        item.set_fontweight('normal')
                        item.set_color('k')
                    for item in w.get_yticklabels():
                        item.set_size(15)
                        item.set_fontweight('normal')
                        item.set_color('k')

                    w.set_xticklabels([label.get_text() + '%' for
                                       label in w.get_xticklabels()])

                    w.set_yticklabels([label.get_text() + '%' for
                                       label in w.get_yticklabels()])

                xlabel = ax1.get_xlabel()
                ylabel = ax1.get_ylabel()
                ax1.set_xlabel(xlabel.replace(
                    'neg', '$^-$').replace('pos', '$^+$'), fontsize=23)
                ax1.set_ylabel(ylabel.replace(
                    'neg', '$^-$').replace('pos', '$^+$'), fontsize=23)

                ax1.xaxis.set_label_coords(1.1, -0.08)
                ax1.yaxis.set_label_coords(-0.15, 0.5)

                if '7' in cond:
                    fig.suptitle(xlabel.replace(
                        'neg', '$^-$').replace(
                            'pos', '$^+$') + ' vs. ' + ylabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' (7dpi)',
                                fontsize=20, fontweight='bold', y=0.98)
                elif '14' in cond:
                    fig.suptitle(xlabel.replace(
                        'neg', '$^-$').replace(
                            'pos', '$^+$') + ' vs. ' + ylabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' (14dpi)',
                                fontsize=20, fontweight='bold', y=0.98)
                elif '30' in cond:
                    fig.suptitle(xlabel.replace(
                        'neg', '$^-$').replace(
                            'pos', '$^+$') + ' vs. ' + ylabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' (30dpi)',
                                fontsize=20, fontweight='bold', y=0.98)
                else:
                    fig.suptitle(xlabel.replace(
                        'neg', '$^-$').replace(
                            'pos', '$^+$') + ' vs. ' + ylabel.replace(
                            'neg', '$^-$').replace(
                                'pos', '$^+$') + ' (all time points)',
                                fontsize=20, fontweight='bold', y=0.98)

                    w.grid(True)

                plt.savefig(os.path.join(
                    cond_dir, pop1 + ' vs. ' + pop2 + '.pdf'))

                plt.close('all')


plot_select_correlations('CD8T_spleen', 'B220posCD8T_spleen')


def sig_correlation_counts():

    banner('RUNNING MODULE: correlation_counts')

    scatter_dir = os.path.join(project_path, 'correlations', 'data',
                               'spearman_correlation_analysis',
                               'orthant', 'scatter_plots')

    os.chdir(correlation_pickle_dir)

    pi_group_sig_dict = open('group_sig_dict.pickle', 'rb')
    group_sig_dict = pickle.load(pi_group_sig_dict)

    cond_list = ['inv_naive_only', 'dir_naive_only',
                 'inv_gl261_only', 'dir_gl261_only',
                 'dir_naive_dir_gl261', 'inv_naive_inv_gl261',
                 'inv_naive_dir_gl261', 'dir_naive_inv_gl261']

    rho_colors = ['r', 'b', 'g', 'm', 'orange', 'y', 'dodgerblue', 'hotpink']
    rho_dict = dict(zip(cond_list, rho_colors))

    network_dict = {
        'inv_naive_only': 'r', 'dir_naive_only': 'b',
        'inv_gl261_only': 'g', 'dir_gl261_only': 'm',
        'dir_naive_dir_gl261': 'y',
        'inv_naive_inv_gl261': 'dodgerblue',
        'inv_naive_dir_gl261': 'black',
        'dir_naive_inv_gl261': 'orchid'}

    master_dict = {}
    master_table = pd.DataFrame()
    for tp in ['_7', '_14', '_30']:

        var_list = []

        if ('gl261' + tp + '_filtered vs. naive' +
                tp + '_filtered (naive_only)') in group_sig_dict:
            a = group_sig_dict[
                'gl261' + tp + '_filtered vs. naive' +
                tp + '_filtered (naive_only)']
            var_list.append(a)
        else:
            pass

        if ('gl261' + tp + '_filtered vs. naive' +
                tp + '_filtered (gl261_only)') in group_sig_dict:
            b = group_sig_dict[
                'gl261' + tp + '_filtered vs. naive' +
                tp + '_filtered (gl261_only)']
            var_list.append(b)
        else:
            pass

        if ('gl261' + tp + '_filtered vs. naive' +
                tp + '_filtered (both)') in group_sig_dict:
            c = group_sig_dict[
                'gl261' + tp + '_filtered vs. naive' +
                tp + '_filtered (both)']
            var_list.append(c)
        else:
            pass

        if var_list:
            d = pd.concat(var_list, axis=0)
            full = d[~pd.isnull(d['product'])].copy()
            full['time_point'] = tp

            master_table = master_table.append(full)
            master_table['time_point'] = master_table['time_point'].map(
                lambda x: x.lstrip('_'))

            naive = full[full['_merge'] == 'naive_only'].copy()
            naive = naive.sort_values(by=u'\u03C1' + '_naive', ascending=True)
            inv_naive_only = naive[naive[u'\u03C1' + '_naive'] < 0]
            inv_naive_only = inv_naive_only.head(10).copy()
            inv_naive_only['cond'] = 'inv_naive_only'
            master_dict['inv_naive_only' + tp] = inv_naive_only
            dir_naive_only = naive[naive[u'\u03C1' + '_naive'] > 0]
            dir_naive_only = dir_naive_only.tail(10).sort_values(
                by=u'\u03C1' + '_naive', ascending=False).copy()
            dir_naive_only['cond'] = 'dir_naive_only'
            master_dict['dir_naive_only' + tp] = dir_naive_only

            gl261 = full[full['_merge'] == 'gl261_only'].copy()
            gl261 = gl261.sort_values(by=u'\u03C1' + '_gl261', ascending=True)
            inv_gl261_only = gl261[gl261[u'\u03C1' + '_gl261'] < 0]
            inv_gl261_only = inv_gl261_only.head(10).copy()
            inv_gl261_only['cond'] = 'inv_gl261_only'
            master_dict['inv_gl261_only' + tp] = inv_gl261_only
            dir_gl261_only = gl261[gl261[u'\u03C1' + '_gl261'] > 0]
            dir_gl261_only = dir_gl261_only.tail(10).sort_values(
                by=u'\u03C1' + '_gl261', ascending=False).copy()
            dir_gl261_only['cond'] = 'dir_gl261_only'
            master_dict['dir_gl261_only' + tp] = dir_gl261_only

            dir_naive_inv_gl261 = full[
                (full['_merge'] == 'both') & (full['product'] < 0.0)
                & (full[u'\u03C1' + '_gl261'] < 0.0)].copy()
            dir_naive_inv_gl261 = dir_naive_inv_gl261.sort_values(
                by='product', ascending=True)
            dir_naive_inv_gl261 = dir_naive_inv_gl261.head(10).copy()
            dir_naive_inv_gl261['cond'] = 'dir_naive_inv_gl261'
            master_dict['dir_naive_inv_gl261' + tp] = dir_naive_inv_gl261

            inv_naive_dir_gl261 = full[
                (full['_merge'] == 'both') & (full['product'] < 0.0)
                & (full[u'\u03C1' + '_naive'] < 0.0)].copy()
            inv_naive_dir_gl261 = inv_naive_dir_gl261.sort_values(
                by='product', ascending=True)
            inv_naive_dir_gl261 = inv_naive_dir_gl261.head(10).copy()
            inv_naive_dir_gl261['cond'] = 'inv_naive_dir_gl261'
            master_dict['inv_naive_dir_gl261' + tp] = inv_naive_dir_gl261

            inv_naive_inv_gl261 = full[
                (full['_merge'] == 'both') & (full['product'] >= 0.0) &
                (full[u'\u03C1' + '_naive'] < 0.0) &
                (full[u'\u03C1' + '_gl261'] < 0.0)].copy()
            inv_naive_inv_gl261 = inv_naive_inv_gl261.sort_values(
                by='product', ascending=False)
            inv_naive_inv_gl261 = inv_naive_inv_gl261.head(10).copy()
            inv_naive_inv_gl261['cond'] = 'inv_naive_inv_gl261'
            master_dict['inv_naive_inv_gl261' + tp] = inv_naive_inv_gl261

            dir_naive_dir_gl261 = full[
                (full['_merge'] == 'both') & (full['product'] >= 0.0) &
                (full[u'\u03C1' + '_naive'] > 0.0) &
                (full[u'\u03C1' + '_gl261'] > 0.0)].copy()
            dir_naive_dir_gl261 = dir_naive_dir_gl261.sort_values(
                by='product', ascending=False)
            dir_naive_dir_gl261 = dir_naive_dir_gl261.head(10).copy()
            dir_naive_dir_gl261['cond'] = 'dir_naive_dir_gl261'
            master_dict['dir_naive_dir_gl261' + tp] = dir_naive_dir_gl261

            # plot clustermaps and barcharts
            sns.set_style('whitegrid')
            # combo_counts = []
            for cond in ['naive_only', 'gl261_only']:
                s1 = full[['x', 'y', u'\u03C1' + '_' + cond.split('_')[0],
                           'product', '_merge']][(
                                full['_merge'] == cond) |
                                (full['_merge'] == 'both')]

                if not s1.empty:
                    s2 = s1.set_index(['x', 'y'])
                    s3 = s2.swaplevel(0, 1)
                    s4 = s2.append(s3)
                    s5 = s4.drop(['product', '_merge'], axis=1)
                    s6 = s5.unstack().astype(float)
                    s6.replace(to_replace=np.nan, value=0.0, inplace=True)
                    s6.columns = s6.columns.droplevel(0)

                    g = sns.clustermap(
                        s6, method='average', metric='euclidean',
                        cmap='cividis', row_cluster=True, col_cluster=True,
                        center=0.0, xticklabels=1, yticklabels=1)

                    for item in g.ax_heatmap.get_xticklabels():
                        item.set_rotation(90)
                        item.set_size(5)
                    for item in g.ax_heatmap.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(5)

                    y_labels = pd.DataFrame({'y_labels': g.data2d.index})

                    plt.savefig(
                        os.path.join(
                            save_dir, cond.split('_')[0] +
                            '_filtered_sig_heatmap' +
                            tp + '.pdf'))
                    plt.close('all')

                    s7 = s1[s1['product'] <= 0]
                    counts = pd.DataFrame(
                        s7['x'].append(s7['y']).value_counts())
                    counts.rename(columns={0: 'count'}, inplace=True)
                    counts.reset_index(inplace=True)
                    counts.rename(columns={'index': 'y_labels'}, inplace=True)

                    m = y_labels.merge(
                        counts, how='outer', on='y_labels', indicator=True)
                    m['count'].replace(
                        to_replace=np.nan, value=0.0, inplace=True)
                    m.sort_index(ascending=False, inplace=True)
                    m.set_index(m['y_labels'], inplace=True)
                    m.drop(['_merge', 'y_labels'], axis=1, inplace=True)

                    o = m.plot(
                        kind='barh', stacked=True, figsize=(6, 9),
                        color='k')
                    o.yaxis.grid(True)
                    o.yaxis.grid(False)
                    for item in o.get_xticklabels():
                        item.set_rotation(0)
                        item.set_size(10)
                    for item in o.get_yticklabels():
                        item.set_rotation(0)
                        item.set_size(4)
                    # plt.xlim(0, 25)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(save_dir, 'bar_' + cond + tp + '.pdf'))
                    plt.close('all')

    master_table.to_csv(
        save_dir + '/sig_corrs_table.csv', index=False, encoding='utf-16',
        sep='\t')

    # plot stacked barchart of most significant correlation differences
    master_table['sort_order'] = abs(master_table['product'])

    master_table.sort_values(by='sort_order', inplace=True)

    plot_data1 = master_table[(master_table.sort_order <= 0.3)]

    plot_data2 = plot_data1.loc[
        :, [u'\u03C1' + '_gl261', u'\u03C1' + '_naive']]

    plot_data2.plot(
        x=plot_data1.x.map(str) + ' vs. ' + plot_data1.y + '_' +
        plot_data1.time_point.map(str),
        kind='bar', stacked=False)
    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, 'rho_barcahrts.pdf'))
    plt.close('all')

    # filter master_dict to isolate and sort most significant correlations
    filt_dict = {}
    master_dict_sorted = copy.deepcopy(master_dict)
    for i in master_dict_sorted.keys():
        # for i in cond_list:
        e = '_'.join(i.split('_')[:-1])
        df = pd.DataFrame()
        # for tp in ['_7', '_14', '_30']:
        if e in ['inv_naive_only', 'dir_naive_only',
                 'inv_gl261_only', 'dir_gl261_only']:
            master_dict_sorted[e+tp]['sort'] = abs(
                master_dict_sorted[e+tp]['product'])
            data = master_dict_sorted[e+tp][
                (master_dict_sorted[e+tp].sort <= 0.3)].sort_values(
                    by='sort').drop('sort', axis=1)
            df = df.append(data)
        elif i in ['dir_naive_dir_gl261', 'inv_naive_inv_gl261']:
            data = master_dict_sorted[e+tp].sort_values(
                by='product', ascending=False)
            df = df.append(data)
        elif i in ['inv_naive_dir_gl261', 'dir_naive_inv_gl261']:
            data = master_dict_sorted[e+tp].sort_values(
                by='product', ascending=False)
            df = df.append(data)
        filt_dict[e] = df

    # get scatter plot for the most significant correlation(s) in each category
    # slice each DataFrame to grab top n regression plots
    plot_dir = os.path.join(save_dir, 'regression_plots')
    os.makedirs(plot_dir)
    low_slice = 0
    high_slice = 100
    for k, v, in filt_dict.items():
        for tp in ['_7', '_14', '_30']:
            if not v[v['time_point'] == tp].empty:
                for i in v[v['time_point'] == tp].iloc[
                  low_slice:high_slice, :].iterrows():

                    data = i

                    dir = '/gl261' + tp + \
                          '_filtered vs. naive' + tp + \
                          '_filtered (' + data[1]['_merge'] + ')/'

                    x = data[1]['x'].replace(u'\u207A', 'pos').replace(
                        u'\u207B', 'neg').replace('CD8' + u'\u03B1', 'CD8a')
                    y = data[1]['y'].replace(u'\u207A', 'pos').replace(
                        u'\u207B', 'neg').replace('CD8' + u'\u03B1', 'CD8a')

                    file = x + ' vs. ' + y + '.pdf'

                    copy2(
                        scatter_dir + dir + file, plot_dir + '/' +
                        file + tp + '.pdf')

    # convert fitered_dict to a single Dataframe
    dd = pd.concat(filt_dict).reset_index(drop=True)

    # identify repeating correlation pairs
    duplicates = dd[
        dd.duplicated(subset=['x', 'y'], keep=False)]
    duplicates.sort_index(inplace=True)

    # show that no repeating correlation pairs were missed because x and y
    # were reversed between time points
    g = pd.DataFrame()
    for i in dd.iterrows():
        x = i[1]['x']
        y = i[1]['y']
        negate_index = dd.index.isin([i[0]])
        g = g.append(
            dd[~negate_index][
                (dd['x'] == x) & (dd['y'] == y) | (dd['x'] == y) &
                (dd['y'] == x)])
    g.sort_index(inplace=True)

    if duplicates.equals(g) is True:
        print('duplicate check is good!')

    duplicates = duplicates.sort_values(by=['x', 'y'], inplace=False)

    # plot naive vs. gl261 rho coefficients
    for tp in ['_7', '_14', '_30']:
        sns.set(style='whitegrid')
        fig, ax = plt.subplots(figsize=(10, 10))

        plt.scatter(
            full[u'\u03C1' + '_naive'], full[u'\u03C1' + '_gl261'], marker='o',
            c='k', s=260, zorder=1)
        for key, value in rho_dict.items():
            w = dd[dd['cond'] == key]
            plt.scatter(
                w[u'\u03C1' + '_naive'], w[u'\u03C1' + '_gl261'], marker='o',
                c=value, s=260, zorder=2)

        for idx, n, g in zip(
          zip(full['x'], full['y']),
          full[u'\u03C1' + '_naive'], full[u'\u03C1' + '_gl261']):

            plt.annotate(idx,
                         xy=(n, g), xytext=(0, 2),
                         size=1, weight='bold',
                         textcoords='offset points',
                         ha='left', va='top', zorder=3)

        plt.xlabel('naive' + '(' + u'\u03C1' + ')', size=25,
                   weight='normal', labelpad=13, color='k')
        plt.ylabel('gl261' + '(' + u'\u03C1' + ')', size=25,
                   weight='normal', labelpad=5, color='k')

        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, 'naive_v_gl261_rho' + tp + '.pdf'))
        plt.close('all')

        # count the number of times a TS-BIP is involved
        # in a condition-specific correlation
        naive_counts = dd['x'][(dd['time_point'] == tp) &
                               ((dd['cond'] == 'dir_naive_only') |
                               (dd['cond'] == 'inv_naive_only') |
                               (dd['cond'] == 'dir_naive_inv_gl261') |
                               (dd['cond'] == 'inv_naive_dir_gl261'))].append(
                               dd['y'][(dd['time_point'] == tp) &
                                       ((dd['cond'] == 'dir_naive_only') |
                                       (dd['cond'] == 'inv_naive_only') |
                                       (dd['cond'] == 'dir_naive_inv_gl261') |
                                       (dd['cond'] == 'inv_naive_dir_gl261'))]
                                       ).value_counts()

        gl261_counts = dd['x'][(dd['time_point'] == tp) &
                               ((dd['cond'] == 'dir_gl261_only') |
                               (dd['cond'] == 'inv_gl261_only') |
                               (dd['cond'] == 'dir_naive_inv_gl261') |
                               (dd['cond'] == 'inv_naive_dir_gl261'))].append(
                               dd['y'][(dd['time_point'] == tp) &
                                       ((dd['cond'] == 'dir_gl261_only') |
                                       (dd['cond'] == 'inv_gl261_only') |
                                       (dd['cond'] == 'dir_naive_inv_gl261') |
                                       (dd['cond'] == 'inv_naive_dir_gl261'))]
                                       ).value_counts()

        shared_counts = dd['x'][(dd['time_point'] == tp) &
                                ((dd['cond'] == 'dir_naive_dir_gl261') |
                                (dd['cond'] == 'inv_naive_inv_gl261'))].append(
                                dd['y'][(dd['time_point'] == tp) &
                                        ((dd['cond'] == 'dir_naive_dir_gl261')
                                        |
                                        (dd['cond'] == 'inv_naive_inv_gl261'))]
                                        ).value_counts()

        # plot pie chart
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.78, wspace=0.9)

        counts_dict = {'Ctrl': (ax1, naive_counts), 'GBM': (ax2, gl261_counts),
                       'Shared': (ax3, shared_counts)}

        keyorder = ['Ctrl', 'GBM', 'Shared']

        counts_dict_sorted = OrderedDict(
            sorted(counts_dict.items(), key=lambda x: keyorder.index(x[0])))

        colors = plt.cm.tab20c((4./3*np.arange(20*3/4)).astype(int))
        for key, value in counts_dict_sorted.items():
            explode_num = len([i for i in value[1] if i != 1])
            wedges, texts, autotexts = value[0].pie(
                value[1], shadow=False, colors=colors, autopct='%1.1f%%',
                startangle=0, radius=0.1, labels=value[1].index,
                rotatelabels=True, explode=[0.006]*explode_num + [0]*(
                    len(value[1])-explode_num))
            for t1, t2, t3, w in zip(texts, autotexts, value[1], wedges):
                t1.set_fontsize(7)
                t2.set_fontsize(9)
                t2.set_text(t3)
                t2.set_fontweight('bold')
                t2.set_color('white')
                w.set_linewidth(2)
                w.set_edgecolor('k')
            value[0].axis('equal')
            value[0].set_title(key, y=1.1)

        fig.suptitle(
            'Condition-specific correlations at ' + tp.strip('_') +
            ' ' + 'dpi', y=0.98)
        plt.savefig(
            os.path.join(save_dir, 'pie_' + tp + '.pdf'))
        plt.close('all')

        # plot time point-specific correlation network diagrams
        s1 = dd[dd['time_point'] == tp]
        s2 = s1.copy()
        s2['color'] = s2['cond'].map(network_dict)

        nodes = set(s2['y'].append(s2['x']).tolist())

        G = nx.Graph()

        for node in nodes:

            G.add_node(node)

        for row in s2.iterrows():
            if row[1]['_merge'] == 'naive_only':

                if row[1][u'\u03C1' + '_naive'] < 0.0:

                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_naive'],
                               style='solid',
                               color='#4479bb',
                               label='naive_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_naive']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_naive'],
                               style='solid',
                               color='#d53d69',
                               label='naive_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_naive']))

            elif row[1]['_merge'] == 'gl261_only':

                if row[1][u'\u03C1' + '_gl261'] < 0.0:

                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='solid',
                               color='#4479bb',
                               label='gl261_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_gl261']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='solid',
                               color='#d53d69',
                               label='gl261_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_gl261']))

            elif row[1]['_merge'] == 'both':

                if (
                  row[1][u'\u03C1' + '_naive'] *
                  row[1][u'\u03C1' + '_gl261']) < 0.0:

                    if row[1][u'\u03C1' + '_naive'] < 0.0:

                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5,
                                   style='solid',
                                   color='darkorange',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5)
                    else:
                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5,
                                   style='dotted',
                                   color='darkorange',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5)

                else:

                    if row[1][u'\u03C1' + '_gl261'] < 0.0:

                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=(row[1][u'\u03C1' + '_naive'] +
                                           row[1][u'\u03C1' + '_gl261'])/2,
                                   style='solid',
                                   color='#4479bb',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs((
                                        row[1][u'\u03C1' + '_naive'] +
                                        row[1][u'\u03C1' + '_gl261'])/2))

                    else:
                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=(row[1][u'\u03C1' + '_naive'] +
                                           row[1][u'\u03C1' + '_gl261'])/2,
                                   style='solid',
                                   color='#d53d69',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs((
                                        row[1][u'\u03C1' + '_naive'] +
                                        row[1][u'\u03C1' + '_gl261'])/2))

        # generate a filtered subgraph
        for label in ['broken', 'induced', 'inverted', 'shared']:

            K = nx.Graph()

            if label == 'broken':
                for (u, v, b) in G.edges(data=True):
                    if (
                      (b['cond'] == 'inv_naive_only') |
                      (b['cond'] == 'dir_naive_only')):

                        K.add_edge(u, v, **b)

            elif label == 'induced':
                for (u, v, b) in G.edges(data=True):
                    if (
                      (b['cond'] == 'inv_gl261_only') |
                      (b['cond'] == 'dir_gl261_only')):

                        K.add_edge(u, v, **b)

            elif label == 'inverted':
                for (u, v, b) in G.edges(data=True):
                    if (
                      (b['cond'] == 'inv_naive_dir_gl261') |
                      (b['cond'] == 'dir_naive_inv_gl261')):

                        K.add_edge(u, v, **b)

            elif label == 'shared':
                for (u, v, b) in G.edges(data=True):
                    if (
                      (b['cond'] == 'inv_naive_inv_gl261') |
                      (b['cond'] == 'dir_naive_dir_gl261')):

                        K.add_edge(u, v, **b)

            plt.figure(figsize=(8, 8))
            plt.axis('off')

            edge_weights = [
                (G[u][v]['abs_weight']**7) * 10 for u, v in K.edges()]

            edge_colors = [G[u][v]['color'] for u, v in K.edges()]

            edge_style = [G[u][v]['style'] for u, v in K.edges()]

            degree = nx.degree(K, weight='abs_weight')

            nx.draw_networkx(
                K, pos=nx.circular_layout(K),
                node_size=[(v**2.1) * 200 for (k, v) in degree],
                node_color='#f2f2f2',
                with_labels=True, font_size=6, font_weight='bold', alpha=1,
                font_color=[0.3, 0.3, 0.3],
                edge_color=edge_colors,
                style=edge_style,
                width=edge_weights)

            plt.savefig(
                os.path.join(save_dir, label + '_network_' + tp + '.pdf'))
            plt.close('all')

        # get time point-specific cross-condition networks
        G = nx.Graph()

        for node in nodes:

            G.add_node(node)

        for row in s2.iterrows():
            if row[1]['_merge'] == 'naive_only':

                if row[1][u'\u03C1' + '_naive'] < 0.0:

                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_naive'],
                               style='dotted',
                               color='darkorange',  # 4479bb
                               label='naive_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_naive']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_naive'],
                               style='solid',
                               color='darkorange',  # d53d69
                               label='naive_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_naive']))

            elif row[1]['_merge'] == 'gl261_only':

                if row[1][u'\u03C1' + '_gl261'] < 0.0:

                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='dotted',
                               color='dodgerblue',
                               label='gl261_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_gl261']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='solid',
                               color='dodgerblue',
                               label='gl261_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_gl261']))

            elif row[1]['_merge'] == 'both':

                if (
                  row[1][u'\u03C1' + '_naive'] *
                  row[1][u'\u03C1' + '_gl261']) < 0.0:

                    if row[1][u'\u03C1' + '_naive'] < 0.0:

                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5,
                                   style='solid',
                                   color='deeppink',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5)
                    else:
                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5,
                                   style='dotted',
                                   color='deeppink',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs(
                                    row[1][u'\u03C1' + '_naive'] -
                                    row[1][u'\u03C1' + '_gl261'])/1.5)

                else:

                    if row[1][u'\u03C1' + '_gl261'] < 0.0:

                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=(row[1][u'\u03C1' + '_naive'] +
                                           row[1][u'\u03C1' + '_gl261'])/2,
                                   style='dotted',
                                   color='gray',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs((
                                        row[1][u'\u03C1' + '_naive'] +
                                        row[1][u'\u03C1' + '_gl261'])/2))

                    else:
                        G.add_edge(u=row[1]['x'], v=row[1]['y'],
                                   weight=(row[1][u'\u03C1' + '_naive'] +
                                           row[1][u'\u03C1' + '_gl261'])/2,
                                   style='solid',
                                   color='gray',
                                   label='both',
                                   cond=row[1]['cond'],
                                   abs_weight=abs((
                                        row[1][u'\u03C1' + '_naive'] +
                                        row[1][u'\u03C1' + '_gl261'])/2))

        # generate a filtered subgraph
        K = nx.Graph()

        for (u, v, b) in G.edges(data=True):

                K.add_edge(u, v, **b)

        plt.figure(figsize=(8, 8))
        plt.axis('off')

        edge_weights = [
            (G[u][v]['abs_weight']**7) * 10 for u, v in K.edges()]

        edge_colors = [G[u][v]['color'] for u, v in K.edges()]

        edge_style = [G[u][v]['style'] for u, v in K.edges()]

        degree = nx.degree(K, weight='abs_weight')

        nx.draw_networkx(
            K, pos=nx.circular_layout(K),
            node_size=[(v**2.1) * 200 for (k, v) in degree],
            node_color='#f2f2f2',
            with_labels=True, font_size=6, font_weight='bold', alpha=1,
            font_color=[0.3, 0.3, 0.3],
            edge_color=edge_colors,
            style=edge_style,
            width=edge_weights)

        plt.savefig(
            os.path.join(save_dir, 'cross-condition_' + tp + '.pdf'))
        plt.close('all')


sig_correlation_counts()


# def missing_populations():
#     banner('RUNNING MODULE: missing_populations')
#
#     os.chdir(correlation_pickle_dir)
#     pi_unfiltered_subset_corrs = open('unfiltered_subset_corrs.pickle', 'rb')
#     unfiltered_subset_corrs = pickle.load(pi_unfiltered_subset_corrs)
#
#     pi_unfiltered_pivot_subsets = open(
#         'unfiltered_pivot_subsets.pickle', 'rb')
#     unfiltered_pivot_subsets = pickle.load(pi_unfiltered_pivot_subsets)
#
#     naive_percentages = unfiltered_pivot_subsets[
#         'experimental_data']['naive_unfiltered']
#
#     gl261_percentages = unfiltered_pivot_subsets[
#         'experimental_data']['gl261_filtered']
#
#     naive_data = unfiltered_subset_corrs[
#         'experimental_data']['naive_unfiltered']
#     gl261_data = unfiltered_subset_corrs[
#         'experimental_data']['gl261_filtered']
#
#     gl261_data.columns = naive_data.columns
#     naive_percentages.columns = naive_data.columns
#     gl261_percentages.columns = naive_data.columns
#
#     naive = []
#     gl261 = []
#     gl261_mean = []
#     naive_mean = []
#     for ncol, gcol, npercent, gpercent in zip(
#       naive_data, gl261_data, naive_percentages, gl261_percentages):
#         if naive_data[ncol].isnull().all():
#             naive.append(ncol)
#             if gl261_data[gcol].isnull().all():
#                 gl261.append(gcol)
#             else:
#                 gl261_mean.append((gcol, gl261_percentages[gpercent].mean()))
#
#         elif gl261_data[gcol].isnull().all():
#             gl261.append(gcol)
#             if naive_data[ncol].isnull().all():
#                 naive.append(ncol)
#             else:
#                 naive_mean.append((ncol, naive_percentages[npercent].mean()))
#
#     intersection = set(naive).intersection(gl261)
#     naive_missing = set(naive) - set(gl261)
#     gl261_missing = set(gl261) - set(naive)
#
#     return (intersection, naive_missing, gl261_missing,
#             naive_mean, gl261_mean)
#
#
# (intersection, naive_missing, gl261_missing,
#     naive_mean, gl261_mean) = missing_populations()
