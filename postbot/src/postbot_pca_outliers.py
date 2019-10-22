# CONFIGURATIONS

# invoke required libraries
import pandas as pd
import sys
import os
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

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

# script call error message
if len(sys.argv) != 2:
    print('Usage: postbot.py <path_to_project>')
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

# path to postbot pickles
pickle_dir = os.path.join(project_path, 'postbot', 'data',
                          'logicle_e20', 'pickled_global_vars')


# make banner to introduce each RUNNING MODULE
def banner(MODULE_title):

    print('=' * 70)
    print(MODULE_title)
    print('=' * 70)


def PCA_outliers():

    banner('RUNNING MODULE: PCA_outliers')

    os.chdir(pickle_dir)
    pi_overall = open('overall.pickle', 'rb')
    overall = pickle.load(pi_overall)

    outlier_plot_dir = os.path.join(
        project_path, 'postbot', 'data', 'logicle_e20', 'outlier_cms')

    os.makedirs(outlier_plot_dir)

    frame = pd.DataFrame(columns=['status', 'time_point', 'tissue',
                                  'replicate', 'cell_type', 'percent'])
    status = []
    time_point = []
    tissue = []
    replicate = []
    cell_type = []
    percent = []

    for name1, group1 in overall.groupby(
      ['status', 'time_point', 'tissue', 'replicate']):
        if all(group1['status'] == 'gl261'):
            if all(group1['time_point'] == 30):
                for name2, group2 in group1.groupby(['cell_type']):
                    if all(group2['cell_type'] != 'unspecified'):

                        cell_num = group2
                        total_cells = group1

                        percent_comp = (len(cell_num)/len(total_cells))*100
                        percent_comp = float('%.6f' % percent_comp)

                        status.append(name1[0])
                        time_point.append(name1[1])
                        tissue.append(name1[2])
                        replicate.append(name1[3])
                        cell_type.append(name2)
                        percent.append(percent_comp)

    frame['status'] = status
    frame['time_point'] = time_point
    frame['tissue'] = tissue
    frame['replicate'] = replicate
    frame['cell_type'] = cell_type
    frame['percent'] = percent

    merge_frame = pd.DataFrame(columns=['status', 'time_point', 'tissue',
                                        'replicate', 'cell_type'])

    IP_num = len(overall['cell_type'].unique().tolist())

    tissues = [
        ['blood']*IP_num, ['marrow']*IP_num, ['nodes']*IP_num,
        ['spleen']*IP_num, ['thymus']*IP_num]*8
    tissues = list(itertools.chain.from_iterable(tissues))
    merge_frame['tissue'] = tissues

    replicate = [[1]*IP_num, [2]*IP_num, [3]*IP_num, [4]*IP_num, [5]*IP_num,
                 [6]*IP_num, [7]*IP_num, [8]*IP_num]*5
    replicate = list(itertools.chain.from_iterable(replicate))
    merge_frame['replicate'] = replicate

    celltypes = [sorted(overall['cell_type'].unique().tolist())]*5*8
    celltypes = list(itertools.chain.from_iterable(celltypes))
    merge_frame['cell_type'] = celltypes

    merge_frame['status'] = 'gl261'
    merge_frame['time_point'] = 30

    master = frame.merge(
        merge_frame, how='right',
        on=merge_frame.columns.tolist(), indicator=True)

    master.sort_values(by=['tissue', 'replicate'], inplace=True)

    master['percent'].replace(to_replace=np.nan, value=0.0, inplace=True)

    master.drop(['_merge'], axis=1, inplace=True)

    # z_list = []
    # for name, group in master.groupby(['tissue', 'cell_type']):
    #     z = group['percent'].tolist()
    #     z_list.append(stats.zscore(z).tolist())
    # z_scores = list(itertools.chain.from_iterable(z_list))
    # master['z_score'] = z_scores
    # master['z_score'].replace(to_replace=np.nan, value=0.0, inplace=True)

    medians = master.groupby(['tissue', 'cell_type']).median().unstack().T
    medians = medians[medians.index.get_level_values(0) == 'percent']

    param_list = []
    a_list = []
    for row in master.iterrows():

        a = row[1][5]
        a_list.append(a)
        b = medians[
            medians.index.get_level_values(1) == row[1][4]][row[1][2]][0]

        param = (0.25 + a)/(0.25 + b)

        param_list.append(param)

    master['param'] = param_list
    master['binned_param'] = pd.cut(master['param'], 4, labels=[1, 2, 3, 4],
                                    retbins=True)[0].tolist()

    for name, group3 in master.groupby(['tissue']):

        clustermap_input = group3.pivot_table(
            index='cell_type', columns='replicate', values='param')

        clustermap_input = clustermap_input.apply(np.log2)

        # for row in clustermap_input.iterrows():
        #     for e, i in enumerate(row[1]):
        #         if -0.5 < i < 0.5:
        #             clustermap_input.set_value(
        #                 row[0], row[1].index[e], 0.0, takeable=False)

        clustermap_input.drop('unspecified', axis=0, inplace=True)

        cmap = sns.diverging_palette(186.7, 75.0, s=99, l=90.4, center='dark',
                                     as_cmap=True)

        g = sns.clustermap(clustermap_input, square=True,
                           linewidth=1.0, cmap=cmap, vmin=-2.5, vmax=2.5)

        from matplotlib.patches import Rectangle
        ax = g.ax_heatmap

        ax.add_patch(Rectangle((0, 0), clustermap_input.shape[1],
                     clustermap_input.shape[0], fill=False,
                     edgecolor='k', lw=3))

        for item in g.ax_heatmap.get_xticklabels():
            item.set_rotation(0)
            item.set_size(15)
            item.set_weight('bold')

        ylabels = [
            item.get_text() for item in
            g.ax_heatmap.get_yticklabels()]

        ylabels_update = [ylabel.replace(
            'neg', '$^-$').replace('pos', '$^+$').replace(
            'CD8a', 'CD8' + u'\u03B1') for ylabel in ylabels]

        g.ax_heatmap.set_yticklabels(ylabels_update)

        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
            item.set_size(10)
            item.set_weight('normal')

        xlabel = g.ax_heatmap.get_xlabel()
        g.ax_heatmap.set_xlabel(xlabel, size=20, weight='bold', labelpad=20)

        g.ax_heatmap.set_ylabel('immunophenotype', size=20,
                                weight='bold', labelpad=10)

        plt.savefig(os.path.join(outlier_plot_dir, name + '.pdf'))


PCA_outliers()
