import pandas as pd
import sys
import os
import numpy as np
from shutil import copy2
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
from datetime import datetime
import pickle
import pathlib
import networkx as nx

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

save_dir = os.path.join(project_path, 'correlations', 'data',
                        'spearman_correlation_analysis', 'orthant')

correlation_pickle_dir = os.path.join(save_dir, 'pickled_global_vars')

scatter_dir = os.path.join(project_path, 'correlations', 'data',
                           'spearman_correlation_analysis',
                           'orthant', 'scatter_plots')

save_dir2 = os.path.join(project_path, 'correlations', 'data',
                         'spearman_correlation_analysis', 'orthant',
                         'correlation_counts')

pathlib.Path(save_dir2).mkdir(parents=False, exist_ok=True)

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
    combo_counts = []
    for cond in ['naive_only', 'gl261_only']:
        s1 = full[['x', 'y', u'\u03C1' + '_' + cond.split('_')[0],
                   'product', '_merge']][(
                        full['_merge'] == cond) |
                        (full['_merge'] == 'both')]
        s2 = s1.set_index(['x', 'y'])
        s3 = s2.swaplevel(0, 1)
        s4 = s2.append(s3)
        s5 = s4.drop(['product', '_merge'], axis=1)
        s6 = s5.unstack().astype(float)
        s6.replace(to_replace=np.nan, value=0.0, inplace=True)
        s6.columns = s6.columns.droplevel(0)

        cmap = sns.color_palette('RdBu_r', n_colors=1000000)

        g = sns.clustermap(
            s6, method='average', metric='euclidean',
            cmap=cmap, row_cluster=True, col_cluster=True, center=0.0,
            xticklabels=1, yticklabels=1)

        for item in g.ax_heatmap.get_xticklabels():
            item.set_rotation(90)
            item.set_size(5)
        for item in g.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
            item.set_size(5)

        y_labels = pd.DataFrame({'y_labels': g.data2d.index})

        plt.savefig(
            os.path.join(
                save_dir2, cond.split('_')[0] + '_filtered_sig_heatmap' +
                tp + '.pdf'))
        plt.close('all')

        s7 = s1[s1['product'] <= 0]
        counts = pd.DataFrame(s7['x'].append(s7['y']).value_counts())
        counts.rename(columns={0: 'count'}, inplace=True)
        counts.reset_index(inplace=True)
        counts.rename(columns={'index': 'y_labels'}, inplace=True)

        m = y_labels.merge(
            counts, how='outer', on='y_labels', indicator=True)
        m['count'].replace(to_replace=np.nan, value=0.0, inplace=True)
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
        plt.xlim(0, 25)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir2, 'bar_' + cond + tp + '.pdf'))
        plt.close('all')

master_table.to_csv(
    save_dir2 + '/sig_corrs_table.csv', index=False, encoding='utf-16',
    sep='\t')

# filter master_dict to isolate and sort most significant correlations
filt_dict = {}
master_dict_sorted = copy.deepcopy(master_dict)
for i in cond_list:
    df = pd.DataFrame()
    for tp in ['_7', '_14', '_30']:
        if i in ['inv_naive_only', 'dir_naive_only',
                 'inv_gl261_only', 'dir_gl261_only']:
            master_dict_sorted[i+tp]['sort'] = abs(
                master_dict_sorted[i+tp]['product'])
            data = master_dict_sorted[i+tp][
                (master_dict_sorted[i+tp].sort <= 0.3)].sort_values(
                    by='sort').drop('sort', axis=1)
            df = df.append(data)
        elif i in ['dir_naive_dir_gl261', 'inv_naive_inv_gl261']:
            data = master_dict_sorted[i+tp].sort_values(
                by='product', ascending=False)
            df = df.append(data)
        elif i in ['inv_naive_dir_gl261', 'dir_naive_inv_gl261']:
            data = master_dict_sorted[i+tp].sort_values(
                by='product', ascending=False)
            df = df.append(data)
    filt_dict[i] = df

# get scatter plot for the most significant correlation(s) in each category
# slice each DataFrame to grab top n regression plots
plot_dir = os.path.join(save_dir2, 'regression_plots')
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
            (dd['x'] == x) & (dd['y'] == y) | (dd['x'] == y) & (dd['y'] == x)])
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
        os.path.join(save_dir2, 'naive_v_gl261_rho' + tp + '.pdf'))
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
                                    ((dd['cond'] == 'dir_naive_dir_gl261') |
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
        os.path.join(save_dir2, 'pie_' + tp + '.pdf'))
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
            os.path.join(save_dir2, label + '_network_' + tp + '.pdf'))
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
        os.path.join(save_dir2, 'cross-condition_' + tp + '.pdf'))
    plt.close('all')

print(
    'Correlation counts timepoints completed in ' +
    str(datetime.now() - startTime))
