import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
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

save_dir2 = os.path.join(project_path, 'correlations', 'data',
                         'spearman_correlation_analysis', 'orthant',
                         'correlation_counts')

pathlib.Path(save_dir2).mkdir(parents=False, exist_ok=True)

os.chdir(correlation_pickle_dir)
pi_group_sig_dict = open('group_sig_dict.pickle', 'rb')
group_sig_dict = pickle.load(pi_group_sig_dict)

master = {}
for tp in ['_7', '_14', '_30', '']:

    a = group_sig_dict[
        'gl261' + tp + '_filtered vs. naive' +
        tp + '_filtered (naive_only)']
    b = group_sig_dict[
        'gl261' + tp + '_filtered vs. naive' +
        tp + '_filtered (gl261_only)']
    c = group_sig_dict[
        'gl261' + tp + '_filtered vs. naive' +
        tp + '_filtered (both)']

    d = pd.concat([a, b, c], axis=0)
    full = d[~pd.isnull(d['product'])].copy()
    full['time_point'] = tp

    naive = full[full['_merge'] == 'naive_only'].copy()
    naive = naive.sort_values(by=u'\u03C1' + '_naive', ascending=True)
    inv_naive_only = naive.head(10).copy()
    inv_naive_only['cond'] = 'inv_naive_only'
    master['inv_naive_only' + tp] = inv_naive_only
    dir_naive_only = naive.tail(10).sort_values(
        by=u'\u03C1' + '_naive', ascending=False).copy()
    dir_naive_only['cond'] = 'dir_naive_only'
    master['dir_naive_only' + tp] = dir_naive_only

    gl261 = full[full['_merge'] == 'gl261_only'].copy()
    gl261 = gl261.sort_values(by=u'\u03C1' + '_gl261', ascending=True)
    inv_gl261_only = gl261.head(10).copy()
    inv_gl261_only['cond'] = 'inv_gl261_only'
    master['inv_gl261_only' + tp] = inv_gl261_only
    dir_gl261_only = gl261.tail(10).sort_values(
        by=u'\u03C1' + '_gl261', ascending=False).copy()
    dir_gl261_only['cond'] = 'dir_gl261_only'
    master['dir_gl261_only' + tp] = dir_gl261_only

    dir_naive_inv_gl261 = full[
        (full['_merge'] == 'both') & (full['product'] < 0.0)
        & (full[u'\u03C1' + '_gl261'] < 0.0)].copy()
    dir_naive_inv_gl261 = dir_naive_inv_gl261.sort_values(
        by='product', ascending=True)
    dir_naive_inv_gl261 = dir_naive_inv_gl261.head(10).copy()
    dir_naive_inv_gl261['cond'] = 'dir_naive_inv_gl261'
    master['dir_naive_inv_gl261' + tp] = dir_naive_inv_gl261

    inv_naive_dir_gl261 = full[
        (full['_merge'] == 'both') & (full['product'] < 0.0)
        & (full[u'\u03C1' + '_naive'] < 0.0)].copy()
    inv_naive_dir_gl261 = inv_naive_dir_gl261.sort_values(
        by='product', ascending=True)
    inv_naive_dir_gl261 = inv_naive_dir_gl261.head(10).copy()
    inv_naive_dir_gl261['cond'] = 'inv_naive_dir_gl261'
    master['inv_naive_dir_gl261' + tp] = inv_naive_dir_gl261

    inv_naive_inv_gl261 = full[
        (full['_merge'] == 'both') & (full['product'] >= 0.0) &
        (full[u'\u03C1' + '_naive'] < 0.0) &
        (full[u'\u03C1' + '_gl261'] < 0.0)].copy()
    inv_naive_inv_gl261 = inv_naive_inv_gl261.sort_values(
        by='product', ascending=False)
    inv_naive_inv_gl261 = inv_naive_inv_gl261.head(10).copy()
    inv_naive_inv_gl261['cond'] = 'inv_naive_inv_gl261'
    master['inv_naive_inv_gl261' + tp] = inv_naive_inv_gl261

    dir_naive_dir_gl261 = full[
        (full['_merge'] == 'both') & (full['product'] >= 0.0) &
        (full[u'\u03C1' + '_naive'] > 0.0) &
        (full[u'\u03C1' + '_gl261'] > 0.0)].copy()
    dir_naive_dir_gl261 = dir_naive_dir_gl261.sort_values(
        by='product', ascending=False)
    dir_naive_dir_gl261 = dir_naive_dir_gl261.head(10).copy()
    dir_naive_dir_gl261['cond'] = 'dir_naive_dir_gl261'
    master['dir_naive_dir_gl261' + tp] = dir_naive_dir_gl261

    # plot naive vs. gl261 rho coefficients
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, x, y, m in zip(
      full.index, full[u'\u03C1' + '_naive'],
      full[u'\u03C1' + '_gl261'],
      full['_merge']):
        if i in inv_naive_only.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='r', s=260, zorder=2)
        elif i in dir_naive_only.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='b', s=260, zorder=2)
        elif i in inv_gl261_only.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='g', s=260, zorder=2)
        elif i in dir_gl261_only.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='m', s=260, zorder=2)
        elif i in dir_naive_inv_gl261.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='hotpink', s=260, zorder=2)
        elif i in inv_naive_dir_gl261.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='dodgerblue', s=260, zorder=2)
        elif i in inv_naive_inv_gl261.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='y', s=260, zorder=2)
        elif i in dir_naive_dir_gl261.index:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='orange', s=260, zorder=2)
        else:
            # z = np.random.random_integers(1, 200, 1)
            plt.scatter(x, y, marker='o', c='k', s=260, zorder=1)

    for idx, n, g in zip(
      zip(full['x'], full['y']),
      full[u'\u03C1' + '_naive'], full[u'\u03C1' + '_gl261']):
        # if abs(g-n) > 1.1:

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

    # plot 'population' isobologram
    ips = pd.DataFrame(full['x'].append(full['y']))
    ips.rename(columns={0: 'celltype'}, inplace=True)

    population = pd.DataFrame(ips.groupby(['celltype']).size())
    population.rename(columns={0: 'count'}, inplace=True)

    mydict = {}
    for i in population.index:
        mylist = []
        for j in full.iterrows():
            if j[1]['x'] == i or j[1]['y'] == i:
                mylist.append(j[1]['product'])
        mydict[i] = mylist

    population['mean_product'] = ''
    for k, v in mydict.items():
        mean = sum(v)/len(v)
        population.at[k, 'mean_product'] = mean

    population['tissue'] = population.index.str.split('_').str[1]

    fig, ax = plt.subplots(figsize=(10, 10))
    color_dict = {'blood': 'r', 'marrow': 'b', 'nodes': 'g',
                  'spleen': 'm', 'thymus': 'y'}

    sns.lmplot(data=population, x='count', y='mean_product',
               fit_reg=False, hue='tissue', palette=color_dict)

    population['score'] = population['count'] * population['mean_product']
    for celltype, count, mean_prod, scr in zip(
      population.index.tolist(), population['count'],
      population['mean_product'], population['score']):

        # if not -1 < scr < 8:
        plt.annotate(celltype,
                     xy=(count, mean_prod), xytext=(0, 2),
                     size=5, weight='bold',
                     textcoords='offset points',
                     ha='left', va='top')

    c_min = 0.001
    c_max = 71
    score_min = min(population['score'])
    score_max = max(population['score'])

    for e, score in enumerate(np.linspace(score_min, score_max, 7)):
        c = []
        d = []
        for count in np.arange(c_min, c_max, 0.5):
            c.append(count)
            dif = score/count
            d.append(dif)

        plt.plot(c, d, color='gray', linestyle='-', lw=0.5, zorder=1)
    plt.xlim(0.0, 60)
    plt.ylim(-0.75, 0.75)
    ax.grid(zorder=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir2, 'isobologram_population' + tp + '.pdf'))
    plt.close('all')

    # plot 'population' score chart
    population['celltype'] = population.index
    population.sort_values(by='score', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    pal = {'blood': 'r', 'marrow': 'b', 'nodes': 'g',
           'spleen': 'm', 'thymus': 'y'}
    g = sns.barplot(x='celltype', y='score', data=population, color='k')
    for item in g.get_xticklabels():
        item.set_rotation(90)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir2, 'population_scorechart' + tp + '.pdf'))
    plt.close('all')

    # plot clustermaps and barcharts
    combo_counts = []
    for cond in ['naive_only', 'gl261_only']:
        if cond == 'naive_only':
            s1 = full[['x', 'y', u'\u03C1' + '_naive', 'product', '_merge']][(
                full['_merge'] == cond) | (full['_merge'] == 'both')]
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
                    save_dir2, 'naive_filtered_sig_heatmap' +
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

        elif cond == 'gl261_only':
            s1 = full[['x', 'y', u'\u03C1' + '_gl261', 'product', '_merge']][(
                full['_merge'] == cond) | (full['_merge'] == 'both')]
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
                    save_dir2, 'gl261_filtered_sig_heatmap' +
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

# generate network graphs
    if tp == '':

        color_dict = {
            'inv_naive_only': 'r', 'dir_naive_only': 'b',
            'inv_gl261_only': 'g', 'dir_gl261_only': 'm',
            'dir_naive_dir_gl261': 'y',
            'inv_naive_inv_gl261': 'dodgerblue',
            'inv_naive_dir_gl261': 'black',
            'dir_naive_inv_gl261': 'orchid'}

        d = pd.concat(
            [inv_naive_only, dir_naive_only,
             inv_gl261_only, dir_gl261_only,
             dir_naive_inv_gl261, inv_naive_dir_gl261,
             inv_naive_inv_gl261, dir_naive_dir_gl261], axis=0)

        d['color'] = d['cond'].map(color_dict)

        nodes = set(d['y'].append(d['x']).tolist())

        G = nx.Graph()

        for node in nodes:

            G.add_node(node)

        for row in d.iterrows():
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
                (G[u][v]['abs_weight']**7) * 15 for u, v in K.edges()]

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
                os.path.join(save_dir2, 'network_' + label + '.pdf'))
            plt.close('all')

        # get thymus network
        a = [i for i in d['x'] if i.rsplit('_', 2)[1] == 'thymus']
        b = [i for i in d['y'] if i.rsplit('_', 2)[1] == 'thymus']
        h = d[(d['x'].isin(a)) & (d['y'].isin(b))]

        G = nx.Graph()

        for node in nodes:

            G.add_node(node)

        for row in h.iterrows():
            if row[1]['_merge'] == 'naive_only':

                if row[1][u'\u03C1' + '_naive'] < 0.0:

                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_naive'],
                               style='dotted',
                               color='#4479bb',
                               label='naive_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_naive']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_naive'],
                               style='dotted',
                               color='#d53d69',
                               label='naive_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_naive']))

            elif row[1]['_merge'] == 'gl261_only':

                if row[1][u'\u03C1' + '_gl261'] < 0.0:

                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='dotted',
                               color='m',
                               label='gl261_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_gl261']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='dotted',
                               color='y',
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
        K = nx.Graph()

        for (u, v, b) in G.edges(data=True):

                K.add_edge(u, v, **b)

        plt.figure(figsize=(8, 8))
        plt.axis('off')

        edge_weights = [
            (G[u][v]['abs_weight']**7) * 15 for u, v in K.edges()]

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
            os.path.join(save_dir2, 'thymus_network_' + '.pdf'))
        plt.close('all')

        # get extended thymus network
        o = [i for i in d['x'] if i.rsplit('_', 2)[1] == 'thymus']
        p = [i for i in d['y'] if i.rsplit('_', 2)[1] == 'thymus']

        q = [i for i in d['x'] if i.rsplit('_', 2)[0] == 'Precursor']
        r = [i for i in d['y'] if i.rsplit('_', 2)[0] == 'Precursor']
        #
        # s = [i for i in d['x'] if i == 'Precursor_spleen']
        # t = [i for i in d['y'] if i == 'Precursor_spleen']
        #
        # u = [i for i in d['x'] if i == 'Precursor_nodes']
        # v = [i for i in d['y'] if i == 'Precursor_nodes']
        #
        # w = [i for i in d['x'] if i == 'NK_spleen']
        # x = [i for i in d['y'] if i == 'NK_spleen']

        # y = [i for i in d['x'] if i == 'CD4' + u'\u207A' + 'T' + '_blood']
        # z = [i for i in d['y'] if i == 'CD4' + u'\u207A' + 'T' + '_blood']

        y = [
            i for i in d['x'] if i.rsplit('_', 2)[0] ==
            'F480' + u'\u207A' + 'B']
        z = [
            i for i in d['y'] if i.rsplit('_', 2)[0] ==
            'F480' + u'\u207A' + 'B']

        m = d[(d['x'].isin(o+q+y)) | (d['y'].isin(p+r+z))]
        # +q+s+u+w+y
        # +r+t+v+x+z
        G = nx.Graph()

        for node in nodes:

            G.add_node(node)

        for row in m.iterrows():
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
                               color='grey',
                               label='gl261_only',
                               cond=row[1]['cond'],
                               abs_weight=abs(row[1][u'\u03C1' + '_gl261']))
                else:
                    G.add_edge(u=row[1]['x'], v=row[1]['y'],
                               weight=row[1][u'\u03C1' + '_gl261'],
                               style='solid',
                               color='grey',
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
                                   color='dodgerblue',
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
                                   color='dodgerblue',
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
            (G[u][v]['abs_weight']**7) * 15 for u, v in K.edges()]

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
            os.path.join(save_dir2, 'thymus_network_extended_' + '.pdf'))
        plt.close('all')

sec_dict = {}
for i in pd.concat(
  [v['cond'] for k, v in master.items()], axis=0).unique().tolist():

    a = pd.concat(
        [v for k, v in master.items() if k.startswith(i)], axis=0)

    sec_dict[i] = a

    ips = pd.DataFrame(a['x'].append(a['y']))
    ips.rename(columns={0: 'celltype'}, inplace=True)

    population = pd.DataFrame(ips.groupby(['celltype']).size())
    population.rename(columns={0: 'count'}, inplace=True)
    population.sort_values(by='count', ascending=False, inplace=True)

    # print(i, population)

print(
    'Correlation counts timepoints completed in ' +
    str(datetime.now() - startTime))
