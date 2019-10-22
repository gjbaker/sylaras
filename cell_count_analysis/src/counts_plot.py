import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle
import os
import seaborn as sns

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

project_path = '/Users/gjbaker/projects/gbm_immunosuppression'

# path to pickles from postbot analysis
orthant_pickle_dir = os.path.join(project_path, 'postbot', 'data',
                                  'logicle_e20', 'orthant',
                                  'pickled_global_vars')
aggregate_data_dir = os.path.join(project_path, 'postbot', 'data',
                                  'logicle_e20', 'orthant',
                                  'aggregate_datasets')

# get required postbot pickles
os.chdir(orthant_pickle_dir)
pi_color_dict = open('color_dict.pickle', 'rb')
color_dict = pickle.load(pi_color_dict)

os.chdir(aggregate_data_dir)
overall = pd.read_csv('overall.csv')

# path to correlation analysis save directory
save_dir = os.path.join(
    project_path, 'cell_count_analysis', 'data')
os.makedirs(save_dir)

cellcount_dict = {}
for stat in ['naive', 'gl261']:
    for tp in [7, 14, 30]:
        for tis in ['blood', 'marrow', 'nodes', 'spleen', 'thymus']:
            for rep in [1, 2, 3, 4, 5, 6, 7, 8]:
                print(str(stat), str(tp), str(tis), str(rep) + ' = ' +
                      str(len(overall[(overall['replicate'] == rep) &
                              (overall['tissue'] == tis) &
                              (overall['time_point'] == tp) &
                              (overall['status'] == stat)])))

                count = len(overall[(overall['replicate'] == rep) &
                            (overall['tissue'] == tis) &
                            (overall['time_point'] == tp) &
                            (overall['status'] == stat)])

                cellcount_dict[(stat, tp, tis, rep)] = count

naive_key = [(k) for (k, v) in cellcount_dict.items() if k[0] == 'naive']
naive_data = [(v) for (k, v) in cellcount_dict.items() if k[0] == 'naive']
gl261_key = [(k) for (k, v) in cellcount_dict.items() if k[0] == 'gl261']
gl261_data = [(v) for (k, v) in cellcount_dict.items() if k[0] == 'gl261']

gb_tis = pd.groupby(overall, 'tissue')
tis_weights = 1/len(gb_tis)/gb_tis.size()

naive = pd.DataFrame({'naive_key': naive_key, 'naive_data': naive_data})
gl261 = pd.DataFrame({'gl261_key': gl261_key, 'gl261_data': gl261_data})
df = pd.concat([naive, gl261], axis=1)

df['tis_hue'] = [i[2] for i in df['naive_key']]

ax = sns.barplot(x='naive_key', y='naive_data', data=df, hue='tis_hue',
                 palette=['r', 'b', 'g', 'm', 'y'])

for item in ax.get_xticklabels():
    item.set_rotation(90)
    item.set_size(4)

for stat in ['naive', 'gl261']:

    fig = plt.figure()
    sns.set(style='whitegrid')
    N = len(df)

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = 0.02
    # np.pi / N

    ax = plt.subplot(111, projection='polar')

    bars = ax.bar(
        theta, df[stat + '_data'], width=width, bottom=0.0)

    xlabels_update = [i for i in df[stat + '_key']]

    plt.xticks(theta, xlabels_update)
    color_dict = {'blood': 'r', 'marrow': 'b', 'nodes': 'g',
                  'spleen': 'm', 'thymus': 'y'}
    for item, tissue_name in zip(
      ax.get_xticklabels(), [i[2] for i in df[stat + '_key']]):
        item.set_rotation(10)
        item.set_size(5)
        item.set_weight('normal')
        item.set_color(color_dict[tissue_name])

    for bar, tissue_name in zip(
      bars, [i[2] for i in df['naive_key']]):
        bar.set_facecolor(color_dict[tissue_name])

    thetaticks = np.linspace(0.0, 360, N, endpoint=False)
    ax.set_thetagrids(thetaticks, frac=1.15)

    ax.grid(color='grey', linestyle='-', linewidth=0.15, alpha=1.0)

    ax.spines['polar'].set_color('grey')
    ax.spines['polar'].set_linewidth(0.3)

    # plt.yticks([0, 3, 6, 9, 12], size=7, zorder=5)

    fig.canvas.draw()

    ylabels_update = [i.get_text() for i in ax.get_yticklabels()]

    ax.set_yticklabels(ylabels_update)

    title = plt.title(stat + ' cell counts', size=20, weight='bold', y=1.1)
    title = title.get_text()

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, title + '.pdf'))
