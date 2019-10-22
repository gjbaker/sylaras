import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle
import os
import seaborn as sns
from natsort import natsorted
from scipy.stats import mannwhitneyu

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
postbot_pickle_dir = os.path.join(project_path, 'postbot', 'data',
                                  'logicle_e20', 'orthant',
                                  'pickled_global_vars')

# get required postbot pickles
os.chdir(postbot_pickle_dir)
pi_color_dict = open('color_dict.pickle', 'rb')
color_dict = pickle.load(pi_color_dict)

# path to correlation analysis save directory
save_dir = os.path.join(
    project_path, 'viability_analysis', 'data')

os.chdir(os.path.join(project_path, save_dir))
data = pd.read_csv('viability.csv')
data.drop([0, 1, 2, 3, 4], axis=0, inplace=True)
data.drop(['Depth', '#Cells'], axis=1, inplace=True)
data = data.iloc[::8, :]
data.rename(
    columns={'Name': 'cond', 'Statistic': 'percent_dead'}, inplace=True)
data.reset_index(drop=True, inplace=True)
for row in data.iterrows():
    cond = row[1][0].split('_', 6)[2:6]
    cond = '_'.join(cond)
    data.loc[row[0], 'cond'] = cond

data['time_point'], data['status'], data['tissue'], data[
    'replicate'] = data['cond'].str.split('_', 4).str

tp_dict = {
    '7': '7', '14': '14', '23': '30', '31': '30', '35': '30', '36': '30'}

data['cond'] = [tp_dict[''.join(i.split('_', 3)[0:1])] + '_' +
                ''.join(i.split('_', 3)[3:4]) for i in data['cond']]

cols = ['status', 'time_point', 'tissue', 'replicate', 'cond', 'percent_dead']
data = data[cols]

data['time_point'] = [tp_dict[i] for i in data['time_point']]

tp_sorter = natsorted(data['time_point'])
tp_sorter_idx = dict(zip(tp_sorter, range(len(tp_sorter))))
data['tp_rank'] = data['time_point'].map(tp_sorter_idx)

data['mean'] = 0.0
data['sem'] = 0.0

# for name, group in data.groupby(['status', 'time_point', 'tissue']):

#     if name[0] == 'naive':
#         data = data.append({'status': name[0], 'time_point': name[1],
#                             'tissue': name[2], 'replicate': '9',
#                             'cond': name[1] + '_' + str(9),
#                             'percent_dead': group.mean()[0],
#                             'tp_rank': tp_sorter_idx[name[1]],
#                             'mean': group.mean()[0], 'sem': group.sem()[0]},
#                            ignore_index=True)
#     elif name[0] == 'gl261':
#         data = data.append({'status': name[0], 'time_point': name[1],
#                             'tissue': name[2], 'replicate': '0',
#                             'cond': name[1] + '_' + str(0),
#                             'percent_dead': group.mean()[0],
#                             'tp_rank': tp_sorter_idx[name[1]],
#                             'mean': group.mean()[0], 'sem': group.sem()[0]},
#                            ignore_index=True)

data.sort_values(
    by=['tp_rank', 'tissue', 'status', 'replicate'],
    ascending=[True, True, False, True], inplace=True)
data.drop('tp_rank', axis=1, inplace=True)

fig = plt.figure()
sns.set(style='whitegrid')
N = len(data)

theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = 0.02
# np.pi / N

ax = plt.subplot(111, projection='polar')

bars = ax.bar(
    theta, data['percent_dead'], yerr=data['sem'], width=width, bottom=0.0)

# offset = 2
#
# ax.plot(theta, ratio_data, lw=1.5)
# ax.set_rmax(1.5 * offset)

xlabels_update = [i for i in data['cond']]

plt.xticks(theta, xlabels_update)

for item, tissue_name, sta in zip(
  ax.get_xticklabels(), data['tissue'].tolist(), data['status'].tolist()):
    item.set_rotation(10)
    if sta == 'naive':
        item.set_alpha(1.0)
    if sta == 'gl261':
        item.set_alpha(0.5)
    item.set_size(5)
    item.set_weight('bold')
    item.set_color(color_dict[tissue_name])

for bar, tissue_name, sta in zip(
  bars, data['tissue'].tolist(), data['status'].tolist()):
    bar.set_facecolor(color_dict[tissue_name])
    if sta == 'naive':
        bar.set_alpha(1.0)
    if sta == 'gl261':
        bar.set_alpha(0.4)

thetaticks = np.linspace(0.0, 360, N, endpoint=False)
ax.set_thetagrids(thetaticks, frac=1.15)

ax.grid(color='grey', linestyle='-', linewidth=0.15, alpha=1.0)

ax.spines['polar'].set_color('grey')
ax.spines['polar'].set_linewidth(0.3)

plt.yticks([0, 3, 6, 9, 12], size=7, zorder=5)
plt.ylim([-12, 12])
fig.canvas.draw()

ylabels_update = [i.get_text() + '%' for i in ax.get_yticklabels()]

ax.set_yticklabels(ylabels_update)

title = plt.title('% dead cells', size=20, weight='bold', y=1.1)
title = title.get_text()

plt.tight_layout()

plt.savefig(os.path.join(save_dir, title + '.pdf'))

# get p values
for name, group in data.groupby(['time_point', 'tissue']):
    print(name)
    naive_data = group['percent_dead'][group['status'] == 'naive']
    gl261_data = group['percent_dead'][group['status'] == 'gl261']

    u_value, p_value = mannwhitneyu(
        gl261_data, naive_data, use_continuity=True, alternative='two-sided')
    print(u_value, p_value)
    if u_value <= 13:
        if p_value <= 0.05:
            print(u_value, p_value)
