import os
import pandas as pd
import numpy as np
from pyeda.inter import exprvar
from pyeda.inter import expr
from pyeda.inter import expr2truthtable
import collections
import itertools
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import math
from matplotlib.gridspec import GridSpec
import operator

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

os.chdir('/Users/gjbaker/projects/gbm_immunosuppression/flowsom/data')

# read and tidy R-generated file path metadata column
metadata = pd.read_csv('metadata.csv', index_col=0)
metadata.reset_index(drop=True, inplace=True)

# drop .fcs extension
metadata = pd.DataFrame(
    metadata.metadata.str.split('.', 1).tolist(),
    columns=['metadata', 'ext']).loc[:, 'metadata']

# split file path metadata column into multiple metadata columns
breakout = pd.DataFrame(
    metadata.str.split('_', 3).tolist(),
    columns=['time point', 'tissue', 'status', 'replicate'])
del(metadata)

# read R-generated flowsom df
df = pd.read_csv('flowsom.csv', index_col=0)
df.reset_index(drop=True, inplace=True)

FULL_final = pd.concat([breakout, df], axis=1)
FULL_final.to_csv('FULL_final.csv', index=False)
del(df)


# make banner to introduce each RUNNING MODULE
def banner(MODULE_title):

    print('=' * 70)
    print(MODULE_title)
    print('=' * 70)


def random_subset(data):
    banner('RUNNING MODULE: random_subset')

    # Calculate random sample weighting to normalize cell counts by tissue.
    groups = data.groupby('tissue')
    tissue_weights = pd.DataFrame({
        'weights': 1 / (groups.size() * len(groups))
    })

    weights = pd.merge(
        data[['tissue']], tissue_weights, left_on='tissue', right_index=True
    )

    sample = data.sample(
        n=10000000, replace=False, weights=weights['weights'],
        random_state=1, axis=0
    )

    sample.reset_index(drop=True, inplace=True)

    return sample


sample = random_subset(FULL_final)

del(FULL_final)
sample.to_csv('sample.csv', index=False)
sample_copy = sample.copy()


def data_discretization(sample):

    banner('RUNNING MODULE: data_discretization')

    channel_list = sample.columns[7:18]

    channel_columns = {}
    for channel in channel_list:
        channel_columns[channel] = sample.loc[:, channel].values
    print()
    for key, value in channel_columns.items():
        print('Converting ' + key + ' protein expression data into its '
              'Boolean representation.')
        for i, v in enumerate(value):
            if v > 0:
                value[i] = 1
            elif v <= 0:
                value[i] = 0
    Boo_data = sample.loc[:, channel_list].astype(int)
    Boo_data = Boo_data[channel_list.sort_values()]
    channel_list_update_dict = {
        'b220': 'B220', 'cd45': 'CD45', 'cd11b': 'CD11b', 'cd11c': 'CD11c',
        'cd3e': 'CD3e', 'cd4': 'CD4', 'cd49b': 'CD49b', 'cd8a': 'CD8a',
        'f480': 'F480', 'ly6c': 'Ly6C', 'ly6g': 'Ly6G'}
    Boo_data1 = Boo_data.rename(columns=channel_list_update_dict)
    Boo_data2 = pd.concat([sample.iloc[:, 0:7], Boo_data1], axis=1)

    # correct for CD49b in blood
    NK = Boo_data2['CD49b'][
        ((Boo_data2['B220'] == 0) & (Boo_data2['CD11b'] == 1) &
         (Boo_data2['CD11c'] == 0) & (Boo_data2['CD3e'] == 0) &
         (Boo_data2['CD4'] == 0) & (Boo_data2['CD45'] == 1) &
         (Boo_data2['CD49b'] == 1) & (Boo_data2['CD8a'] == 0) &
         (Boo_data2['F480'] == 0) & (Boo_data2['Ly6C'] == 0) &
         (Boo_data2['Ly6G'] == 0))]

    non_NK = Boo_data2['CD49b'][
        ~((Boo_data2['B220'] == 0) & (Boo_data2['CD11b'] == 1) &
          (Boo_data2['CD11c'] == 0) & (Boo_data2['CD3e'] == 0) &
          (Boo_data2['CD4'] == 0) & (Boo_data2['CD45'] == 1) &
          (Boo_data2['CD49b'] == 1) & (Boo_data2['CD8a'] == 0) &
          (Boo_data2['F480'] == 0) & (Boo_data2['Ly6C'] == 0) &
          (Boo_data2['Ly6G'] == 0))]

    non_NK[:] = 0

    new_cd49b_col = non_NK.append(NK).sort_index()

    del NK
    del non_NK

    Boo_data2['CD49b'] = new_cd49b_col

    channel_list_update = sorted(
        [v for k, v in channel_list_update_dict.items()])

    return Boo_data2, channel_list, channel_list_update


Boo_data2, channel_list, channel_list_update = data_discretization(sample_copy)
del(sample_copy)


def Boolean_classifier(Boo_data):

    banner('RUNNING MODULE: Boolean_classifier')

    # define the input variable names
    B220, CD3e, CD4, CD8a, Ly6G, Ly6C, F480, CD11b, CD11c, CD49b, CD45 = map(
        exprvar, ['B220', 'CD3e', 'CD4', 'CD8a', 'Ly6G', 'Ly6C', 'F480',
                  'CD11b', 'CD11c', 'CD49b', 'CD45'])

    # Care dictionaries
    dict_yes = {}

    # dict_yes['test_yes'] = {'B220', 'CD11b', 'CD11c', 'CD3e', 'CD4', 'CD45',
    # 'CD49b', 'CD8a', 'F480', 'Ly6C', 'Ly6G'}

    # B cells
    dict_yes['B_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}
    dict_yes['CD45negB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['F480posB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD8aposB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # cytotoxic T cells
    dict_yes['CD8T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposCD8T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', 'Ly6C', '~Ly6G'}
    dict_yes['B220posCD8T_yes'] = {
        'B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # helper T cells
    dict_yes['CD4T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', 'CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposCD4T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', 'CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}

    # myeloid DCs
    dict_yes['DC_yes'] = {
        '~B220', 'CD11b', 'CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', 'Ly6C', '~Ly6G'}

    # NK cells
    dict_yes['NK_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', 'CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # PMN cells
    dict_yes['PMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', 'Ly6G'}
    dict_yes['CD45negPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', 'Ly6G'}
    dict_yes['Ly6CnegPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', 'Ly6G'}
    dict_yes['F480posPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', 'Ly6C', 'Ly6G'}
    dict_yes['F480posLy6CnegPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', 'Ly6G'}

    # Monocytes
    dict_yes['Mono_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}
    dict_yes['Ly6CnegMono_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD11bnegMono_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}

    # Macrophages
    dict_yes['Mac_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposMac_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', 'Ly6C', '~Ly6G'}
    dict_yes['CD11bnegMac_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', '~Ly6G'}

    # Double positive T cells
    dict_yes['DPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', 'CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD45negDPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', 'CD4', '~CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD3eposDPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', 'CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # Immature single positive T cells
    dict_yes['ISPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD45negISPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # Lymphoid tissue inducer cells
    dict_yes['LTi_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', 'CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # Double negative T cells
    dict_yes['DNT_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposDNT_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}

    # Precursor immune cells
    dict_yes['Precursor_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD45negPrecursor_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}

    dict_yes = collections.OrderedDict(sorted(dict_yes.items()))

    # Don't care dictionaries
    dict_no = {}

    # dict_no['test_no'] = {}

    # B cells
    dict_no['B_no'] = {}
    dict_no['Ly6CposB_no'] = {}
    dict_no['CD45negB_no'] = {}
    dict_no['F480posB_no'] = {}
    dict_no['CD8aposB_no'] = {}

    # cytotoxic T cells
    dict_no['CD8T_no'] = {}
    dict_no['Ly6CposCD8T_no'] = {}
    dict_no['B220posCD8T_no'] = {}

    # helper T cells
    dict_no['CD4T_no'] = {}
    dict_no['Ly6CposCD4T_no'] = {}

    # myeloid DCs
    dict_no['DC_no'] = {}

    # NK cells
    dict_no['NK_no'] = {}

    # PMN cells
    dict_no['PMN_no'] = {}
    dict_no['CD45negPMN_no'] = {}
    dict_no['Ly6CnegPMN_no'] = {}
    dict_no['F480posPMN_no'] = {}
    dict_no['F480posLy6CnegPMN_no'] = {}

    # Monocytes
    dict_no['Mono_no'] = {}
    dict_no['Ly6CnegMono_no'] = {}
    dict_no['CD11bnegMono_no'] = {}

    # Macrophages
    dict_no['Mac_no'] = {}
    dict_no['Ly6CposMac_no'] = {}
    dict_no['CD11bnegMac_no'] = {}

    # Double positive T cells
    dict_no['DPT_no'] = {}
    dict_no['CD45negDPT_no'] = {}
    dict_no['CD3eposDPT_no'] = {}

    # Immature single positive T cells
    dict_no['ISPT_no'] = {}
    dict_no['CD45negISPT_no'] = {}

    # Lymphoid tissue inducer cells
    dict_no['LTi_no'] = {}

    # Double negative T cells
    dict_no['DNT_no'] = {}
    dict_no['Ly6CposDNT_no'] = {}

    # Precursor immune cells
    dict_no['Precursor_no'] = {}
    dict_no['CD45negPrecursor_no'] = {}

    dict_no = collections.OrderedDict(sorted(dict_no.items()))

    # Boolean expression generator
    exp = {}
    truth = {}
    vectors = pd.DataFrame()
    for (k1, v1), (k2, v2) in zip(dict_no.items(), dict_yes.items()):
        cell_type = '%s' % ((k1.rsplit('_', 2)[0]))
        current_dont_cares = []
        current_cares = []
        q = len(v1)
        s = 'Or('
        for x1 in v1:
            current_dont_cares.append(x1)
            a = '~' + x1
            current_dont_cares.append(a)
        for x2 in v2:
            current_cares.append(x2)

        for i, v in enumerate(list(itertools.combinations
                                   (current_dont_cares, q))):
            c = list(v)
            d = [x[-3:] for x in c]
            if len(d) == len(set(d)):
                f = str(current_cares + c)[1:-1].replace("'", "")
                f = 'And(' + f + '), '
                s += f
        w = s + ')'
        w = w[:-3] + w[-3:-2].replace(",", "") + \
            w[-2:-1].replace(" ", "") + w[-1:]
        exp[cell_type] = expr(w)
        truth[cell_type] = expr2truthtable(exp[cell_type])
        v = len(list(exp[cell_type].satisfy_all()))
        vector = pd.DataFrame(exp[cell_type].satisfy_all())
        vector.columns = [str(u) for u in vector.columns]
        vector['cell_type'] = cell_type
        print(vector['cell_type'].to_string(index=False))
        print((vector.loc[:, vector.columns != 'cell_type'])
              .to_string(index=False))
        print()
        vectors = vectors.append(vector)
    vectors = vectors.reset_index(drop=True)

    # show duplicate vector report
    vector_counts = vectors.sort_values(channel_list_update) \
        .groupby(channel_list_update).count()
    dupe_vectors = vector_counts[vector_counts.cell_type > 1]
    dupe_vectors = dupe_vectors.reset_index() \
        .rename(columns={'cell_type': 'count'})
    dupe_report = pd.merge(vectors, dupe_vectors, on=channel_list_update)
    if not dupe_report.empty:
        print('Duplicate vector report:')
        print(dupe_report)
        dupe_report.to_csv('duplicate_report.csv')
        print()

    # conflict resolution
    if not dupe_report.empty:
        vectors = pd.DataFrame()

        # specify vector assignments to drop from the classifier
        # (put into cons square brackets) dupe_report.loc[1:1, cols2].values,
        # dupe_report.loc[2:2, cols2].values
        conflicts = []

        conflicts = [val for sublist in conflicts for val in sublist]
        for c in range(len(conflicts)):
            j = vectors[(vectors['B220'] != conflicts[c][0]) |
                        (vectors['CD11b'] != conflicts[c][1]) |
                        (vectors['CD11c'] != conflicts[c][2]) |
                        (vectors['CD3e'] != conflicts[c][3]) |
                        (vectors['CD4'] != conflicts[c][4]) |
                        (vectors['CD49b'] != conflicts[c][5]) |
                        (vectors['CD8a'] != conflicts[c][6]) |
                        (vectors['F480'] != conflicts[c][7]) |
                        (vectors['Ly6C'] != conflicts[c][8]) |
                        (vectors['Ly6G'] != conflicts[c][9]) |
                        (vectors['cell_type'] != conflicts[c][10])]
            vectors = j
            vectors = j

    # print classifier statistics
    count = vectors['cell_type'].value_counts().tolist()
    total_vectors = sum(count)
    name = vectors['cell_type'].value_counts().index.tolist()
    print('Classifier report:')
    print(str(total_vectors) +
          ' unique vectors are specified under the current classifer.')
    print()
    print('Of which,...')
    for count, name in zip(count, name):
        print(str(count) + ' satisfies the ' + name + ' cell phenotype.')
    print()

    classified = pd.merge(Boo_data2, vectors, how='left',
                          on=channel_list_update)
    classified = classified.fillna(value='unspecified')
    count2 = classified['cell_type'].value_counts()
    percent_coverage = (sum(count2) - count2['unspecified']) \
        / sum(count2) * 100
    print('The current classifier covers ' + str(percent_coverage) +
          ' percent of cells in the dataset,'
          ' which contains ' +
          str(len(Boo_data2.loc[:, channel_list_update].drop_duplicates())) +
          ' unique vectors.')
    print()

    # check residual, unclassified single-cell data
    unspecified = classified[classified['cell_type'] == 'unspecified']
    unspecified = unspecified.groupby(channel_list_update).size() \
        .reset_index().rename(columns={0: 'count'})
    unspecified = unspecified.sort_values(by='count', ascending=False)
    if not unspecified.empty:
        print('unspecified vector report:')
        print(unspecified)
        print('The sum of the unspecified cells is: ' +
              str(unspecified['count'].sum()))
        print()

    return classified


classified = Boolean_classifier(Boo_data2)
del(Boo_data2)


def split_combine_celltypes(classified):

    banner('RUNNING MODULE: split_combine_celltypes')

    classified.loc[
        (classified.cell_type == 'Mac') &
        (classified.ssc > 100000), 'cell_type'] = 'Eo'

    classified.loc[
        (classified.cell_type == 'Ly6CposMac') &
        (classified.ssc > 125000), 'cell_type'] = 'Ly6CposEo'

    classified.loc[
        (classified.cell_type == 'CD45negPMN'),
        'cell_type'] = 'PMN'
    classified.loc[
        (classified.cell_type == 'PMN'),
        'CD45'] = 1

    classified.loc[
        (classified.cell_type == 'CD45negPrecursor'),
        'cell_type'] = 'Precursor'
    classified.loc[
        (classified.cell_type == 'Precursor'),
        'CD45'] = 1

    classified.loc[
        (classified.cell_type == 'CD45negB'),
        'cell_type'] = 'B'
    classified.loc[
        (classified.cell_type == 'B'),
        'CD45'] = 1

    classified.loc[
        (classified.cell_type == 'CD45negDPT'),
        'cell_type'] = 'DPT'
    classified.loc[
        (classified.cell_type == 'DPT'),
        'CD45'] = 1

    classified.loc[
        (classified.cell_type == 'CD45negISPT'),
        'cell_type'] = 'ISPT'
    classified.loc[
        (classified.cell_type == 'ISPT'),
        'CD45'] = 1

    classified.to_csv('classified.csv', index=False)

    s = classified.groupby(channel_list_update)
    print('There are ' + str(s.ngroups) + ' BIPs in the dataframe')

    return classified


classified = split_combine_celltypes(classified)


def vector_coverage(classified, alpha):

    banner('RUNNING MODULE: vector_coverage')

    total_vectors1 = classified
    total_vectors2 = total_vectors1.groupby(
        ['tissue', 'time point', 'replicate', 'status', 'cell_type'] +
        channel_list_update).size().reset_index().rename(columns={0: 'count'})
    total_vectors3 = total_vectors2.sort_values(
        by=['tissue', 'time point', 'replicate', 'status', 'count'],
        ascending=[True, True, True, False, False]).reset_index(drop=True)

    total_vectors4 = total_vectors1.groupby(
        ['tissue', 'time point', 'replicate', 'status']).size().reset_index() \
        .rename(columns={0: 'count'})

    alpha = alpha
    alpha_vectors_list = []
    condition_frame_alpha_dict = {}
    for s, group in enumerate(total_vectors3.groupby(['tissue', 'time point',
                                                     'replicate', 'status'])):

        condition_name = group[0]
        group = pd.DataFrame(group[1])
        vector_list = []
        denom = total_vectors4['count'][s]
        for i in group.iterrows():
            if i[1][16]/denom >= alpha:
                vector_list.append(group.loc[i[0], :])
                alpha_vectors_list.append(i[1][4:16])
            else:
                break

        condition_frame_alpha_dict[condition_name] = pd.DataFrame(vector_list)

    alpha_vectors = pd.DataFrame(alpha_vectors_list)
    subset_to_drop_on = channel_list_update.append('cell_type')
    alpha_vectors.drop_duplicates(
        subset=subset_to_drop_on, inplace=True)
    alpha_vectors.reset_index(drop=True, inplace=True)
    alpha_vectors.to_csv(
        'alpha_' + str(alpha * 100) + '%_vectors' + '.csv')

    condition_frame_alpha = pd.concat(condition_frame_alpha_dict,
                                      axis=0).reset_index(drop=True)

    condition_frame_alpha_unique = condition_frame_alpha \
        .drop_duplicates(channel_list_update).reset_index(drop=False)
    condition_frame_alpha_unique = condition_frame_alpha_unique \
        .rename(columns={'index': 'vector_index'})

    condition_frame_alpha_index = condition_frame_alpha.merge(
        condition_frame_alpha_unique[['vector_index'] + channel_list_update],
        how='left', on=channel_list_update)

    # get vectors unique to tissue
    tissue_sets = []
    tissue_name = []
    for tissue in condition_frame_alpha_index['tissue'].unique():

        n = condition_frame_alpha_index[
            condition_frame_alpha_index['cell_type'] != 'unspecified']
        idx_tissue = n[n['tissue'] == tissue]['vector_index'].tolist()
        tissue_sets.append(set(idx_tissue))
        tissue_name.append(str(tissue))

    for j in tissue_name:
        if j == 'blood':
            try:
                blood_set_dif = list(tissue_sets[0] - tissue_sets[1] -
                                     tissue_sets[2] - tissue_sets[3] -
                                     tissue_sets[4])
                blood_set_dif_frame = pd.DataFrame(blood_set_dif)
                blood_set_dif_frame = blood_set_dif_frame \
                    .rename(columns={0: 'vector_index'})
                blood_unique_vectors = condition_frame_alpha_index \
                    .merge(blood_set_dif_frame, how='inner', on='vector_index')
                print('Vectors unique to ' + j + ' at the ' +
                      str(float(alpha * 100)) + '% cutoff')
                print(blood_unique_vectors)
                print()
            except Exception:
                pass
        elif j == 'marrow':
            try:
                marrow_set_dif = list(tissue_sets[1] - tissue_sets[0] -
                                      tissue_sets[2] - tissue_sets[3] -
                                      tissue_sets[4])
                marrow_set_dif_frame = pd.DataFrame(marrow_set_dif)
                marrow_set_dif_frame = marrow_set_dif_frame \
                    .rename(columns={0: 'vector_index'})
                marrow_unique_vectors = condition_frame_alpha_index \
                    .merge(marrow_set_dif_frame, how='inner',
                           on='vector_index')
                print('Vectors unique to ' + j + ' at the ' +
                      str(float(alpha * 100)) + '% cutoff')
                print(marrow_unique_vectors)
                print()
            except Exception:
                pass
        elif j == 'nodes':
            try:
                nodes_set_dif = list(tissue_sets[2] - tissue_sets[0] -
                                     tissue_sets[1] - tissue_sets[3] -
                                     tissue_sets[4])
                nodes_set_dif_frame = pd.DataFrame(nodes_set_dif)
                nodes_set_dif_frame = nodes_set_dif_frame \
                    .rename(columns={0: 'vector_index'})
                nodes_unique_vectors = condition_frame_alpha_index \
                    .merge(nodes_set_dif_frame, how='inner', on='vector_index')
                print('Vectors unique to ' + j + ' at the ' +
                      str(float(alpha * 100)) + '% cutoff')
                print(nodes_unique_vectors)
                print()
            except Exception:
                pass
        elif j == 'spleen':
            try:
                spleen_set_dif = list(tissue_sets[3] - tissue_sets[0] -
                                      tissue_sets[1] - tissue_sets[2] -
                                      tissue_sets[4])
                spleen_set_dif_frame = pd.DataFrame(spleen_set_dif)
                spleen_set_dif_frame = spleen_set_dif_frame \
                    .rename(columns={0: 'vector_index'})
                spleen_unique_vectors = condition_frame_alpha_index \
                    .merge(spleen_set_dif_frame, how='inner',
                           on='vector_index')
                print('Vectors unique to ' + j + ' at the ' +
                      str(float(alpha * 100)) + '% cutoff')
                print(spleen_unique_vectors)
                print()
            except Exception:
                pass
        elif j == 'thymus':
            try:
                thymus_set_dif = list(tissue_sets[4] - tissue_sets[0] -
                                      tissue_sets[1] - tissue_sets[2] -
                                      tissue_sets[3])
                thymus_set_dif_frame = pd.DataFrame(thymus_set_dif)
                thymus_set_dif_frame = thymus_set_dif_frame \
                    .rename(columns={0: 'vector_index'})
                thymus_unique_vectors = condition_frame_alpha_index \
                    .merge(thymus_set_dif_frame, how='inner',
                           on='vector_index')
                print('Vectors unique to ' + j + ' at the ' +
                      str(float(alpha * 100)) + '% cutoff')
                print(thymus_unique_vectors)
                print()
            except Exception:
                pass

    # assess vector tissue coverage
    slider_condition_vectors_dict = {}
    percent_range = list(range(101))
    for s, group in enumerate(total_vectors3.groupby(['tissue', 'time point',
                                                      'replicate', 'status'])):
        condition_name = group[0]
        group = pd.DataFrame(group[1])
        slider_condition_vectors_dict[condition_name] = [[], []]
        for j in percent_range:
            # print('Counting ' + str(condition_name) +
            #       ' vectors at the ' + str(j) +
            #       '%' + ' percent cutoff.')
            alpha_slide = j * 0.01
            vector_list = []
            denom = total_vectors4['count'][s]
            for i in group.iterrows():
                if i[1][16]/denom >= alpha_slide:
                    vector_list.append(group.loc[i[0], :])
            condition_frame = pd.DataFrame(vector_list)
            if vector_list:
                condition_frame_unique = condition_frame \
                    .drop_duplicates(channel_list_update) \
                    .reset_index(drop=False)
                num_cases = len(condition_frame_unique)
            else:
                num_cases = 0
            slider_condition_vectors_dict[condition_name][0].append(j)
            slider_condition_vectors_dict[condition_name][1].append(num_cases)

    return alpha_vectors


alpha_vectors = vector_coverage(classified, 0.01)


def overall(sample, classified):

    banner('RUNNING MODULE: overall')

    print('Combining continuous and Boolean classifier'
          ' results into overall DataFrame.')

    expression_values = sample.loc[
        :, list(channel_list) + ['fsc', 'ssc']]
    overall = pd.merge(
        classified, expression_values, left_index=True, right_index=True)
    overall.drop(['fsc_x', 'ssc_x'], axis=1, inplace=True)
    overall = overall.rename(columns={'fsc_y': 'fsc', 'ssc_y': 'ssc'})
    overall_cols = ['time point', 'tissue', 'status', 'replicate', 'cluster',
                    'B220', 'CD11b', 'CD11c', 'CD3e', 'CD4', 'CD45', 'CD49b',
                    'CD8a', 'F480', 'Ly6C', 'Ly6G', 'cell_type',
                    'fsc', 'ssc', 'b220', 'cd11b', 'cd11c', 'cd3e', 'cd4',
                    'cd45', 'cd49b', 'cd8a', 'f480', 'ly6c', 'ly6g']
    overall = overall[overall_cols]
    overall.to_csv('overall.csv', index=False)
    print()

    return overall


overall = overall(sample, classified)


def cluster_entropy(overall):

    banner('RUNNING MODULE: cluster_entropy')

    # compute Shannon Entropy
    entropy_df = pd.DataFrame(
        index=sorted(overall.cluster.unique()),
        columns=overall.cell_type.unique())

    entropy_list = []
    for cluster, group in sorted(overall.groupby(['cluster'])):
        counts_list = []
        for celltype in overall.cell_type.unique():
            if any(group.cell_type == celltype) is True:
                counts = sum(group.cell_type == celltype)
                counts_list.append(counts)
            else:
                counts_list.append(0)
        probs = np.array(counts_list).astype(float)/np.array(
            counts_list).sum()
        entropy = scipy.stats.entropy(probs, base=2)
        entropy_list.append(entropy)
        entropy_df.loc[cluster] = probs
    entropy_df['entropy'] = entropy_list

    order = list(
        overall.groupby('cluster').size().sort_values(
            ascending=False).index)

    entropy_df = entropy_df.reindex(order)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    print('Plotting stacked celltype cluster distribution.')
    entropy_df.loc[:, entropy_df.columns != 'entropy'].plot(
        kind='bar', stacked=True, edgecolor='none', zorder=5,
        figsize=(9, 7), ax=ax1)

    ax1.set_title('cluster distribution by immunophenotype',
                  y=1.05, size=20, fontweight='bold')

    for item in ax1.get_xticklabels():
        item.set_rotation(90)
        item.set_weight('normal')

    ax1.set_xlabel(xlabel='immunophenotype',
                   size=15, weight='bold')
    ax1.set_ylabel(ylabel='% of immunophenotype',
                   size=15, weight='bold')

    print('Superimposing information entropy.')
    ax2.plot(
        list(entropy_df['entropy'].index.map(str)),
        entropy_df['entropy'], ls='--', c='k')

    ax2.set_ylabel(ylabel='information entropy (S)',
                   size=15, weight='bold')
    # ax1.get_legend().remove()
    plt.tight_layout()
    plt.savefig('information_entropy' + '.pdf')
    plt.close('all')


cluster_entropy(overall)


def mapper(classified):

    banner('RUNNING MODULE: mapper')

    print('Mapping phenograph clusters to Boolean classifier.')
    cluster_dist = classified.groupby(['cluster', 'cell_type']).size()
    cluster_dist = cluster_dist.groupby(['cluster'])
    cluster_dist = cluster_dist.apply(lambda x: x.sort_values(ascending=False))
    cluster_dist.index = cluster_dist.index.droplevel(level=0)

    print('Plotting cluster distribution.')
    fig, ax = plt.subplots(figsize=(60, 10))

    df = cluster_dist.reset_index().rename(columns={0: 'count'})
    pivot = df.pivot('cluster', 'cell_type', 'count').fillna(value=0.0)

    bar_width = 0.05
    percentage = 0.5
    cluster_dict = {}
    for i in pivot.iterrows():

        spacer = 0
        df_filt = pd.DataFrame()

        for j in i[1].sort_values(ascending=False).reset_index().index:

            r = (i[1].sort_values(ascending=False)
                 .reset_index().iloc[j])
            cluster = r.index[1]
            celltype = r[0]
            y = r[1]

            append = pd.DataFrame(
                {'cluster': [cluster], 'cell_type': [celltype], 'count': [y]})

            df_filt = df_filt.append(append)

        df_filt.reset_index(drop=True, inplace=True)

        cutoff = df_filt['count'].sum()*percentage
        max_idx = [i for i, x in enumerate(
            np.cumsum(
                df_filt['count'])) if x >= cutoff][0]
        indices = list(range(0, max_idx+1))

        label_list = []

        for k in df_filt.iterrows():
            rects = ax.bar(k[1][1]+spacer, k[1][2], width=bar_width, color='k')

            spacer += bar_width

            for rect in rects:

                if k[0] in indices:

                    label = k[1][0].replace(
                        'neg', '$^-$').replace('pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1')

                    label_list.append(k[1][0])

                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                            '%s' % (label),
                            ha='center', va='bottom')

                    cluster_dict[cluster] = ', '.join(label_list)

    plt.xticks(np.arange(pivot.index[0], pivot.index[-1]+1, 1.0))

    ax.set_title('cluster distribution',
                 y=1.05, size=20, fontweight='bold')

    ax.set_xlabel(xlabel='cluster',
                  size=15, weight='bold')
    ax.set_ylabel(ylabel='cell count',
                  size=15, weight='bold')

    plt.tight_layout()
    plt.savefig('cluster_distribution' + '.pdf')
    plt.close('all')

    consensus_Boo = cluster_dist.unstack(level=0).idxmax()
    classified_choice_map = pd.merge(classified, pd.DataFrame(
        {'consensus': consensus_Boo}), left_on='cluster', right_index=True)
    consensus_Boo = pd.DataFrame(consensus_Boo)
    consensus_Boo.columns = ['consensus']
    consensus_Boo.reset_index(level=0, inplace=True)
    consensus_Boo['cluster'] = consensus_Boo.cluster.map(str) + ' ' + \
        '(' + consensus_Boo.consensus + ')'
    consensus_Boo = consensus_Boo.drop('consensus', axis=1)
    consensus_Boo = consensus_Boo['cluster']
    keys = [int(i.split(' ', 2)[0]) for i in consensus_Boo]
    cluster_map = dict(zip(keys, consensus_Boo))

    return cluster_dict, cluster_map


cluster_dict, cluster_map = mapper(classified)


def find_cells(classified):

    find_cells_dir = os.path.join(os.getcwd(), 'find_cells')
    if not os.path.exists(find_cells_dir):
        os.makedirs(find_cells_dir)

    all_classes = set(classified['cell_type'].unique())
    identified_classes = set([j for i in cluster_dict.values() for j in i])
    missing_classes = all_classes - identified_classes

    for clss in missing_classes:
        fig, ax = plt.subplots()
        data = classified[classified['cell_type'] == clss].groupby(
            'cluster').size()/len(
                classified[classified['cell_type'] == clss])*100
        data1 = data[data >= 1.0]
        data1.index = data1.index.map(str)
        data2 = pd.Series(data[data < 1.0].sum())
        data2.index = ['others']

        data1 = data1.append(data2)

        labels = data1.index.tolist()

        patches, texts, autotexts = plt.pie(
            data1, shadow=False, autopct='%1.1f%%', startangle=90)

        plt.axis('equal')

        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

        ax.set_title(clss)
        plt.legend(labels, loc='upper right')
        plt.savefig(os.path.join(find_cells_dir, clss + '.pdf'))
        plt.close('all')
    print()


find_cells(classified)

# NOTE: postbot_update.py extract_TRUNC() grabs Logicle-transformed,
# zero-centered data from
# '/Volumes/SysBio/SORGER PROJECTS/gbmi/data/output/phenobot/1hiZ36K/
# logicle/8000/euclidean/20/2/00/truncated'.
# looks like Gabe set the negative values to 0 then normalized
# the positive values per channel.

# Recapitulate the above mentioned process.
# channel_cols = overall[overall.columns[-13:]].copy()
# channel_cols.iloc[channel_cols < 0] = 0
#
# min_max_scaler = preprocessing.MinMaxScaler()
# scaled = min_max_scaler.fit_transform(channel_cols)
# scaled_df = pd.DataFrame(scaled, columns=channel_cols.columns)
#
# overall_normalized = overall.copy()
# overall_normalized.update(scaled_df)

# get un-normalized positive signal intensity only
channel_cols = overall[overall.columns[-13:]].copy()
channel_cols.iloc[channel_cols < 0] = 0

overall_zeroed = overall.copy()
overall_zeroed.update(channel_cols)


def phenograph_heatmaps(overall_zeroed):

    banner('RUNNING MODULE: phenograph_heatmaps')

    heatmap_df = pd.DataFrame(
        index=sorted(overall_zeroed['cluster'].unique()),
        columns=overall_zeroed.columns[-11:])

    for cluster, group in overall_zeroed.groupby('cluster'):
        for channel in group.iloc[:, -11:]:
            heatmap_df.at[cluster, channel] = group[channel].mean()

    heatmap_df = heatmap_df.astype(float)

    order = list(
        overall_zeroed.groupby('cluster').size().sort_values(
            ascending=False).index)

    heatmap_df = heatmap_df.reindex(order)

    grid_kws = {'height_ratios': (0.9, 0.05), 'hspace': 0}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax.set_title('average_expression (global)')
    ax = sns.heatmap(heatmap_df.T, ax=ax, cbar_ax=cbar_ax, square=True,
                     linewidths=0.25, cmap='Greens', linecolor='k',
                     cbar_kws={'orientation': 'horizontal'})
    for item in ax.get_yticklabels():
        item.set_rotation(0)
        item.set_size(5)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
        item.set_size(5)
    plt.savefig('average_expression_heatmap.pdf')
    plt.close('all')


phenograph_heatmaps(overall_zeroed)


def pheno_piecharts(overall_zeroed):

    banner('RUNNING MODULE: pheno_piecharts')

    pie_dir = os.path.join(os.getcwd(), 'pie_charts')
    if not os.path.exists(pie_dir):
        os.makedirs(pie_dir)

    # define factor generator
    def factors(n):
        flatten_iter = itertools.chain.from_iterable
        return set(flatten_iter((i, n//i)
                   for i in range(1, int(n**0.5)+1) if n % i == 0))

    color_dict = dict(zip(sorted(overall_zeroed['tissue'].unique()),
                          ['r', 'b', 'g', 'm', 'y']))

    for name, group in overall_zeroed.groupby(['cluster']):
        print('Plotting cluster ' + str(name) + ' piechart.')
        data = group['tissue'].value_counts()
        fig, ax = plt.subplots()
        labels = data.index.tolist()
        colors = [color_dict[x] for x in labels]
        patches, texts, autotexts = ax.pie(
            data, shadow=False, colors=colors, autopct='%1.1f%%',
            startangle=90)
        plt.axis('equal')
        plt.legend(labels, loc='upper right')

        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

        ax.set_title(str(name))
        plt.savefig(os.path.join(pie_dir, str(name) + '.pdf'))
        plt.close('all')
    print()

    # get best factorization of total number of Phenograph clusters
    TRUNC_n = len(overall_zeroed['cluster'].unique())

    TRUNC_factors = factors(TRUNC_n)
    TRUNC_factors = list(TRUNC_factors)

    TRUNC_tuple_list = []
    for i, v in enumerate(list(itertools.combinations(TRUNC_factors, 2))):
        if v[0] * v[1] == TRUNC_n:
            TRUNC_tuple_list.append(v)

    TRUNC_dif_list = []
    for pair in TRUNC_tuple_list:
        TRUNC_dif_list.append(abs(pair[0] - pair[1]))

    TRUNC_tuple_dict = dict(zip(TRUNC_tuple_list, TRUNC_dif_list))
    TRUNC_target_tuple = min(TRUNC_tuple_dict, key=TRUNC_tuple_dict.get)

    # plot tissue pie charts per Phenograph cluster where piechart radius is
    # proportional to population size
    the_grid = GridSpec(TRUNC_target_tuple[0], TRUNC_target_tuple[1])
    the_grid.update(hspace=50, wspace=0, left=0.35, right=0.65)
    coordinates = [(x, y) for x in range(
        TRUNC_target_tuple[0]) for y in range(TRUNC_target_tuple[1])]
    fig, ax = plt.subplots()

    percent_dict = {}

    for coordinate, (name, group) in itertools.zip_longest(
      coordinates, overall_zeroed.groupby(['cluster'])):
        data = group['tissue'].value_counts()
        total = group['tissue'].value_counts().sum()
        percent = (total/(len(overall_zeroed.index))*100)

        percent_dict[name] = round(percent, 2)

        print('Plotting population size proportioned flowSOM metacluster '
              + str(name) + ' (' + str(percent) + ')' + ' piechart.')
        radius = math.sqrt(percent)*10
        ax = plt.subplot(the_grid[coordinate], aspect=1)
        ax.set_title(str(name), loc='left', fontsize=4.0)
        labels = data.index.tolist()
        colors = [color_dict[x] for x in labels]
        patches, texts = ax.pie(data, shadow=False, radius=radius,
                                colors=colors, startangle=90)
        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

    plt.savefig(os.path.join(pie_dir, 'flowsom_pie_charts' + '.pdf'))
    plt.close('all')
    print()

    percentages = sorted(
        percent_dict.items(), key=operator.itemgetter(1), reverse=True)

    return percentages


percentages = pheno_piecharts(overall_zeroed)


def pheno_vio_cluster(sample, cluster_dict):

    banner('RUNNING MODULE: pheno_vio_cluster')

    violin_cluster_dir = os.path.join(
        os.getcwd(), 'violin_cluster_plots')
    os.makedirs(violin_cluster_dir)

    aggregate_choice_clusterdict = sample.copy()

    aggregate_choice_clusterdict['cluster'] = (
        aggregate_choice_clusterdict['cluster'].map(str) +
        ' ' + [cluster_dict[i] for i in
               aggregate_choice_clusterdict['cluster']])

    aggregate_choice_clusterdict = aggregate_choice_clusterdict.sort_values(
        ['cluster', 'status'], ascending=[True, False])

    # filter outliers
    for channel in channel_list:
        aggregate_choice_clusterdict_filt = aggregate_choice_clusterdict[
            (aggregate_choice_clusterdict[channel] > -3.5) &
            (aggregate_choice_clusterdict[channel] < 3.5)]

    dataframe_list = []
    for name, group in aggregate_choice_clusterdict_filt.groupby(['cluster']):
        print('Extracting protein expression data for the ' + str(name) +
              ' cell Phenograph cluster.')
        group.columns = [str(col) + '_' + name for col in group.columns]
        data_melt = pd.melt(group)
        for channel in channel_list:
            x = data_melt['variable'][data_melt['variable'] == channel +
                                      '_' + name].reset_index(drop=True)
            y = data_melt['value'][data_melt['variable'] == channel +
                                   '_' + name].reset_index(drop=True)
            hue = data_melt['value'][data_melt['variable'] == 'status' +
                                     '_' + name].reset_index(drop=True)
            data = pd.concat([x, y, hue], axis=1)
            data.columns = ['marker', 'value', 'status']
            dataframe_list.append(data)
            overall_dataframe_TRUNC = pd.concat(dataframe_list)
    print()

    print('Splitting marker column into channel and cell_type columns in '
          'overall_dataframe_TRUNC')
    marker_split = overall_dataframe_TRUNC['marker'] \
        .str.split('_', expand=True)
    l_list = [marker_split, overall_dataframe_TRUNC.iloc[:, 1:]]
    l = pd.concat(l_list,  axis=1)
    l.columns = ['channel', 'cell_type', 'value', 'status']
    print()

    sns.set(style='whitegrid')
    for cell in l['cell_type'].unique():
        print('Plotting protein expression data across all channels for the '
              + cell + ' cell Phenograph cluster.')
        data = l[l['cell_type'] == cell].copy()
        data['value'] = data['value'].astype('float')

        if len(l['status'][l['cell_type'] == cell].unique()) == 1:
            split = False
        elif len(l['status'][l['cell_type'] == cell].unique()) == 2:
            split = True

        g = sns.violinplot(
            x='channel', y='value', hue='status', data=data, split=split,
            inner='box', palette={'gl261': sns.xkcd_rgb['lighter green'],
                                  'naive': sns.xkcd_rgb['faded purple']},
            order=['cd45', 'b220', 'cd11b', 'cd11c', 'cd3e', 'cd4', 'cd49b',
                   'cd8a', 'f480', 'ly6c', 'ly6g'])
        sns.despine(left=True)

        g.set_xlabel('channel', size=15, weight='normal')
        g.set_ylabel('intensity', size=15, weight='normal')

        legend_text_properties = {'size': 10, 'weight': 'normal'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        xlabels = [item.get_text() for item in g.get_xticklabels()]

        xlabels_update = [xlabel.replace('cd8a', 'cd8' + u'\u03B1').replace(
            'cd3e', 'cd3' + u'\u03B5') for xlabel in xlabels]
        g.set_xticklabels(xlabels_update)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')
        plt.ylim(-3.0, 3.0)
        g.set_title('cluster_' + cell, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(violin_cluster_dir,
                    'cluster_' + str(cell.replace(' ', '_')) + '.pdf'))
        plt.close('all')
    print()

    l.to_csv('l.csv', index=False)

    return aggregate_choice_clusterdict_filt, l


aggregate_choice_clusterdict_filt, l = pheno_vio_cluster(sample, cluster_dict)


def pheno_box_cluster(l):

    banner('RUNNING MODULE: pheno_box_cluster')

    box_cluster_dir = os.path.join(os.getcwd(), 'box_cluster_plots')
    os.makedirs(box_cluster_dir)

    sns.set(style='whitegrid')
    for cell in l['cell_type'].unique():
        print('Plotting protein expression data across all channels for the '
              + cell + ' cell Phenograph cluster.')
        data = l[l['cell_type'] == cell].copy()
        data['value'] = data['value'].astype('float')

        g = sns.boxplot(
            x='channel', y='value', hue='status', data=data, linewidth=0.5,
            palette={'gl261': sns.xkcd_rgb['lighter green'],
                     'naive': sns.xkcd_rgb['faded purple']},
            order=['cd45', 'b220', 'cd11b', 'cd11c', 'cd3e', 'cd4', 'cd49b',
                   'cd8a', 'f480', 'ly6c', 'ly6g'])

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
        g.xaxis.grid(False)
        g.yaxis.grid(True)

        legend_text_properties = {'size': 10, 'weight': 'normal'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        xlabels = [item.get_text() for item in g.get_xticklabels()]

        xlabels_update = [xlabel.replace('cd8a', 'cd8' + u'\u03B1').replace(
            'cd3e', 'cd3' + u'\u03B5') for xlabel in xlabels]
        g.set_xticklabels(xlabels_update)

        sns.despine(left=True)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')

        g.set_xlabel('', size=15, weight='normal')
        g.set_ylabel('intensity', size=15, weight='normal')

        plt.ylim(-3.0, 3.0)
        g.set_title('cluster_' + cell, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(box_cluster_dir, 'cluster_' +
                    str(cell.replace(' ', '_')) + '.png'), dpi=500)
        plt.close('all')
    print()


pheno_box_cluster(l)


def pheno_box_channel(aggregate_choice_clusterdict_filt):

    banner('RUNNING MODULE: pheno_box_channel')

    box_channel_dir = os.path.join(os.getcwd(), 'box_channel_plots')
    os.makedirs(box_channel_dir)

    sns.set(style='whitegrid')
    for channel in list(channel_list) + ['fsc', 'ssc']:

        fig, ax = plt.subplots(figsize=(8, 5.5))

        print('Plotting ' + channel + ' protein expression across all'
              ' Phenograph clusters.')

        order = []

        for name, group in aggregate_choice_clusterdict_filt.groupby(
          ['cluster', 'status']):
            if name[1] == 'naive':
                order.append((name[0], group[channel].median()))

        if channel in ['fsc', 'ssc']:
            order.sort(key=lambda x: x[1])
        else:
            order.sort(key=lambda x: x[1], reverse=True)

        order = [i[0] for i in order]

        g = sns.boxplot(
            x='cluster', y=channel, hue='status', linewidth=0.5,
            data=aggregate_choice_clusterdict_filt,
            palette={'gl261': sns.xkcd_rgb['lighter green'],
                     'naive': sns.xkcd_rgb['faded purple']},
            order=order)

        sns.despine(left=True)

        xlabels = [item.get_text() for item in g.get_xticklabels()]

        xlabels_update = [xlabel.replace('neg', '$^-$').replace(
            'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1').replace(
                'CD3e', 'CD3' + u'\u03B5') for xlabel in xlabels]

        g.set_xticklabels(xlabels_update)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')

        legend_text_properties = {'size': 10, 'weight': 'normal'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
        g.xaxis.grid(False)
        g.yaxis.grid(True)

        g.set_xlabel('cluster', size=15, weight='normal')
        g.set_ylabel('intensity', size=15, weight='normal', labelpad=15)

        channel_update = channel.replace(
            'cd8a', 'CD8' + u'\u03B1').replace('cd3e', 'CD3' + u'\u03B5')

        g.set_title(str(channel_update), size=20, fontweight='bold', y=1.2)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')
        if channel in channel_list:
            plt.ylim(-3.0, 3.0)
        else:
            plt.ylim(0.0, 250000)
        plt.tight_layout()
        plt.savefig(os.path.join(box_channel_dir, 'cluster_vs_'
                    + str(channel) + '.pdf'))
        plt.close('all')
    print()


pheno_box_channel(aggregate_choice_clusterdict_filt)


def compare_clusters(overall, cluster1, cell_type1, cluster2, cell_type2):

    comparison_dir = os.path.join(os.getcwd(), 'cluster_comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    sns.set(style='whitegrid')

    data = overall[
        (overall['cluster'] == cluster1) &
        (overall['cell_type'] == cell_type1) |
        (overall['cluster'] == cluster2) &
        (overall['cell_type'] == cell_type1)]

    # take a random sample of the data
    data = data.head(100000)

    lt = pd.DataFrame(columns=['cluster', 'channel', 'value'])

    for cluster in data['cluster'].unique():
        for channel in channel_list:
            d = pd.DataFrame(data[channel][data['cluster'] == cluster])
            d.columns = ['value']
            d['cluster'] = cluster
            d['channel'] = channel
            lt = lt.append(d)

    lt['value'] = lt['value'].astype('float')

    g = sns.boxplot(
        x='channel', y='value', hue='cluster', data=lt, linewidth=0.5,
        palette={cluster1: sns.xkcd_rgb['lighter green'],
                 cluster2: sns.xkcd_rgb['faded purple']},
        order=['cd45', 'b220', 'cd11b', 'cd11c', 'cd3e', 'cd4', 'cd49b',
               'cd8a', 'f480', 'ly6c', 'ly6g'])

    g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
    g.xaxis.grid(False)
    g.yaxis.grid(True)

    legend_text_properties = {'size': 10, 'weight': 'normal'}
    legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

    for legobj, text, label in zip(
      legend.legendHandles,
      legend.get_texts(),
      [('cluster' + str(cluster1)), ('cluster' + str(cluster2))]):
        legobj.set_linewidth(0)
        text.set_text(label)

    xlabels = [item.get_text() for item in g.get_xticklabels()]

    xlabels_update = [xlabel.replace('cd8a', 'cd8' + u'\u03B1').replace(
        'cd3e', 'cd3' + u'\u03B5') for xlabel in xlabels]
    g.set_xticklabels(xlabels_update)

    sns.despine(left=True)

    for item in g.get_xticklabels():
        item.set_rotation(90)
        item.set_fontweight('normal')

    g.set_xlabel('', size=15, weight='normal')
    g.set_ylabel('intensity', size=15, weight='normal')

    plt.ylim(-3.0, 3.0)
    g.set_title(
        'cluster' + str(cluster1) + '_' + cell_type1 + ' vs. ' +
        'cluster' + str(cluster2) + '_' + cell_type2,
        fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            comparison_dir,
            'cluster' + str(cluster1) + '_' + cell_type1 + ' vs. ' +
            'cluster' + str(cluster2) + '_' + cell_type2 + '.pdf'))
    plt.close('all')

    return data


data = compare_clusters(overall, 17, 'CD8T', 30, 'CD8T')
