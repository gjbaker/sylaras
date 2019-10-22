from invoke_init import *
from constants_syl import *
from helpers import *


# select FULL and classified dataset versions for downstream analysis
def TOTAL_and_classified_choices():

    banner('RUNNING MODULE: FULL_and_classified')

    os.chdir(PICKLE_DIR)
    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    pi_classified_dir = open(PO_CLASSIFIED_DIR, 'rb')
    classified_dir = pickle.load(pi_classified_dir)

    os.chdir(aggregate_data_dir)
    aggregate_choice = pd.read_csv(TOTAL_SAMPLE)

    os.chdir(classified_dir)
    classified_choice = pd.read_csv('FULL_classified_data.csv')

    os.chdir(PICKLE_DIR)
    po_aggregate_choice = open(PO_AGGREGATE_CHOICE, 'wb')
    pickle.dump(aggregate_choice, po_aggregate_choice)
    po_aggregate_choice.close()

    po_classified_choice = open(PO_CLASSIFIED_CHOICE, 'wb')
    pickle.dump(classified_choice, po_classified_choice)
    po_classified_choice.close()

# split or combine select celltypes
def split_combine_celltypes():
    banner('RUNNING MODULE: split_combine_celltypes')

    os.chdir(PICKLE_DIR)
    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    classified_choice.loc[
        (classified_choice.cell_type == 'Mac') &
        (classified_choice.ssc > 100000), 'cell_type'] = 'Eo'

    classified_choice.loc[
        (classified_choice.cell_type == 'Ly6CposMac') &
        (classified_choice.ssc > 125000), 'cell_type'] = 'Ly6CposEo'

    classified_choice.loc[
        (classified_choice.cell_type == 'CD45negPMN'),
        'cell_type'] = 'PMN'
    classified_choice.loc[
        (classified_choice.cell_type == 'PMN'),
        'CD45'] = 1

    classified_choice.loc[
        (classified_choice.cell_type == 'CD45negPrecursor'),
        'cell_type'] = 'Precursor'
    classified_choice.loc[
        (classified_choice.cell_type == 'Precursor'),
        'CD45'] = 1

    classified_choice.loc[
        (classified_choice.cell_type == 'CD45negB'),
        'cell_type'] = 'B'
    classified_choice.loc[
        (classified_choice.cell_type == 'B'),
        'CD45'] = 1

    classified_choice.loc[
        (classified_choice.cell_type == 'CD45negDPT'),
        'cell_type'] = 'DPT'
    classified_choice.loc[
        (classified_choice.cell_type == 'DPT'),
        'CD45'] = 1

    classified_choice.loc[
        (classified_choice.cell_type == 'CD45negISPT'),
        'cell_type'] = 'ISPT'
    classified_choice.loc[
        (classified_choice.cell_type == 'ISPT'),
        'CD45'] = 1

    po_classified_choice = open(PO_CLASSIFIED_CHOICE, 'wb')
    pickle.dump(classified_choice, po_classified_choice)
    po_classified_choice.close()

    os.chdir(PICKLE_DIR)
    pi_classified_dir = open(PO_CLASSIFIED_DIR, 'rb')
    classified_dir = pickle.load(pi_classified_dir)

    os.chdir(classified_dir)
    for dirpath, dirnames, filenames in os.walk(
      classified_dir):
            for i in filenames:

                data = pd.read_csv(i)

                data.loc[
                    (data.cell_type == 'Mac') &
                    (data.ssc > 100000), 'cell_type'] = 'Eo'

                data.loc[
                    (data.cell_type == 'Ly6CposMac') &
                    (data.ssc > 125000), 'cell_type'] = 'Ly6CposEo'

                data.loc[
                    (data.cell_type == 'CD45negPMN'),
                    'cell_type'] = 'PMN'
                data.loc[
                    (data.cell_type == 'PMN'),
                    'CD45'] = 1

                data.loc[
                    (data.cell_type == 'CD45negPrecursor'),
                    'cell_type'] = 'Precursor'
                data.loc[
                    (data.cell_type == 'Precursor'),
                    'CD45'] = 1

                data.loc[
                    (data.cell_type == 'CD45negB'),
                    'cell_type'] = 'B'
                data.loc[
                    (data.cell_type == 'B'),
                    'CD45'] = 1

                data.loc[
                    (data.cell_type == 'CD45negDPT'),
                    'cell_type'] = 'DPT'
                data.loc[
                    (data.cell_type == 'DPT'),
                    'CD45'] = 1

                data.loc[
                    (data.cell_type == 'CD45negISPT'),
                    'cell_type'] = 'ISPT'
                data.loc[
                    (data.cell_type == 'ISPT'),
                    'CD45'] = 1

                s = data.groupby(channel_list_update)
                print('There are ' + str(s.ngroups) + ' BIPs in ' + i)


# assess vector/tissue coverage
def vector_coverage():

    banner('RUNNING MODULE: vector_coverage')

    os.chdir(PICKLE_DIR)
    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_classified_dir = open(PO_CLASSIFIED_DIR, 'rb')
    classified_dir = pickle.load(pi_classified_dir)

    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    os.chdir(aggregate_data_dir)
    print('Reading TOTAL_sample.')
    TOTAL_sample = pd.read_csv(TOTAL_SAMPLE)

    os.chdir(classified_dir)
    total_vectors1 = classified_choice
    total_vectors2 = total_vectors1.groupby(
        ['tissue', 'time_point', 'replicate', 'status', 'cell_type'] +
        channel_list_update).size().reset_index().rename(columns={0: 'count'})
    total_vectors3 = total_vectors2.sort_values(
        by=['tissue', 'time_point', 'replicate', 'status', 'count'],
        ascending=[True, True, True, False, False]).reset_index(drop=True)

    total_vectors4 = total_vectors1.groupby(
        ['tissue', 'time_point', 'replicate', 'status']).size().reset_index() \
        .rename(columns={0: 'count'})

    alpha = 0.01
    alpha_vectors_list = []
    condition_frame_alpha_dict = {}
    for s, group in enumerate(total_vectors3.groupby(['tissue', 'time_point',
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
    for tissue in sorted(condition_frame_alpha_index['tissue'].unique()):

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
    for s, group in enumerate(total_vectors3.groupby(['tissue', 'time_point',
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
                if i[1]['count']/denom >= alpha_slide:
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

    # plot percent of tissue specified vs. # of vectors
    plt.rcParams['font.weight'] = 'normal'
    color_dict = dict(zip(sorted(TOTAL_sample['tissue'].unique()),
                          ['r', 'b', 'g', 'm', 'y']))
    line_dict = {'gl261': 'dashed', 'naive': 'solid'}
    hue_dict = {7: 1.0, 14: 0.66, 30: 0.33}
    sns.set_style('whitegrid')
    fig = plt.figure()
    fig.suptitle('', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_title('# of vectors whose percentage is greater than x')
    ax.set_xlabel('percentage cutoff')
    ax.set_ylabel('# of vectors')
    k_list = []
    for key, value in slider_condition_vectors_dict.items():
        x, y = value
        color_label = key[0]
        hue_label = key[1]
        line_label = key[3]
        k_list.append(color_label)
        color = color_dict[color_label]
        hue = hue_dict[hue_label]
        line = line_dict[line_label]
        plt.plot(x, y, color=color, linestyle=line, alpha=hue)
    legend_list = []
    for key, value in color_dict.items():
        line = mlines.Line2D([], [], color=value, marker='', markersize=30,
                             label=key)
        legend_list.append(line)
    legend_text_properties = {'weight': 'bold'}
    legend = plt.legend(handles=legend_list, prop=legend_text_properties)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(5.0)
    ax.set_xscale('linear')
    ax.set_yscale('symlog')
    plt.axvline(x=(alpha * 100), ymin=0.0, ymax=5.0, linewidth=1,
                linestyle='dashed', color='k', alpha=0.7)
    ax.annotate('Alpha cutoff = ' + str(alpha * 100) + '%', xy=(0, 0),
                xytext=((30, 65)))
    plt.savefig(os.path.join(ORTHANT_DIR, 'tissue_vectors' + '.pdf'))
    plt.close('all')

    os.chdir(PICKLE_DIR)
    po_color_dict = open(PO_COLOR_DICT, 'wb')
    pickle.dump(color_dict, po_color_dict)
    po_color_dict.close()

    return(alpha_vectors)


alpha_vectors = vector_coverage()


# create a dictionary to accumulate celltype stats for aggregate plot
def dashboard_dict():

    banner('RUNNING MODULE: dashboard_dict')

    os.chdir(PICKLE_DIR)
    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    dashboard = {}
    for celltype in set(classified_choice['cell_type']):
        dashboard[celltype] = {}

    print('dashboard dictionary initialized')

    dashboard_shlf = shelve.open(DASHBOARD_SHLF)
    dashboard_shlf.update(dashboard)
    dashboard_shlf.close()

# combine classified_choice Boolean DataFrame with continuous expression values
def overall():

    banner('RUNNING MODULE: overall')

    os.chdir(PICKLE_DIR)
    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    pi_aggregate_choice = open(PO_AGGREGATE_CHOICE, 'rb')
    aggregate_choice = pickle.load(pi_aggregate_choice)

    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_channel_list = open(PO_CHANNEL_LIST, 'rb')
    channel_list = pickle.load(pi_channel_list)

    print('Combining continuous and Boolean classifier'
          ' results into overall DataFrame.')

    expression_values = aggregate_choice.loc[:, channel_list +
                                             ['fsc', 'ssc', 'index']]
    overall = pd.merge(classified_choice, expression_values, on='index')
    overall.drop(['fsc_x', 'ssc_x'], axis=1, inplace=True)
    overall = overall.rename(index=str,
                             columns={'fsc_y': 'fsc', 'ssc_y': 'ssc'})
    overall_cols = ['time_point', 'tissue', 'status', 'replicate', 'cluster',
                    'B220', 'CD11b', 'CD11c', 'CD3e', 'CD4', 'CD45', 'CD49b',
                    'CD8a', 'F480', 'Ly6C', 'Ly6G', 'cell_type',
                    'fsc', 'ssc', 'b220', 'cd11b', 'cd11c', 'cd3e', 'cd4',
                    'cd45', 'cd49b', 'cd8', 'f480', 'ly6c', 'ly6g',
                    'row', 'index']
    overall = overall[overall_cols]
    overall = overall.rename(index=str, columns={'cd8': 'cd8a'})
    overall.to_csv(
        os.path.join(aggregate_data_dir, OVERALL_CSV), index=False)
    print()

# run pair-wise t-tests on celltypes
def celltype_stats():

    banner('RUNNING MODULE: celltype_stats')

    os.chdir(PICKLE_DIR)
    pi_ave = open(PO_AVE, 'rb')
    ave = pickle.load(pi_ave)

    g1 = ave.unstack()
    g2 = g1.reset_index()
    g3 = g2.iloc[:, :4]
    g4 = g3[g3['status'] == 'naive']
    g5 = g4.replace(to_replace='naive', value='gl261')
    g6 = pd.concat([g4, g5], axis=0).reset_index(drop=True)
    g7 = pd.merge(g6, g2, how='left')
    g8 = g7.replace(to_replace='NaN', value=0.0)

    TOTAL_gl261 = g8.loc[g8['status'] == 'gl261'].reset_index(drop=True)
    TOTAL_gl261 = pd.melt(TOTAL_gl261, ['time_point', 'tissue', 'status',
                          'replicate'])
    TOTAL_gl261 = TOTAL_gl261.rename(columns={'value': 'gl261_percent'})

    TOTAL_naive = g8.loc[g8['status'] == 'naive'].reset_index(drop=True)
    TOTAL_naive = pd.melt(TOTAL_naive, ['time_point', 'tissue', 'status',
                          'replicate'])
    TOTAL_naive = TOTAL_naive.rename(columns={'value': 'naive_percent'})

    TOTAL_t_input = pd.concat([TOTAL_gl261, TOTAL_naive], axis=1)
    t_left = TOTAL_t_input.iloc[:, 0:5]
    t_right = TOTAL_t_input[['gl261_percent', 'naive_percent']]
    TOTAL_t_input = pd.concat([t_left, t_right], axis=1)
    TOTAL_t_input = TOTAL_t_input.drop(['status', 'replicate'], axis=1)
    TOTAL_t_input.fillna(value=0.0, inplace=True)

    TOTAL_t_stats_list = []
    for name, group in TOTAL_t_input.groupby(['time_point', 'tissue',
                                             'cell_type']):
        statistics = ttest_ind(group['gl261_percent'], group['naive_percent'],
                               axis=0, equal_var=True, nan_policy='propagate')
        print(name)
        print(statistics)
        TOTAL_t_stats_list.append(statistics)
        print()

    TOTAL_statistics = pd.DataFrame(TOTAL_t_stats_list)

    TOTAL_t_output = TOTAL_t_input.groupby(['time_point', 'tissue',
                                           'cell_type']).sum()
    TOTAL_t_output.reset_index(drop=False, inplace=True)
    TOTAL_t_output = TOTAL_t_output.drop(['gl261_percent', 'naive_percent'],
                                         axis=1)

    TOTAL_t_output = pd.concat([TOTAL_t_output, TOTAL_statistics], axis=1)

    TOTAL_t_means = TOTAL_t_input.groupby(['time_point', 'tissue',
                                          'cell_type']).mean()
    TOTAL_t_means = TOTAL_t_means.reset_index(drop=False, inplace=False)

    x_t = TOTAL_t_means
    y_t = TOTAL_t_output.iloc[:, 3:5]
    t_dfs = [x_t, y_t]
    TOTAL_sig_dif_all = pd.concat(t_dfs,  axis=1)
    TOTAL_sig_dif_all = TOTAL_sig_dif_all.sort_values(by=['pvalue'])
    TOTAL_sig_dif_all = TOTAL_sig_dif_all.replace(to_replace='NaN', value=0.0)
    TOTAL_sig_dif = TOTAL_sig_dif_all[TOTAL_sig_dif_all['pvalue'] <= 0.05]
    TOTAL_sig_dif = TOTAL_sig_dif[abs(TOTAL_sig_dif['statistic']) > 2.131]
    TOTAL_sig_dif.reset_index(drop=True, inplace=True)
    TOTAL_sig_dif = TOTAL_sig_dif.sort_values(['time_point', 'tissue',
                                              'cell_type', 'pvalue'])
    TOTAL_sig_dif.reset_index(drop=True, inplace=True)

    TOTAL_sig_dir = os.path.join(ORTHANT_DIR, 'statistics')
    os.mkdir(TOTAL_sig_dir)
    TOTAL_sig_dif.to_csv(os.path.join(TOTAL_sig_dir, TOTAL_SIG_DIF_CSV),
                         index=False)

    # perform FDR correction
    data = TOTAL_sig_dif_all.copy()
    data = data[data['cell_type'] != 'unspecified']
    stats = importr('stats')
    p_adjust = stats.p_adjust(
        FloatVector(data['pvalue'].fillna(value=0).tolist()),
        method='BH')
    data['corrected_pvalue'] = p_adjust

    sig_conds = data[data['corrected_pvalue'] <= 0.05]
    sig_dif_FDRcorrected = sig_conds.sort_values(
        by='corrected_pvalue', inplace=False, ascending=True)
    sig_dif_FDRcorrected.dropna(inplace=True)

    sig_dif_FDRcorrected['dif'] = abs(
        sig_dif_FDRcorrected['gl261_percent'] -
        sig_dif_FDRcorrected['naive_percent'])

    sig_dif_FDRcorrected['ratio'] = np.log2(
        ((0.01 + sig_dif_FDRcorrected['gl261_percent']) /
         (0.01 + sig_dif_FDRcorrected['naive_percent'])))

    sig_dif_FDRcorrected.sort_values(by='ratio', inplace=True, ascending=False)

    sig_dif_FDRcorrected.to_csv(
        os.path.join(TOTAL_sig_dir, SIG_DIF_FDRCORRECTED_CSV),
        index=False)

    os.chdir(PICKLE_DIR)
    po_TOTAL_sig_dir = open(PO_TOTAL_SIG_DIR, 'wb')
    pickle.dump(TOTAL_sig_dir, po_TOTAL_sig_dir)
    po_TOTAL_sig_dir.close()

    po_TOTAL_sig_dif_all = open(PO_TOTAL_SIG_DIF_ALL, 'wb')
    pickle.dump(TOTAL_sig_dif_all, po_TOTAL_sig_dif_all)
    po_TOTAL_sig_dif_all.close()

    po_TOTAL_sig_dif = open(PO_TOTAL_SIG_DIF, 'wb')
    pickle.dump(TOTAL_sig_dif, po_TOTAL_sig_dif)
    po_TOTAL_sig_dif.close()

    po_sig_dif_FDRcorrected = open(PO_SIG_DIF_FDRCORRECTED, 'wb')
    pickle.dump(sig_dif_FDRcorrected, po_sig_dif_FDRcorrected)
    po_sig_dif_FDRcorrected.close()


# run pair-wise t-tests on vectors
def vector_stats():
    banner('RUNNING MODULE: vector_stats')
    os.chdir(PICKLE_DIR)
    pi_ave_vec = open(PO_AVE_VEC, 'rb')
    ave_vec = pickle.load(pi_ave_vec)

    pi_classified_choice_copy = open(PO_CLASSIFIED_CHOICE_COPY, 'rb')
    classified_choice_copy = pickle.load(pi_classified_choice_copy)

    pi_TOTAL_sig_dir = open(PO_TOTAL_SIG_DIR, 'rb')
    TOTAL_sig_dir = pickle.load(pi_TOTAL_sig_dir)

    g1_vec = ave_vec.unstack()
    g2_vec = g1_vec.reset_index()
    g3_vec = g2_vec.iloc[:, :4]
    g4_vec = g3_vec[g3_vec['status'] == 'naive']
    g5_vec = g4_vec.replace(to_replace='naive', value='gl261')
    g6_vec = pd.concat([g4_vec, g5_vec], axis=0).reset_index(drop=True)
    g7_vec = pd.merge(g6_vec, g2_vec, how='left')
    g8_vec = g7_vec.replace(to_replace='NaN', value=0.0)

    TOTAL_gl261_vec = g8_vec.loc[g8_vec['status'] == 'gl261'] \
        .reset_index(drop=True)
    TOTAL_gl261_vec = pd.melt(
        TOTAL_gl261_vec, ['time_point', 'tissue', 'status',
                          'replicate'])
    TOTAL_gl261_vec = TOTAL_gl261_vec.rename(
        columns={'value': 'gl261_percent'})

    TOTAL_naive_vec = g8_vec.loc[g8_vec['status'] == 'naive'] \
        .reset_index(drop=True)
    TOTAL_naive_vec = pd.melt(
        TOTAL_naive_vec, ['time_point', 'tissue', 'status',
                          'replicate'])
    TOTAL_naive_vec = TOTAL_naive_vec.rename(
        columns={'value': 'naive_percent'})

    TOTAL_t_input_vec = pd.concat([TOTAL_gl261_vec, TOTAL_naive_vec], axis=1)
    t_left_vec = TOTAL_t_input_vec.iloc[:, 0:5]
    t_right_vec = TOTAL_t_input_vec[['gl261_percent', 'naive_percent']]
    TOTAL_t_input_vec = pd.concat([t_left_vec, t_right_vec], axis=1)
    TOTAL_t_input_vec = TOTAL_t_input_vec.drop(['status', 'replicate'], axis=1)
    TOTAL_t_input_vec.fillna(value=0.0, inplace=True)

    TOTAL_t_stats_list_vec = []
    for name, group in TOTAL_t_input_vec.groupby(['time_point', 'tissue',
                                                 'vector']):
        statistics = ttest_ind(group['gl261_percent'], group['naive_percent'],
                               axis=0, equal_var=True, nan_policy='propagate')
        print(name)
        print(statistics)
        TOTAL_t_stats_list_vec.append(statistics)
        print()

    TOTAL_statistics_vec = pd.DataFrame(TOTAL_t_stats_list_vec)

    TOTAL_t_output_vec = TOTAL_t_input_vec.groupby(['time_point', 'tissue',
                                                   'vector']).sum()
    TOTAL_t_output_vec.reset_index(drop=False, inplace=True)
    TOTAL_t_output_vec = TOTAL_t_output_vec.drop(['gl261_percent',
                                                 'naive_percent'], axis=1)

    TOTAL_t_output_vec = pd.concat(
        [TOTAL_t_output_vec, TOTAL_statistics_vec], axis=1)

    TOTAL_t_means_vec = TOTAL_t_input_vec.groupby(['time_point', 'tissue',
                                                  'vector']).mean()
    TOTAL_t_means_vec = TOTAL_t_means_vec.reset_index(
        drop=False, inplace=False)

    x_t_vec = TOTAL_t_means_vec
    y_t_vec = TOTAL_t_output_vec.iloc[:, 3:5]
    t_dfs_vec = [x_t_vec, y_t_vec]
    TOTAL_sig_dif_all_vec = pd.concat(t_dfs_vec,  axis=1)
    TOTAL_sig_dif_all_vec = TOTAL_sig_dif_all_vec.sort_values(by=['pvalue'])
    TOTAL_sig_dif_all_vec = TOTAL_sig_dif_all_vec.replace(
        to_replace='NaN', value=0.0)
    TOTAL_sig_dif_vec = TOTAL_sig_dif_all_vec[
        TOTAL_sig_dif_all_vec['pvalue'] <= 0.05]
    TOTAL_sig_dif_vec = TOTAL_sig_dif_vec[
        abs(TOTAL_sig_dif_vec['statistic']) > 2.131]
    TOTAL_sig_dif_vec.reset_index(drop=True, inplace=True)
    TOTAL_sig_dif_vec = TOTAL_sig_dif_vec.sort_values(['time_point', 'tissue',
                                                      'vector', 'pvalue'])
    TOTAL_sig_dif_vec.reset_index(drop=True, inplace=True)

    vector_dict = dict(zip(classified_choice_copy.vector,
                       classified_choice_copy.cell_type))
    TOTAL_sig_dif_vec['cell_type'] = TOTAL_sig_dif_vec['vector'].map(
        vector_dict)
    TOTAL_sig_dif_vec.to_csv(
        os.path.join(TOTAL_sig_dir, TOTAL_SIG_DIF_VEC_CSV), index=False)

    TOTAL_sig_dif_vec['dif'] = abs(TOTAL_sig_dif_vec['gl261_percent'] -
                                   TOTAL_sig_dif_vec['naive_percent'])

    sig7 = TOTAL_sig_dif_vec[(TOTAL_sig_dif_vec['time_point'] == 7) &
                             (TOTAL_sig_dif_vec['cell_type'] == 'unspecified')]
    sig7 = sig7.sort_values(by='dif', ascending=False, inplace=False)
    sig14 = TOTAL_sig_dif_vec[
        (TOTAL_sig_dif_vec['time_point'] == 14) &
        (TOTAL_sig_dif_vec['cell_type'] == 'unspecified')]
    sig14 = sig14.sort_values(by='dif', ascending=False, inplace=False)
    sig30 = TOTAL_sig_dif_vec[
        (TOTAL_sig_dif_vec['time_point'] == 30) &
        (TOTAL_sig_dif_vec['cell_type'] == 'unspecified')]
    sig_30 = sig30.sort_values(by='dif', ascending=False, inplace=False)




def replicate_counts():

    banner('RUNNING MODULE: replicate_counts')

    os.chdir(PICKLE_DIR)
    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    pi_color_dict = open(PO_COLOR_DICT, 'rb')
    color_dict = pickle.load(pi_color_dict)

    os.chdir(aggregate_data_dir)
    overall = pd.read_csv(OVERALL_CSV)

    replicate_plot_dir = os.path.join(ORTHANT_DIR, 'replicate_counts')
    os.makedirs(replicate_plot_dir)

    for celltype in sorted(overall['cell_type'].unique()):
        print(celltype)
        x_overall = []
        y_blood = []
        y_marrow = []
        y_nodes = []
        y_spleen = []
        y_thymus = []
        for status in sorted(overall['status'].unique()):
            for timepoint in sorted(overall['time_point'].unique()):
                for tissue in sorted(overall['tissue'].unique()):
                    for replicate in sorted(overall['replicate'].unique()):

                        cell_num = overall[
                            (overall['cell_type'] == celltype) &
                            (overall['replicate'] == replicate) &
                            (overall['tissue'] == tissue) &
                            (overall['status'] == status) &
                            (overall['time_point'] == timepoint)]

                        total_cells = overall[
                            (overall['replicate'] == replicate) &
                            (overall['tissue'] == tissue) &
                            (overall['status'] == status) &
                            (overall['time_point'] == timepoint)]

                        percent_comp = (len(cell_num)/len(total_cells)) * 100
                        percent_comp = float('%.2f' % percent_comp)

                        # print(tuple([status, timepoint, tissue, replicate]),
                        #       '= ' + str(percent_comp) + '%')

                        condition = tuple([status, timepoint,
                                          tissue, replicate])

                        if condition[2] == 'blood':
                            x_overall.append(
                                tuple([status, timepoint, replicate]))
                            y_blood.append(percent_comp)

                        elif condition[2] == 'marrow':
                            x_overall.append(
                                tuple([status, timepoint, replicate]))
                            y_marrow.append(percent_comp)

                        elif condition[2] == 'nodes':
                            x_overall.append(
                                tuple([status, timepoint, replicate]))
                            y_nodes.append(percent_comp)

                        elif condition[2] == 'spleen':
                            x_overall.append(
                                tuple([status, timepoint, replicate]))
                            y_spleen.append(percent_comp)

                        elif condition[2] == 'thymus':
                            x_overall.append(
                                tuple([status, timepoint, replicate]))
                            y_thymus.append(percent_comp)

        x_overall_set = sorted(list(set(x_overall)))
        y_blood_dict = dict(zip(x_overall_set, y_blood))
        y_marrow_dict = dict(zip(x_overall_set, y_marrow))
        y_nodes_dict = dict(zip(x_overall_set, y_nodes))
        y_spleen_dict = dict(zip(x_overall_set, y_spleen))
        y_thymus_dict = dict(zip(x_overall_set, y_thymus))

        y_blood_list = []
        for key, value in y_blood_dict.items():
            y_blood_list.append(key + (value,))
        y_blood_list.sort(key=itemgetter(0), reverse=True)
        y_blood_list.sort(key=itemgetter(1), reverse=False)
        y_blood_final = [x[3] for x in y_blood_list]
        dashboard_shlf[celltype]['blood_rep_data'] = y_blood_final

        y_marrow_list = []
        for key, value in y_marrow_dict.items():
            y_marrow_list.append(key + (value,))
        y_marrow_list.sort(key=itemgetter(0), reverse=True)
        y_marrow_list.sort(key=itemgetter(1), reverse=False)
        y_marrow_final = [x[3] for x in y_marrow_list]
        dashboard_shlf[celltype]['marrow_rep_data'] = y_marrow_final

        y_nodes_list = []
        for key, value in y_nodes_dict.items():
            y_nodes_list.append(key + (value,))
        y_nodes_list.sort(key=itemgetter(0), reverse=True)
        y_nodes_list.sort(key=itemgetter(1), reverse=False)
        y_nodes_final = [x[3] for x in y_nodes_list]
        dashboard_shlf[celltype]['nodes_rep_data'] = y_nodes_final

        y_spleen_list = []
        for key, value in y_spleen_dict.items():
            y_spleen_list.append(key + (value,))
        y_spleen_list.sort(key=itemgetter(0), reverse=True)
        y_spleen_list.sort(key=itemgetter(1), reverse=False)
        y_spleen_final = [x[3] for x in y_spleen_list]
        dashboard_shlf[celltype]['spleen_rep_data'] = y_spleen_final

        y_thymus_list = []
        for key, value in y_thymus_dict.items():
            y_thymus_list.append(key + (value,))
        y_thymus_list.sort(key=itemgetter(0), reverse=True)
        y_thymus_list.sort(key=itemgetter(1), reverse=False)
        y_thymus_final = [x[3] for x in y_thymus_list]
        dashboard_shlf[celltype]['thymus_rep_data'] = y_thymus_final

        x_blood_list_sep = [x[:3] for x in y_blood_list]
        x_final = ['%s, %s, %s' % u for u in x_blood_list_sep]
        dashboard_shlf[celltype]['x_final_rep_data'] = x_final

        y_overall = []
        y_overall.extend(y_blood_final)
        y_overall.extend(y_marrow_final)
        y_overall.extend(y_nodes_final)
        y_overall.extend(y_spleen_final)
        y_overall.extend(y_thymus_final)
        dashboard_shlf[celltype]['y_overall_rep_data'] = y_overall

        sns.set(style='whitegrid')
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(7, 6),
                                                      sharex=True)

        fig.suptitle(celltype, fontsize=10, fontweight='bold', y=0.99)

        hue_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5, 5, 5]

        # to get hex code for a seaborn color palette
        # pal = sns.color_palette('RdGy', 10)
        # pal.as_hex()

        colors = {0: 'b', 1: 'mediumaquamarine', 2: 'b',
                  3: 'mediumaquamarine', 4: 'b', 5: 'mediumaquamarine'}

        sns.barplot(x_final, y_blood_final, hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax1)
        ax1.legend_.remove()

        sns.barplot(x_final, y_marrow_final, hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax2)
        ax2.legend_.remove()

        sns.barplot(x_final, y_nodes_final, hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax3)
        ax3.legend_.remove()

        sns.barplot(x_final, y_spleen_final, hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax4)
        ax4.legend_.remove()

        sns.barplot(x_final, y_thymus_final, hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax5)
        ax5.legend_.remove()

        for ax, tissue in zip([ax1, ax2, ax3, ax4, ax5],
                              sorted(overall['tissue'].unique())):
            ax.set_ylabel('% composition').set_size(7)
            ax.set_ylim(0, max(y_overall))
            ax.tick_params(axis='y', which='both', length=0)
            ax.zorder = 1
            for item in ax.get_yticklabels():
                item.set_rotation(0)
                item.set_size(7)
            for item in ax.get_xticklabels():
                item.set_rotation(90)
                item.set_size(7)
            ax1 = ax.twinx()
            ax1.set_ylim(0, max(y_overall))
            ax1.set_yticklabels([])
            ax1.set_ylabel(tissue, color=color_dict[tissue],
                           fontweight='bold')
            ax1.tick_params(axis='y', which='both', length=0)

            for n, bar in enumerate(ax.patches):
                width = bar.get_width()
                bar.set_width(width*5)
                if 48 < n < 96:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.13)
                elif 96 < n < 144:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.25)
                elif 144 < n < 192:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.38)
                elif 192 < n < 240:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.51)
                elif 240 < n < 288:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.75)
        plt.xlim(-1.1, len(x_final))
        plt.tight_layout()
        plt.savefig(os.path.join(replicate_plot_dir, celltype + '.pdf'))
        plt.close('all')
    print()

    dashboard_shlf.close()
