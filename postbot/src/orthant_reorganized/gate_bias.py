from invoke_init import *
from constants_syl import *
from helpers import *


# -----------------------------------------------------------------------------
# GATE BIAS

# identify vectors sensitive to gate placement
def gate_bias():
    banner('RUNNING MODULE: gate_bias')

    os.chdir(PICKLE_DIR)
    pi_channel_list = open('channel_list.pickle', 'rb')
    channel_list = pickle.load(pi_channel_list)

    pi_aggregate_data_dir = open('aggregate_data_dir.pickle', 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    os.chdir(aggregate_data_dir)
    print('Reading TOTAL_sample.')
    TOTAL_sample = pd.read_csv('TOTAL_sample.csv')

    # generate a DataFrame of kernel data from the FULL_final dataset (i.e.
    # FULL_kernel_frame)
    FULL_kernel_frame = TOTAL_sample.copy()
    for i in channel_list:
        data = FULL_kernel_frame[i]
        pct_1 = np.percentile(data, 1)
        pct_99 = np.percentile(data, 99)
        reject = (data < pct_1) | (data > pct_99)
        FULL_kernel_frame.loc[reject, i] = np.nan

    # compute the jitter bias per channel (5% of full positive range)
    FULL_plus_range = {}
    for i in channel_list:
        col_data = TOTAL_sample[i]
        pos_data = 0.05 * col_data.max()
        FULL_plus_range[col_data.name] = pos_data

    # add or subtract the computed jitter bias per channel to or from
    # the values in FULL_kernel_frame
    jp_list = []
    jm_list = []
    for col in TOTAL_sample[channel_list]:
        plus_data = TOTAL_sample[col] + FULL_plus_range[col]
        minus_data = TOTAL_sample[col] - FULL_plus_range[col]
        jp_list.append(plus_data)
        jm_list.append(minus_data)
    TOTAL_sample_jp = pd.concat(jp_list, axis=1)
    TOTAL_sample_jm = pd.concat(jm_list, axis=1)

    # get metadata column headers from TOTAL_sample data and
    # add to jitter frames (i.e. TOTAL_sample_jp/jm)
    TOTAL_sample_cols = TOTAL_sample.columns.tolist()
    meta_cols = list(set(TOTAL_sample_cols) - set(channel_list))
    for i in meta_cols:
        TOTAL_sample_jp[i] = TOTAL_sample[i]
        TOTAL_sample_jm[i] = TOTAL_sample[i]

    # reorder columns according to TOTAL_sample
    TOTAL_sample_jp = TOTAL_sample_jp[TOTAL_sample_cols]
    TOTAL_sample_jm = TOTAL_sample_jm[TOTAL_sample_cols]

    # status filter
    TOTAL_sample_jp_status_filtered = TOTAL_sample_jp[
        TOTAL_sample_jp['status'] != 'baseline']
    TOTAL_sample_jm_status_filtered = TOTAL_sample_jm[
        TOTAL_sample_jm['status'] != 'baseline']

    # generate kernel DataFrames of TOTAL_sample_jp and and TOTAL_sample_jm
    jp_kernel_frame = TOTAL_sample_jp.copy()
    jm_kernel_frame = TOTAL_sample_jm.copy()
    for frame in [jp_kernel_frame, jm_kernel_frame]:
        for i in channel_list:
            data = frame[i]
            pct_1 = np.percentile(data, 1)
            pct_99 = np.percentile(data, 99)
            reject = (data < pct_1) | (data > pct_99)
            frame.loc[reject, i] = np.nan

    # create a dictionary containing all kernel columns (i.e.
    # total_kernel_dict)
    total_kernel_dict = {}
    for col in channel_list:
        for (i, j) in zip(['initial_' + col, 'back_' + col, 'forward_' + col],
                          [FULL_kernel_frame[col], jp_kernel_frame[col],
                          jm_kernel_frame[col]]):
            total_kernel_dict[i] = j

    # plot and save univariate kernel distributions for each channel
    FULL_jitter_plot_dir = os.path.join(ORTHANT_DIR, 'jitter_plots')
    os.makedirs(FULL_jitter_plot_dir)

    sns.set(style='white')
    for i in channel_list:
        for key, value in total_kernel_dict.items():
                if key.endswith(i):
                    print('Plotting ' + key)
                    value = value.dropna()
                    ax = sns.distplot(value, kde=True, hist=False,
                                      kde_kws={'lw': 2.0}, label=key)
        ax.set_ylabel('density')
        ax.set_xlabel('signal intensity')
        ax.get_yaxis().set_visible(True)
        # plt.axvline(y=1.0, xmin=0, xmax=5.0, linewidth=1, linestyle='dashed',
        # color = 'k', alpha=0.7)
        ax.legend()
        plt.savefig(os.path.join(FULL_jitter_plot_dir, str(i) + '.pdf'))
        plt.close('all')
        print()

    # filter TOTAL_sample and FULL_kernel_frame using the bias dictionary
    TOTAL_sample_copy = TOTAL_sample.copy()
    FULL_kernel_frame_copy = FULL_kernel_frame.copy()
    filter_dict = {}
    for (i, j) in zip(['FULL', 'FULL_KERNEL'],
                      [TOTAL_sample_copy, FULL_kernel_frame_copy]):
        a_list = []
        for key, value in FULL_plus_range.items():
            data = pd.DataFrame([np.nan if -FULL_plus_range[key] <= x <=
                                FULL_plus_range[key]
                                else x for x in j[key].values])

            data.columns = [key]
            a_list.append(data)
        filter_frame = pd.concat(a_list, axis=1)
        filter_dict[i] = filter_frame

    # drop rows containing at least 1 NaN in filter_dict
    jitter_dict = {}
    for (i, (key, value)) in zip(['FULL_JITTERED', 'KERNEL_JITTERED'],
                                 filter_dict.items()):

        value.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        value = pd.merge(value, TOTAL_sample, how='left', left_index=True,
                         right_index=True)
        y_cols = [c for c in value.columns if '_y' not in c]
        value = value[y_cols]
        x_cols = [c for c in value.columns if '_x' in c]
        col_dict = dict(zip(x_cols, channel_list))
        value = value.rename(columns=col_dict, inplace=False)
        value = value[TOTAL_sample.columns]
        jitter_dict[key] = value
        value.to_csv(os.path.join(aggregate_data_dir, i + '_data.csv'),
                     index=False)

    # drop rows containing at least 1 NaN FULL_kernel_frame
    FULL_kernel_frame.dropna(axis=0, how='any', thresh=None, subset=None,
                             inplace=True)
    FULL_kernel_frame.to_csv(os.path.join(aggregate_data_dir,
                             'FULL_KERNEL_data.csv'), index=False)

    # create a dictionary containing FULL, kernel, and kernel_jitter datasets
    # (total_kernel_dict)
    final_frames_dict = {}
    for (i, j) in zip(['FULL', 'KERNEL', 'FULL_JITTERED', 'KERNEL_JITTERED'],
                      [TOTAL_sample, FULL_kernel_frame, jitter_dict['FULL'],
                      jitter_dict['FULL_KERNEL']]):
        final_frames_dict[i] = j

    os.chdir(PICKLE_DIR)
    import shelve

    final_frames_dict_shlf = shelve.open('final_frames_dict.shelve')
    final_frames_dict_shlf.update(final_frames_dict)
    final_frames_dict_shlf.close()