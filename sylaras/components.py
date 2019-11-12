import logging
import functools
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Pipeline module order, to be filled in by the @module decorator.
pipeline_modules = []


def module(func):
    """
    Annotation for pipeline module functions.

    This function adds the given function to the registry list. It also wraps
    the given function to log a pre/post-call banner.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("in wrapper")
        logger.info("=" * 70)
        logger.info("RUNNING MODULE: %s", func.__name__)
        result = func(*args, **kwargs)
        logger.info("=" * 70)
        logger.info("")
        return result
    pipeline_modules.append(wrapper)
    return wrapper


def save_data(data, path, **kwargs):
    """
    Save a DataFrame as csv, creating all intermediate directories.

    Extra kwargs will be passed through to pandas.DataFrame.to_csv.

    """

    if not path.name.endswith(".csv"):
        raise ValueError("Path must end with .csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, **kwargs)


@module
def weighted_random_sample(data, config):
    """Generate a random sample of data, weighted by tissue"""

    # Calculate random sample weighting to normalize cell counts by tissue.
    groups = data.groupby('tissue')
    tissue_weights = pd.DataFrame({
        'weights': 1 / (groups.size() * len(groups))
    })
    weights = pd.merge(
        data[['tissue']], tissue_weights, left_on='tissue', right_index=True
    )

    logger.info('Subsampling data.')
    sample = data.sample(
        n=config.random_sample_size, replace=False, weights=weights['weights'],
        random_state=config.random_seed, axis=0
    )
    sample.reset_index(drop=True, inplace=True)

    save_data(
        sample,
        config.output_path / 'weighted_random_sample' / 'wrs.csv'
    )

    return sample


#@module
def gate_bias(sample, config):
    """Generate kernel/jittered subsets of data."""

    # get the kernel DataFrame
    kernel_frame = sample.copy()
    for channel in config.id_channels:
        data = kernel_frame[channel]
        pct_1 = np.percentile(data, config.kernel_low)
        pct_99 = np.percentile(data, config.kernel_high)
        reject = (data < pct_1) | (data > pct_99)
        kernel_frame.loc[reject, channel] = np.nan

    # compute and store the jitter bias per channel in a dictionary
    jitter_values = {}
    for channel in config.id_channels:
        data = sample[channel]
        value = config.jitter * data.max()
        jitter_values[data.name] = value

    # get jittered DataFrames by adding or subtracting the jitter biases
    # to or from their respective channel values.
    jit_pos_list = []
    jit_neg_list = []
    for channel in config.id_channels:
        plus_data = sample[channel] + jitter_values[channel]
        minus_data = sample[channel] - jitter_values[channel]
        jit_pos_list.append(plus_data)
        jit_neg_list.append(minus_data)
    sample_jit_pos = pd.concat(jit_pos_list, axis=1)
    sample_jit_neg = pd.concat(jit_neg_list, axis=1)

    # add metadata columns from sample to jit_pos and jit_neg frames
    cols = sample.columns.tolist()
    meta_cols = list(set(cols) - set(config.id_channels))
    for meta_col in meta_cols:
        sample_jit_pos[meta_col] = sample[meta_col]
        sample_jit_neg[meta_col] = sample[meta_col]

    # reorder columns according to sample
    sample_jit_pos = sample_jit_pos[cols]
    sample_jit_neg = sample_jit_neg[cols]

    # get kernels of jittered DataFrames
    jit_pos_kernel_frame = sample_jit_pos.copy()
    jit_neg_kernel_frame = sample_jit_neg.copy()
    for frame in [jit_pos_kernel_frame, jit_neg_kernel_frame]:
        for channel in config.id_channels:
            data = frame[channel]
            pct_1 = np.percentile(data, config.kernel_low)
            pct_99 = np.percentile(data, config.kernel_high)
            reject = (data < pct_1) | (data > pct_99)
            frame.loc[reject, channel] = np.nan

    # create a dictionary containing all kernel columns
    kernel_dict = {}
    for channel in config.id_channels:

        for (name, data) in zip(

            ['original_' + channel,
             'back_' + channel,
             'forward_' + channel],

            [kernel_frame[channel],
             jit_pos_kernel_frame[channel],
             jit_neg_kernel_frame[channel]]):

            kernel_dict[name] = data

    # plot kernel distributions for each channel

    sns.set(style='white')
    for channel in config.id_channels:
        for key, value in kernel_dict.items():
            if key.endswith(channel):
                print('Plotting ' + key)
                value = value.dropna()
                ax = sns.distplot(value, kde=True, hist=False,
                                  kde_kws={'lw': 2.0}, label=key)
        ax.set_ylabel('density')
        ax.set_xlabel('signal intensity')
        ax.get_yaxis().set_visible(True)
        ax.legend()
        # plt.savefig()
        plt.close('all')
        print()

    # remove cells from the sample and kernel DataFrames whose signal
    # intensity values are between -jitter_value and +jitter_value
    sample_copy = sample.copy()
    kernel_frame_copy = kernel_frame.copy()

    filter_dict = {}
    for (name, data) in zip(

      ['FULL', 'FULL_KERNEL'],
      [sample_copy, kernel_frame_copy]):

        data_list = []

        for key, value in jitter_values.items():
            data = pd.DataFrame([np.nan if -value <= x <= value
                                else x for x in data[key].values])

            data.columns = [key]
            data_list.append(data)
        filter_frame = pd.concat(data_list, axis=1)
        filter_dict[name] = filter_frame

    # then drop rows containing at least 1 NaN in filter_dict
    jitter_dict = {}
    for (name, (key, value)) in zip(
      ['FULL_JITTERED', 'KERNEL_JITTERED'], filter_dict.items()):

        value.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        value = pd.merge(value, sample, how='left', left_index=True,
                         right_index=True)
        y_cols = [c for c in value.columns if '_y' not in c]
        value = value[y_cols]

        x_cols = [c for c in value.columns if '_x' in c]

        col_dict = dict(zip(x_cols, config.id_channels))

        value = value.rename(columns=col_dict, inplace=False)
        value = value[sample.columns]

        jitter_dict[key] = value
        # value.to_csv(
        #     os.path.join(name + '_data.csv'), index=False)

    # drop rows containing at least 1 NaN kernel_frame
    kernel_frame.dropna(axis=0, how='any', thresh=None, subset=None,
                        inplace=True)

    kernel_frame.to_csv(
        os.path.join(
            config['aggregate_data_dir'], 'FULL_KERNEL_data.csv'), index=False)

    # create a dictionary containing sample, kernel, jittered, and
    # kernel_jittered datasets
    final_frames_dict = {}
    for (name, data) in zip(
        ['FULL', 'KERNEL', 'FULL_JITTERED', 'KERNEL_JITTERED'],

        [sample, kernel_frame, jitter_dict['FULL'],
         jitter_dict['FULL_KERNEL']]):

        final_frames_dict[name] = data

    return final_frames_dict


#@module
def data_discretization(df_dict, config):
    """Binarize kernel/jittered subsets of data."""

    Boo_frames_dict = {}
    for name, frame in df_dict.items():
        channel_columns = {}
        for channel in config.id_channels:
            channel_columns[channel] = frame.loc[:, channel].values
        print()
        for key, value in channel_columns.items():
            print('Converting ' + key + ' protein expression data into its '
                  'Boolean representation in the ' + name + ' DataFrame.')
            for i, v in enumerate(value):
                if v > 0:
                    value[i] = 1
                elif v <= 0:
                    value[i] = 0
        Boo_data = frame.iloc[:, config.id_channels].astype(int)
        config.id_channels.sort()
        Boo_data = Boo_data[config.id_channels]
        channel_list_update_dict = {
            'b220': 'B220', 'cd45': 'CD45', 'cd11b': 'CD11b', 'cd11c': 'CD11c',
            'cd3e': 'CD3e', 'cd4': 'CD4', 'cd49b': 'CD49b', 'cd8': 'CD8a',
            'f480': 'F480', 'ly6c': 'Ly6C', 'ly6g': 'Ly6G'}
        Boo_data1 = Boo_data.rename(columns=channel_list_update_dict)
        Boo_data2 = pd.concat([frame.iloc[:, 0:7], Boo_data1,
                              frame.iloc[:, 18:20]], axis=1)

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

        Boo_frames_dict[name] = Boo_data2

        channel_list_update = list(channel_list_update_dict.values())
        channel_list_update.sort()
        unique_vectors = Boo_data2.drop_duplicates(channel_list_update)

        g = sns.heatmap(unique_vectors.loc[:, channel_list_update])
        for item in g.get_yticklabels():
            item.set_rotation(90)
        # plt.savefig(name + '_unique_vectors' + '.pdf')
        plt.close('all')
    print()

    return Boo_frames_dict, channel_list_update
