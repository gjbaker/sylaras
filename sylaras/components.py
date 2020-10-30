import logging
import functools

import os
import shelve
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import matplotlib

import math
from natsort import natsorted
from decimal import Decimal

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from scipy.stats import ttest_ind

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

from inspect import getmembers, isclass

from .config import FilterChoice

logger = logging.getLogger(__name__)

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

# Pipeline module order, to be filled in by the @module decorator.
pipeline_modules = []
pipeline_module_names = []


def module(func):
    """
    Annotation for pipeline module functions.

    This function adds the given function to the registry list. It also wraps
    the given function to log a pre/post-call banner.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("=" * 70)
        logger.info("RUNNING MODULE: %s", func.__name__)
        result = func(*args, **kwargs)
        logger.info("=" * 70)
        logger.info("")
        return result
    pipeline_modules.append(wrapper)
    pipeline_module_names.append(wrapper.__name__)
    return wrapper


def save_data(path, data, **kwargs):
    """
    Save a DataFrame as csv, creating all intermediate directories.

    Extra kwargs will be passed through to pandas.DataFrame.to_csv.

    """

    if not path.name.endswith(".csv"):
        raise ValueError("Path must end with .csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, **kwargs)


def save_figure(path, figure=None, **kwargs):
    """
    Save a matplotlib figure as pdf, creating all intermediate directories.

    A figure may be passed explicitly, or the matplotlib current figure will
    be used.

    Extra kwargs will be passed through to matplotlib.pyplot.savefig.

    """

    if not path.name.endswith(".pdf"):
        raise ValueError("Path must end with .pdf")
    if figure is None:
        figure = plt.gcf()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), **kwargs)


def open_dashboards(path, **kwargs):
    """
    Open Python shelve containing dashboard data

    in dictionary form.

    """

    os.chdir(path)
    dashboards_shlf = shelve.open('dashboards.shelve', writeback=True)

    return dashboards_shlf


def categorical_cmap(numUniqueSamples, uniqueSampleNames, numCatagories,
                     reverseSampleOrder=False, flipColorOrder=False,
                     cmap='seaborn_default', continuous=False):
    """
    Generate a categorical colormap of length numUniqueSamples.
    cmap = 'tab10', 'seaborn_default', etc.
    """
    if cmap == 'seaborn_default':
        channel_color_list = sns.color_palette()
    else:
        # specify and apply color list index order
        base_colors = plt.get_cmap(cmap)
        channel_color_list = [base_colors(i) for i in range(base_colors.N)]

    color_order = [3, 0, 2, 4, 8, 6, 1, 5, 9, 7]
    if flipColorOrder:
        color_order = np.flip(color_order)
    cmap = colors.ListedColormap([channel_color_list[i] for i in color_order])

    numSubcatagories = math.ceil(numUniqueSamples/numCatagories)

    if numCatagories > plt.get_cmap(cmap).N:
        raise ValueError('Too many categories for colormap.')
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, numCatagories))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(numCatagories, dtype=int))

    cols = np.zeros((numCatagories * numSubcatagories, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, numSubcatagories).reshape(numSubcatagories, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, numSubcatagories)
        arhsv[:, 2] = np.linspace(chsv[2], 1, numSubcatagories)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * numSubcatagories:(i + 1) * numSubcatagories, :] = rgb
    cmap = colors.ListedColormap(cols)

    # trim colors if necessary
    if len(cmap.colors) > numUniqueSamples:
        trim = len(cmap.colors) - numUniqueSamples
        cmap_colors = cmap.colors[:-trim]
        cmap = colors.ListedColormap(cmap_colors, name='from_list', N=None)

    color_dict = dict(
        zip(sorted(uniqueSampleNames, reverse=reverseSampleOrder), cmap.colors)
        )

    return color_dict


def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components
    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.
    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig
    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized
    Note: does not work correctly with round=True in Basemap
    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line
    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if
    # no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet)
                      or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(
        #    item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder + 100)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder + 100)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder + 100)

    # dpi is a savefig keyword argument, but treat it as special]
    # since it is important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


def log_banner(log_function, msg):
    """Call log_function with a blank line, msg, and an underline."""
    log_function("")
    log_function(msg)
    log_function("-" * len(msg))


def log_multiline(log_function, msg):
    """Call log_function once for each line of msg."""
    for line in msg.split("\n"):
        log_function(line)


@module
def read_data(data, config):
    """Read the data file specified in the config."""

    # The data arg to this module is ignored.
    # csv or parquet file formats are accepted.

    if str(config.data_path.resolve()).endswith('.csv'):
        data = pd.read_csv(config.data_path)
    elif str(config.data_path.resolve()).endswith('.parquet'):
        data = pd.read_parquet(config.data_path)

    return data


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
    # Reorder id_columns to match order specified in config.
    meta_columns = sample.columns.difference(config.id_channels)
    sample = pd.concat(
        {
            'metadata': sample[meta_columns],
            'data': sample[config.id_channels],
        },
        axis=1
    )

    save_data(config.filtered_data_path / 'full.csv', sample, index=False)

    return sample


@module
def gate_bias(data, config):
    """Generate kernel/jittered subsets of data."""

    # get the kernel DataFrame
    kernel = data['data'].copy()
    cutoff_values = np.percentile(
        kernel, [config.kernel_low, config.kernel_high], axis=0
    )
    cutoff = pd.DataFrame(
        cutoff_values.T, index=kernel.columns, columns=['low', 'high']
    )
    kernel.mask(kernel < cutoff['low'], inplace=True)
    kernel.mask(kernel > cutoff['high'], inplace=True)

    kernel_bias = kernel.copy()
    bias_values = np.abs(kernel_bias.max() * config.jitter)
    kernel_bias.mask(
        (kernel_bias > -bias_values) & (kernel_bias < bias_values),
        inplace=True
    )

    kernel = data[kernel.notna().any(axis=1)]
    kernel_bias = data[kernel_bias.notna().any(axis=1)]

    # plot distributions for each channel
    sns.set(style='white')
    for channel in data['data']:
        logger.info('Plotting %s', channel)
        ax = sns.distplot(
            kernel[('data', channel)], kde=True, hist=False,
            kde_kws={'lw': 2.0}, label=channel
        )
        ax.axvline(0, lw=1, c="black", label="gate \u00B1 jitter")
        for f in (1, -1):
            ax.axvline(bias_values[channel] * f, lw=1, ls=":", c="black")
        ax.set_ylabel('density')
        ax.set_xlabel('signal intensity')
        ax.get_yaxis().set_visible(True)
        ax.legend()

        save_figure(
            config.figure_path /
            'gates' /
            f'gate_{channel}.pdf')
        plt.close('all')

    output_path = config.filtered_data_path
    save_data(output_path / 'kernel.csv', kernel, index=False)
    save_data(output_path / 'kernel_bias.csv', kernel_bias, index=False)

    choices = {
        FilterChoice['full']: data,
        FilterChoice['kernel']: kernel,
        FilterChoice['kernel_bias']: kernel_bias,
    }
    assert set(choices) == set(FilterChoice), \
        "Did not handle all FilterChoices"
    result = choices[config.filter_choice]

    return result


@module
def data_discretization(data, config):
    """Binarize data."""

    data_bool = data['data'] > 0
    # Add a level to the column index.
    data_bool = pd.concat({'boolean': data_bool}, axis=1)
    data = pd.concat([data, data_bool], axis=1)
    save_data(config.filtered_data_path / 'overall.csv', data, index=False)

    data_bool_unique = data['boolean'].drop_duplicates()
    g = sns.clustermap(data_bool_unique)
    for label in g.ax_heatmap.get_xticklabels():
        label.set_rotation(45)
    g.ax_heatmap.set_yticks([])
    save_figure(config.figure_path / 'unique_vectors.pdf')
    plt.close('all')

    return data


@module
def boolean_classifier(data, config):
    """Map Boolean vectors to cell states."""

    data[('class', 'boolean')] = None
    for class_name, terms in config.classes.items():
        positives = pd.Series(np.ones(len(data), dtype=bool), index=data.index)
        for term in terms:
            col = data[('boolean', term.name)]
            if term.negated:
                col = ~col
            positives = positives[col]
        indexer = (positives.index, ('class', 'boolean'))
        conflicts = set(data.loc[indexer][data.loc[indexer].notna()])
        if conflicts:
            raise ValueError(
                f"Boolean class '{class_name}' overlaps with {conflicts}"
            )
        data.loc[indexer] = class_name

    classified_counts = data.groupby(('class', 'boolean')).size()
    classified_counts.sort_values(ascending=False, inplace=True)
    classified_counts.name = 'cell_count'
    classified_counts.index.name = 'class'
    classified_counts = classified_counts.reset_index()
    log_banner(logger.info, "Boolean classifications")
    log_multiline(logger.info, classified_counts.to_string(index=False))

    pct_classified = classified_counts['cell_count'].sum() / len(data) * 100
    logger.info("(accounting for %.2f%% of the data)", pct_classified)

    unclassified = data.loc[data['class', 'boolean'].isna(), 'boolean']
    cols = list(data['boolean'].columns)
    unclassified_counts = unclassified.groupby(cols).size()
    unclassified_counts.sort_values(ascending=False, inplace=True)
    unclassified_counts.name = 'cell_count'
    unclassified_counts = unclassified_counts.reset_index().astype(int)
    log_banner(logger.info, "Unspecified boolean classes")
    log_multiline(logger.info, unclassified_counts.to_string(index=False))

    logger.info(
        "(accounting for the remaining %.2f%% of the data)",
        100 - pct_classified
    )

    # keep track of unclassified cells by referring to them as 'unclassified'
    data[('class', 'boolean')] = [
        'unclassified' if i is None else i for i in data[('class', 'boolean')]
        ]

    return data


@module
def frequent_cell_states(data, config):
    """Identify cell states greater than <alpha> % of cells
    in one or more samples."""

    cell_states = []
    for s, total in enumerate(data.groupby(
      [('metadata', 'tissue'), ('metadata', 'timepoint'),
       ('metadata', 'replicate'), ('metadata', 'status')]
      )):
        for name, specific in total[1].groupby([('class', 'boolean')]):
            specific.reset_index(inplace=True, drop=True)
            if len(specific)/len(total) >= config.alpha:
                cell_states.append(
                    specific.loc[
                        0, [('class', 'boolean')] +
                        [('boolean', i) for i in config.id_channels]])

    unique_cell_states = (
        pd.DataFrame(cell_states)
        .reset_index(drop=True)
        .drop_duplicates()
        .reset_index(drop=True)
        )

    save_data(
        config.alpha_vectors_path / f'alpha_vectors_{config.alpha}%.csv',
        unique_cell_states, index=False
        )

    # get cell states at alpha cutoff unique to treatment status
    sets = []
    for name, group in data.groupby([('metadata', 'status')]):
        cell_state_set = set(group[('class', 'boolean')])
        sets.append(cell_state_set)
    print(set.union(*sets) - set.intersection(*sets))

    return data


@module
def initialize_dashboards(data, config):
    """Create a dictionary to store
    celltype stats for aggregate plot."""

    path = config.dashboards_path
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'Path exists')
        os.remove(os.path.join(path, 'dashboards.shelve'))

    dashboards_shlf = shelve.open(os.path.join(path, 'dashboards.shelve'))

    dashboards = {}
    for celltype in set(data[('class', 'boolean')]):
        if celltype != 'unclassified':
            dashboards[celltype] = {}
            dashboards[celltype]['signature'] = (
                sorted([j for j in [str(i) for i in config.classes[celltype]]
                        if not j.startswith('~')])
                )
            dashboards[celltype]['lineage'] = config.lineages[celltype]
            dashboards[celltype]['landmark'] = config.landmarks[celltype]
    dashboards_shlf.update(dashboards)
    dashboards_shlf.close()
    print('Dashboards dictionary initialized.')

    return data


@module
def check_celltypes(data, config):
    """Bi-variate plot of protein expression for
    selected cell types"""

    test = data[
        [('data', config.xaxis_marker),
         ('data', config.yaxis_marker),
         ('class', 'boolean')]][
        (data[('metadata', 'tissue')] == 'blood') &
        (data[('metadata', 'status')] == 'naive') &
        ((data[('class', 'boolean')] == config.celltype1) |
         (data[('class', 'boolean')] == config.celltype2))]

    color_dict = {
        config.celltype1: 'b',
        config.celltype2: 'orange'}

    color_list = [color_dict[i] for i in test[('class', 'boolean')]]

    sns.set_style('whitegrid')
    plt.scatter(
        test[('data', config.xaxis_marker)],
        test[('data', config.yaxis_marker)],
        c=color_list, s=1.5
        )
    plt.xlabel(config.xaxis_marker)
    plt.ylabel(config.yaxis_marker)

    save_figure(config.figure_path / 'celltype_scatter.pdf')

    plt.close('all')

    return data


@module
def celltype_barcharts(data, config):
    """plot cell frequency barcharts with SEM bars"""

    classified_data = data[data[('class', 'boolean')] != 'unclassified']

    metadata_cols = [
        i for i in data.columns if i[1] in
        ['timepoint', 'tissue', 'status', 'replicate']
        ]

    per_well_counts = (
        classified_data
        .groupby(metadata_cols + [('class', 'boolean')])
        .size()
        .replace(to_replace='NaN', value=0)
        .astype(int)
        )

    per_well_percentages = (
        per_well_counts.groupby(metadata_cols)
        .apply(lambda x: (x / x.sum())*100)
        )

    replicate_groupby_obj = (
        per_well_percentages
        .groupby(
            [i for i in metadata_cols if i[1] != 'replicate'] +
            [('class', 'boolean')])
        )

    plot_input = pd.concat(
        [replicate_groupby_obj.mean(), replicate_groupby_obj.sem()], axis=1
        )
    plot_input.rename(columns={0: 'mean', 1: 'sem'}, inplace=True)

    # pad missing cell types with 0.0 for mean and SEM
    idx = pd.MultiIndex.from_product(
        [classified_data['metadata', 'status'].unique(),
         classified_data['metadata', 'timepoint'].unique(),
         classified_data['metadata', 'tissue'].unique(),
         list(config.classes.keys())],
        names=[i for i in metadata_cols if i[1] != 'replicate'] +
        [('class', 'boolean')]
        )
    plot_input = plot_input.reindex(idx, fill_value=0.0).sort_index()

    for (tp_name, ts_name), group in plot_input.groupby(
      [i for i in metadata_cols if i[1] in ['timepoint', 'tissue']]
      ):

        print(
            f'Plotting celltype barcharts for {ts_name} at '
            f'time point {tp_name}.'
            )

        group.sort_index(
            level=[('class', 'boolean'), ('metadata', 'status')],
            ascending=[True, False], inplace=True
            )

        condition_color_dict = categorical_cmap(
            numUniqueSamples=len(data[('metadata', 'status')].unique()),
            uniqueSampleNames=sorted(data[('metadata', 'status')].unique()),
            numCatagories=10,
            reverseSampleOrder=True,
            flipColorOrder=True,
            cmap='seaborn_default',
            continuous=False
            )

        sns.set_style('whitegrid')
        g = group['mean'].plot(
            yerr=group['sem'], kind='bar', grid=False, width=0.78, linewidth=1,
            figsize=(20, 10), color=[condition_color_dict['naive'],
                                     condition_color_dict['gl261']],
            alpha=1.0, title=f'{ts_name}, {tp_name}'
            )

        xlabels = [
            item.get_text() for item in g.get_xticklabels()]
        xlabels_update = [
            i.strip('()').split(',')[-1].strip(' ') for i in xlabels
            ]
        xlabels_update = [xlabel.replace(
            'neg', '$^-$').replace('pos', '$^+$') for xlabel in xlabels_update]
        g.set_xticklabels(xlabels_update)

        for item in g.get_xticklabels():
            item.set_size(15)
            item.set_weight('normal')

        for item in g.get_yticklabels():
            item.set_size(15)

        g.set_xlabel(xlabel='', size=18, weight='bold')
        g.set_ylabel(ylabel='% tissue composition', size=18, weight='bold')
        g.set_ylim(0.0, 100.0)

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        g.xaxis.grid(False)

        g.set_title(g.get_title(), size=18, weight='bold', y=1.02)

        legend_elements = [
            Patch(facecolor=condition_color_dict['naive'],
                  edgecolor=condition_color_dict['naive'], label='naive'),
            Patch(facecolor=condition_color_dict['gl261'],
                  edgecolor=condition_color_dict['gl261'], label='gl261')
            ]
        legend_text_properties = {'size': 20, 'weight': 'bold'}
        plt.legend(
            handles=legend_elements,
            prop=legend_text_properties,
            loc='upper right'
            )

        plt.tight_layout()

        save_figure(
            config.figure_path /
            'frequency_barcharts(classified)' /
            f'{ts_name}_{tp_name}.pdf'
            )

        plt.close('all')

    return data


@module
def celltype_stats(data, config):
    """Compute frequency statistics between classified vectors from
    test and control groups"""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)

    classified_data = data[data[('class', 'boolean')] != 'unclassified']

    metadata_cols = [
        i for i in data.columns if i[1] in
        ['timepoint', 'tissue', 'status', 'replicate']
        ]

    per_well_counts = (
        classified_data
        .groupby(metadata_cols + [('class', 'boolean')])
        .size()
        .replace(to_replace='NaN', value=0)
        .astype(int)
        )

    per_well_percentages = (
        per_well_counts.groupby(metadata_cols)
        .apply(lambda x: (x / x.sum())*100)
        )

    # pad per_well_percentages with Cartesian product of
    # replicate, status, timepoint, tissue, and cell class
    arrays = [
        classified_data[('metadata', 'replicate')].unique(),
        classified_data[('metadata', 'status')].unique(),
        classified_data[('metadata', 'timepoint')].unique(),
        classified_data[('metadata', 'tissue')].unique(),
        classified_data[('class', 'boolean')].unique()
        ]
    idx = pd.MultiIndex.from_product(
        arrays, names=[
            ('metadata', 'replicate'),
            ('metadata', 'status'),
            ('metadata', 'timepoint'),
            ('metadata', 'tissue'),
            ('class', 'boolean')
            ]
        )

    per_well_percentages = (
        per_well_percentages
        .reindex(idx, fill_value=0.0)
        .sort_index()
        )

    stats_df = pd.DataFrame(
        columns=['timepoint', 'tissue', 'cell_class', 'gl261_percent',
                 'naive_percent', 'dif', 'ratio', 't-stat', 'pval']
        )
    row_idx = 0

    for (tp_name, ts_name, ct_name), group in per_well_percentages.groupby(
      [('metadata', 'timepoint'), ('metadata', 'tissue'),
       ('class', 'boolean')]):

        t_stat, pval = ttest_ind(
            group[:, 'gl261'], group[:, 'naive'],
            axis=0, equal_var=True, nan_policy='propagate')
        gl261_mean = group[:, 'gl261'].mean()
        naive_mean = group[:, 'naive'].mean()
        dif = (gl261_mean-naive_mean)
        ratio = np.log2((0.01 + gl261_mean) / (0.01 + naive_mean))

        stats_df.loc[row_idx] = [
            tp_name, ts_name, ct_name, gl261_mean, naive_mean,
            dif, ratio, t_stat, pval]
        row_idx += 1

    # perform FDR correction
    stats = importr('stats')
    p_adjust = stats.p_adjust(
        FloatVector(stats_df['pval'].tolist()),
        method='BH')
    stats_df['qval'] = p_adjust

    stats_df.sort_values(
        by='qval', inplace=True, ascending=True
        )

    if stats_df[stats_df['qval'] <= 0.05].empty:
        print('No statistically significant differences to report.')
    else:
        print('Statistically significant differences (qval<=0.05):')
        print(stats_df[stats_df['qval'] <= 0.05])

    save_data(config.stats_path / 'celltype_stats.csv', stats_df, index=False)

    for name, group in stats_df.groupby(['cell_class']):
        group_dif = group.pivot_table(
            index='tissue', columns='timepoint', values='dif', dropna=False)
        dashboards_shlf[name]['dif_heatmap'] = group_dif

        group_ratio = group.pivot_table(
            index='tissue', columns='timepoint', values='ratio', dropna=False)
        dashboards_shlf[name]['ratio_heatmap'] = group_ratio

        group_qval = group.pivot_table(
            index='tissue', columns='timepoint', values='qval', dropna=False)
        group_qval[group_qval > 0.05] = np.nan
        for col_idx in group_qval:
            series = group_qval[col_idx]
            for i in series.iteritems():
                row_idx = i[0]
                if not i[1] == np.nan:
                    if 0.01 < i[1] <= 0.05:
                        group_qval.loc[row_idx, col_idx] = '*'
                    elif 0.001 < i[1] <= 0.01:
                        group_qval.loc[row_idx, col_idx] = '**'
                    elif i[1] <= 0.001:
                        group_qval.loc[row_idx, col_idx] = '***'
        group_qval.replace(to_replace=np.nan, value='', inplace=True)
        dashboards_shlf[name]['qval_heatmap'] = group_qval

    # plot (q-val vs. magnitude) and (q-val vs. ratio) scatter plots of
    # statistically-significant classified data.
    plot_input = stats_df[stats_df['qval'] <= 0.05].copy()

    tissue_color_dict = categorical_cmap(
        numUniqueSamples=len(plot_input['tissue'].unique()),
        uniqueSampleNames=plot_input['tissue'].unique(),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=False,
        cmap='seaborn_default',
        continuous=False,
        )

    for tp, group in plot_input.groupby(['timepoint']):

        group['hue'] = [tissue_color_dict[i] for i in group['tissue']]
        group['qval'] = -(group['qval'].apply(np.log10))

        for x, y in [('dif', 'qval'), ('dif', 'ratio')]:

            fig, ax = plt.subplots()
            plt.scatter(
                group[x],
                group[y],
                c=group['hue'],
                s=60,
                )
            ax.set_axisbelow(True)
            plt.grid(True, linestyle='dashed')
            plt.axvline(x=0.0, linewidth=1.0,
                        linestyle='solid', color='k', alpha=1.0)
            ax.set_title(
                f'time point = {str(tp)}', size=20, y=1.02, weight='bold'
                )

            if y == 'qval':
                print(
                    'Plotting mean difference vs. q-val for' +
                    ' statistically-significant' +
                    f' classified data at time point = {str(tp)}.'
                      )
                ax.set_xlabel('mean difference (%)', size=15, weight='normal')
                ax.set_ylabel('-log10(q-val)', size=15, weight='normal')
            elif y == 'ratio':
                print(
                    'Plotting mean difference vs. weighted log2(fold-change)'
                    ' for statistically-significant' +
                    f' classified data at time point = {str(tp)}.'
                      )
                ax.set_xlabel('mean difference (%)', size=15, weight='normal')
                ax.set_ylabel(
                    'weighted log2(fold-change)', size=15, weight='normal'
                    )
                plt.axhline(y=0.0, linewidth=1.0,
                            linestyle='solid', color='k', alpha=1.0)

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # point annotations
            for i, txt in enumerate(group['cell_class']):
                txt = txt.replace(
                    'neg', '$^-$').replace('pos', '$^+$')
                ax.annotate(
                    txt,
                    xy=(group[x].iloc[i], group[y].iloc[i]),
                    xytext=None,  # (x, y) positions of text
                    size=10,
                    )

            # legend
            handles = []
            for key, value in tissue_color_dict.items():
                line = mlines.Line2D(
                    [], [], color=value, linestyle='None', marker='o',
                    markersize=10, label=key)
                handles.append(line)
            legend_text_properties = {'weight': 'bold'}
            plt.legend(handles=handles, prop=legend_text_properties)

            plt.tight_layout()

            save_figure(
                config.figure_path /
                'stats_plots' /
                f'{x}_vs_{y}' /
                f'{str(tp)}.pdf'
                )

            plt.close('all')

    dashboards_shlf.close()

    return data


@module
def vector_barcharts(data, config):
    """plot cell frequency barcharts with SEM bars for all Boolean vectors"""

    unclassified_data = data[data[('class', 'boolean')] == 'unclassified']

    metadata_cols = [
        i for i in data.columns if i[1] in
        ['timepoint', 'tissue', 'status', 'replicate']
        ]

    boolean_cols = [
        i for i in data.columns if i[0] == 'boolean'
        ]

    per_well_counts = (
        unclassified_data
        .groupby(metadata_cols + boolean_cols)
        .size()
        .replace(to_replace='NaN', value=0)
        .astype(int)
        )

    per_well_percentages = (
        per_well_counts.groupby(metadata_cols)
        .apply(lambda x: (x / x.sum())*100)
        )

    replicate_groupby_obj = (
        per_well_percentages
        .groupby(
            [i for i in metadata_cols if i[1] != 'replicate'] +
            boolean_cols)
        )

    plot_input = pd.concat(
        [replicate_groupby_obj.mean(), replicate_groupby_obj.sem()], axis=1
        )
    plot_input.rename(columns={0: 'mean', 1: 'sem'}, inplace=True)
    vectors_total = [
        [j for j in i[0] if type(j) == bool] for i in plot_input.iterrows()
        ]
    vectors_total = [
        ['1' if j is True else '0' for j in i] for i in vectors_total
        ]
    vectors_total = [" ".join(i) for i in vectors_total]
    plot_input['vector'] = vectors_total

    unique_vectors = (
        unclassified_data['boolean'].drop_duplicates().reset_index(drop=True)
        )
    vectors_unique = [
        [j for j in i[1] if type(j) == bool] for i in unique_vectors.iterrows()
        ]
    vectors_unique = [
        ['1' if j is True else '0' for j in i] for i in vectors_unique
        ]
    vectors_unique = [" ".join(i) for i in vectors_unique]
    vector_dict = dict(zip(vectors_unique, unique_vectors.index))

    unique_vectors.reset_index(drop=False, inplace=True)
    unique_vectors.rename(columns={'index': 'vector_id'}, inplace=True)

    plot_input['vector_id'] = [vector_dict[i] for i in plot_input['vector']]

    save_data(
        config.figure_path /
        'frequency_barcharts(unspecified)' /
        f'vector_legend.csv',
        unique_vectors, index=False
        )

    for (tp_name, ts_name), group in plot_input.groupby(
      [i for i in metadata_cols if i[1] in ['timepoint', 'tissue']]
      ):

        print(
            f'Plotting vector barcharts for {ts_name} at '
            f'time point {tp_name}.'
            )

        group.sort_index(
            level=boolean_cols + [('metadata', 'status')],
            ascending=[True for i in boolean_cols] + [False], inplace=True
            )
        group.set_index('vector_id', inplace=True)

        sns.set_style('whitegrid')
        g = group['mean'].plot(
            yerr=group['sem'], kind='bar', grid=False, width=0.78, linewidth=1,
            figsize=(20, 10), color=['b', 'g'], alpha=0.6,
            title=f'{ts_name}, {tp_name}'
            )

        for item in g.get_xticklabels():
            item.set_size(4)
            item.set_weight('normal')

        for item in g.get_yticklabels():
            item.set_size(15)

        g.set_xlabel(xlabel='', size=18, weight='bold')
        g.set_ylabel(ylabel='% tissue composition', size=18, weight='bold')
        g.set_ylim(0.0, 100.0)

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        g.xaxis.grid(False)

        g.set_title(g.get_title(), size=18, weight='bold', y=1.02)

        legend_elements = [
            Patch(facecolor='b', edgecolor='b', label='naive'),
            Patch(facecolor='g', edgecolor='g', label='gl261')
            ]
        legend_text_properties = {'size': 20, 'weight': 'bold'}
        plt.legend(
            handles=legend_elements,
            prop=legend_text_properties,
            loc='upper right'
            )

        plt.tight_layout()

        save_figure(
            config.figure_path /
            'frequency_barcharts(unspecified)' /
            f'{ts_name}_{tp_name}.pdf'
            )

        plt.close('all')

    return data


@module
def vector_stats(data, config):
    """Compute frequency statistics between unclassified vectors from
    test and control groups"""

    unclassified_data = data[data[('class', 'boolean')] == 'unclassified']

    metadata_cols = [
        i for i in data.columns if i[1] in
        ['timepoint', 'tissue', 'status', 'replicate']
        ]

    boolean_cols = [
        i for i in data.columns if i[0] == 'boolean'
        ]

    per_well_counts = (
        unclassified_data
        .groupby(metadata_cols + boolean_cols)
        .size()
        .replace(to_replace='NaN', value=0)
        .astype(int)
        )

    per_well_percentages = (
        per_well_counts.groupby(metadata_cols)
        .apply(lambda x: (x / x.sum())*100)
        )

    # add vector strings to multiindex
    vectors = [
        [j for j in i[0] if type(j) == bool] for
        i in per_well_percentages.iteritems()
        ]
    vectors = [
        ['1' if j is True else '0' for j in i] for i in vectors
        ]
    vectors = [" ".join(i) for i in vectors]
    per_well_percentages = pd.DataFrame(
        per_well_percentages).rename(columns={0: 'percent'})
    per_well_percentages['vector'] = vectors
    per_well_percentages.set_index('vector', append=True, inplace=True)
    per_well_percentages = per_well_percentages['percent']

    stats_df = pd.DataFrame(
        columns=['timepoint', 'tissue', 'vector', 'gl261_percent',
                 'naive_percent', 'dif', 'ratio', 't-stat', 'pval']
        )
    row_idx = 0

    for (tp_name, ts_name, v_name), group in per_well_percentages.groupby(
      [('metadata', 'timepoint'), ('metadata', 'tissue')] + ['vector']):

        # ensure at least 2 replicates per treatment group
        # (avoids runtime warnings and NaNs in statistics computation)
        if all(
            i >= 2 for i in list(group.groupby(('metadata', 'status')).size())
          ):

            if len(list(group.groupby(('metadata', 'status')).size())) >= 2:
                a = group
                t_stat, pval = ttest_ind(
                    group[:, 'gl261'], group[:, 'naive'],
                    axis=0, equal_var=True, nan_policy='propagate')

                gl261_mean = group[:, 'gl261'].mean()
                naive_mean = group[:, 'naive'].mean()
                dif = (gl261_mean-naive_mean)
                ratio = np.log2((0.01 + gl261_mean) / (0.01 + naive_mean))

                stats_df.loc[row_idx] = [
                    tp_name, ts_name, v_name, gl261_mean, naive_mean,
                    dif, ratio, t_stat, pval]
                row_idx += 1

    # perform FDR correction
    stats = importr('stats')
    p_adjust = stats.p_adjust(
        FloatVector(stats_df['pval'].tolist()),
        method='BH')
    stats_df['qval'] = p_adjust

    stats_df.sort_values(
        by='qval', inplace=True, ascending=True
        )

    if stats_df[stats_df['qval'] <= 0.05].empty:
        print('No statistically significant differences to report.')
    else:
        print('Statistically significant differences (qval<=0.05):')
        print(stats_df[stats_df['qval'] <= 0.05])

    save_data(config.stats_path / 'vector_stats.csv', stats_df, index=False)

    return data


@module
def replicate_counts(data, config):
    """Plot cell type percentages per tissue per replicate"""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)

    tissue_color_dict = categorical_cmap(
        numUniqueSamples=len(data[('metadata', 'tissue')].unique()),
        uniqueSampleNames=sorted(data[('metadata', 'tissue')].unique()),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=False,
        cmap='seaborn_default',
        continuous=False
        )

    num_conditions = len(data[('metadata', 'status')].unique())
    num_timepoints = len(data[('metadata', 'timepoint')].unique())
    num_replicates = len(data[('metadata', 'replicate')].unique())

    # per condition per timepoint hue scheme
    # list_of_lists = [
    #     [i]*num_replicates for i in
    #     range(num_conditions * num_timepoints)
    #     ]

    # per condition hue scheme
    list_of_lists = [
        [i]*num_replicates for i in
        range(num_conditions)] * num_timepoints

    # flatten list of lists
    hue_list = [item for sublist in list_of_lists for item in sublist]

    color_dict = categorical_cmap(
        numUniqueSamples=len(set(hue_list)),
        uniqueSampleNames=sorted(set(hue_list)),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=True,
        cmap='seaborn_default',
        continuous=False
        )

    for celltype in sorted(data[('class', 'boolean')].unique()):
        if not celltype == 'unclassified':
            print(celltype)

            axes_objs = tuple(
                [f'ax{i}' for i in
                 range(1, len(data['metadata', 'tissue'].unique()) + 1)]
                 )

            sns.set(style='whitegrid')
            fig, axes_objs = plt.subplots(
                5, figsize=(7, 6), sharex=True
                )

            fig.suptitle(celltype, fontsize=10, fontweight='bold', y=0.99)

            maxima = []

            for ax, tissue in zip(
              axes_objs, sorted(data[('metadata', 'tissue')].unique())):
                y_percents = []
                x_labels = []
                spec_ts_ct = data[
                    (data[('metadata', 'tissue')] == tissue) &
                    (data[('class', 'boolean')] == celltype)
                    ]
                spec_denom_ts_ct = data[
                    (data[('metadata', 'tissue')] == tissue)
                    ]

                for tp in sorted(data[('metadata', 'timepoint')].unique()):
                    spec_ts_ct_tp = spec_ts_ct[
                        (spec_ts_ct[('metadata', 'timepoint')] == tp)
                        ]
                    spec_denom_ts_ct_tp = spec_denom_ts_ct[
                        (spec_denom_ts_ct[('metadata', 'timepoint')] == tp)
                        ]

                    for status in sorted(
                      data[('metadata', 'status')].unique(), reverse=True):
                        spec_ts_ct_tp_st = spec_ts_ct_tp[
                            (spec_ts_ct_tp[('metadata', 'status')] == status)
                            ]
                        spec_denom_ts_ct_tp_st = spec_denom_ts_ct_tp[
                            (spec_denom_ts_ct_tp[
                                ('metadata', 'status')] == status)
                            ]

                        for rep in sorted(
                          data[('metadata', 'replicate')].unique()):
                            spec_ts_ct_tp_st_rp = spec_ts_ct_tp_st[
                                (spec_ts_ct_tp_st[
                                    ('metadata', 'replicate')] == rep)
                                ]
                            spec_denom_ts_ct_tp_st_rp = spec_denom_ts_ct_tp_st[
                                (spec_denom_ts_ct_tp_st[
                                    ('metadata', 'replicate')] == rep)
                                ]
                            try:
                                y_percents.append(
                                    len(spec_ts_ct_tp_st_rp) /
                                    len(spec_denom_ts_ct_tp_st_rp)
                                    )
                            except ZeroDivisionError:
                                y_percents.append(0.0)

                            x_labels.append(f'{status}, {tp}, {rep}')

                dashboards_shlf[celltype][
                    f'{tissue}_replicate_data'] = y_percents

                ts_tp_st = pd.DataFrame(
                    {tissue: y_percents}, index=x_labels
                    )

                maxima.append(ts_tp_st.max().values)

                sns.barplot(x_labels, y_percents, hue=hue_list,
                            palette=color_dict, linewidth=0.25,
                            edgecolor='b', ax=ax)
                ax.legend_.remove()

                ax.set_ylabel('% composition').set_size(7)
                ax.tick_params(axis='y', which='both', length=0)
                ax.zorder = 1
                for item in ax.get_yticklabels():
                    item.set_rotation(0)
                    item.set_size(7)
                for item in ax.get_xticklabels():
                    item.set_rotation(90)
                    item.set_size(7)

                ax1 = ax.twinx()
                ax1.set_yticklabels([])
                ax1.set_ylabel(tissue, color=tissue_color_dict[tissue],
                               fontweight='bold')
                ax1.tick_params(axis='y', which='both', length=0)

                for n, bar in enumerate(ax.patches):

                    # customize bar width
                    width = (len(x_labels)/75)
                    bar.set_width(width)

                    # adjust misaligned bars
                    if 48 < n < 96:
                        bar_coord = bar.get_x()
                        bar.set_x(bar_coord - 0.43)

            # set global maximum
            for ax in axes_objs:
                ax.set_ylim(0, max(maxima))

            dashboards_shlf[celltype][
                'replicate_data_ymax'] = max(maxima)[0]
            dashboards_shlf[celltype]['replicate_data_xlabels'] = x_labels

            plt.xlim(-1.1, len(x_labels))
            plt.tight_layout()

            save_figure(
                config.figure_path /
                'replicate_counts' /
                f'{celltype}.pdf'
                )

            plt.close('all')

    dashboards_shlf.close()

    return data


@module
def celltype_boxplots(data, config):
    """Generate boxplots of cell type-specific immunomarker expression."""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)

    condition_color_dict = categorical_cmap(
        numUniqueSamples=len(data[('metadata', 'status')].unique()),
        uniqueSampleNames=sorted(data[('metadata', 'status')].unique()),
        numCatagories=10,
        reverseSampleOrder=True,
        flipColorOrder=True,
        cmap='seaborn_default',
        continuous=False
        )

    for celltype, group in sorted(data.groupby([('class', 'boolean')])):
        if celltype != 'unclassified':
            print(celltype)

            for k in ['immunomarker_boxplots', 'scatter_boxplots']:

                if k == 'immunomarker_boxplots':
                    plot_input = (
                        group[[('data', j) for j in [i for i in group['data']]]
                              + [('metadata', 'status')]]
                        .droplevel(level=0, axis=1)
                        .melt(id_vars='status')
                        .sort_values(
                            by=['variable', 'status'], ascending=[True, False])
                        )
                    figsize = (6.4, 4.8)  # mpl default
                    ylim = (-3.0, 3.0)
                    dashboards_shlf[celltype]['abx_signals'] = plot_input

                elif k == 'scatter_boxplots':
                    plot_input = (
                        group[[('metadata', 'fsc'), ('metadata', 'ssc')]
                              + [('metadata', 'status')]]
                        .droplevel(level=0, axis=1)
                        .melt(id_vars='status')
                        .sort_values(
                            by=['variable', 'status'], ascending=[True, False])
                        )
                    figsize = (3.0, 10.0)
                    # ylim = (0.0, 250000)
                    dashboards_shlf[celltype]['scatter_signals'] = plot_input

                fig, ax = plt.subplots(figsize=figsize)

                ax = sns.boxplot(
                    x='variable',
                    y='value',
                    hue='status',
                    data=plot_input,
                    palette=condition_color_dict,
                    linewidth=0.5,
                    )

                sns.despine(left=True)
                ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)

                title = celltype.replace('neg', '$^-$').replace('pos', '$^+$')

                for item in ax.get_xticklabels():
                    item.set_rotation(90)
                    item.set_fontweight('normal')

                ax.set_xlabel('', size=15, weight='normal')
                ax.set_ylabel('intensity', size=15, weight='normal')
                ax.set_title(title, fontweight='bold')

                legend_text_properties = {'size': 10, 'weight': 'normal'}
                legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

                for legobj in legend.legendHandles:
                    legobj.set_linewidth(0)

                # plt.ylim(ylim)
                plt.tight_layout()

                save_figure(
                    config.figure_path /
                    k /
                    f'{celltype}.pdf'
                    )

                plt.close('all')

    dashboards_shlf.close()

    return data


@module
def celltype_boxplots_perChannel(data, config):
    """Generate boxplots of immunomarker-specific
       signal intensity across celltypes."""

    condition_color_dict = categorical_cmap(
        numUniqueSamples=len(data[('metadata', 'status')].unique()),
        uniqueSampleNames=sorted(data[('metadata', 'status')].unique()),
        numCatagories=10,
        reverseSampleOrder=True,
        flipColorOrder=True,
        cmap='seaborn_default',
        continuous=False
        )

    channel_data = data[
        [('class', 'boolean'),
         ('metadata', 'status'),
         ('metadata', 'fsc'),
         ('metadata', 'ssc')] +
        [('data', j) for j in [i for i in data['data']]]
         ].copy()

    channel_data = channel_data[
        channel_data[('class', 'boolean')] != 'unclassified'
        ]

    for channel in sorted(list(channel_data['data'].columns) + ['fsc', 'ssc']):
        print(channel)

        if channel in ['fsc', 'ssc']:
            level_0_idx = 'metadata'
            # ylim = (0.0, 250000)

        else:
            level_0_idx = 'data'
            # ylim = (-3, 3)

        plot_input = (
            channel_data[
                [('class', 'boolean'),
                 ('metadata', 'status'),
                 (level_0_idx, channel)]]
            .droplevel(level=0, axis=1)
            .sort_values(
                by=['status'], ascending=[False])
            )

        # sort in increasing rank-order according to
        # median values of control data.
        order = []
        for name, group in plot_input.groupby(['boolean', 'status']):
            if name[1] == 'naive':
                order.append((name[0], group[channel].median()))
        if channel in ['fsc', 'ssc']:
            reverse = False
        else:
            reverse = True
        order.sort(key=lambda x: x[1], reverse=reverse)
        order = [i[0] for i in order]

        figsize = (6.4, 4.8)  # mpl default
        fig, ax = plt.subplots(figsize=figsize)

        ax = sns.boxplot(
            x='boolean',
            y=channel,
            hue='status',
            data=plot_input,
            palette=condition_color_dict,
            linewidth=0.5,
            order=order,
            )

        sns.despine(left=True)
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        xlabels_update = [
            xlabel.replace('neg', '$^-$').replace('pos', '$^+$') for
            xlabel in xlabels
            ]
        ax.set_xticklabels(xlabels_update)

        for item in ax.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')

        ax.set_xlabel('', size=15, weight='normal')
        ax.set_ylabel('intensity', size=15, weight='normal')
        ax.set_title(channel, fontsize=20, fontweight='bold')

        legend_text_properties = {'size': 10, 'weight': 'normal'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        # plt.ylim(ylim)
        plt.tight_layout()

        save_figure(
            config.figure_path /
            'channel_boxplots' /
            f'{channel}.pdf'
            )

        plt.close('all')

    return data


@module
def celltype_piecharts(data, config):
    """Plot celltype distribution across tissue types."""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)

    tissue_color_dict = categorical_cmap(
        numUniqueSamples=len(data[('metadata', 'tissue')].unique()),
        uniqueSampleNames=sorted(data[('metadata', 'tissue')].unique()),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=False,
        cmap='seaborn_default',
        continuous=False
        )

    # define factor generator
    def factors(n):
        flatten_iter = itertools.chain.from_iterable
        return set(flatten_iter((i, n//i)
                   for i in range(1, int(n**0.5)+1) if n % i == 0))

    # plot piecharts individually
    for celltype, group in data.groupby(('class', 'boolean')):
        if celltype != 'unclassified':
            print(celltype)

            plot_input = group[('metadata', 'tissue')].value_counts()

            dashboards_shlf[celltype]['data'] = plot_input

            color_list = [tissue_color_dict[i] for i in plot_input.index]

            fig, ax = plt.subplots()
            patches, texts, autotexts = ax.pie(
                plot_input, shadow=False, colors=color_list,
                autopct='%1.1f%%', startangle=90, radius=0.1)

            title = celltype.replace('neg', '$^-$').replace('pos', '$^+$')

            ax.set_title(title)
            plt.axis('equal')
            plt.legend(plot_input.index, loc='upper right')

            for w in patches:
                w.set_linewidth(0.25)
                w.set_edgecolor('k')

            save_figure(
                config.figure_path /
                'celltype_piecharts' /
                f'{celltype}.pdf'
                )

            plt.close('all')

    # plot grid of piecharts with radii
    # proportioned according to celltype population size.
    plot_input = data[data[('class', 'boolean')] != 'unclassified'].copy()
    print('Plotting piechart grid.')

    # identify optimal number of rows and columns
    # given total number of celltypes
    num_celltypes = len(plot_input.groupby(('class', 'boolean')))
    factors_list = list(factors(num_celltypes))
    tuple_list = []
    for i, v in enumerate(list(itertools.combinations(factors_list, 2))):
        if v[0] * v[1] == num_celltypes:
            tuple_list.append(v)
    dif_list = []
    for pair in tuple_list:
        dif_list.append(abs(pair[0] - pair[1]))
    tuple_dict = dict(zip(tuple_list, dif_list))
    target_tuple = min(tuple_dict, key=tuple_dict.get)

    the_grid = GridSpec(target_tuple[0], target_tuple[1])
    the_grid.update(hspace=30, wspace=30, left=0.1,
                    right=0.93, bottom=0.08, top=0.83)

    coordinates = [(x, y) for x in range(
        target_tuple[0]) for y in range(target_tuple[1])]
    dif = len(coordinates) - num_celltypes
    if dif > 0:
        coordinates = coordinates[:-dif]

    for coordinate, (name, group) in itertools.zip_longest(
      coordinates, plot_input.groupby(('class', 'boolean'))):

        celltype_cnts_per_tissue = group[('metadata', 'tissue')].value_counts()
        celltype_cnt_total = celltype_cnts_per_tissue.sum()
        total_cells_in_dataset = len(data)

        prcnt_of_total_cells = (celltype_cnt_total/total_cells_in_dataset)*100

        radius = math.sqrt(prcnt_of_total_cells)*10

        dashboards_shlf[name]['percent'] = prcnt_of_total_cells

        ax = plt.subplot(the_grid[coordinate], aspect=1)
        patches, texts = ax.pie(
            celltype_cnts_per_tissue, shadow=False, radius=radius,
            colors=color_list, startangle=90
            )

        title = name.replace('neg', '$^-$').replace('pos', '$^+$')
        ax.set_title(title, y=(radius/2.5), loc='left',
                     fontsize=6.0, weight='normal')

        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

    lgd = plt.gcf().legend(
        celltype_cnts_per_tissue.index,
        bbox_to_anchor=(1.15, 0.9),
        prop={'size': 8}
        )

    save_figure(
        config.figure_path /
        'celltype_piecharts' /
        'combined_piecharts.pdf',
        bbox_extra_artists=(lgd,),
        pad_inches=0.5,
        bbox_inches='tight'
        )

    plt.close('all')

    dashboards_shlf.close()

    return data


@module
def stats_heatmaps(data, config):
    """Plot mean percent difference between control and test data."""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)
    heatmap_row_order = sorted(
        dashboards_shlf, key=lambda x: dashboards_shlf[x]['percent'],
        reverse=True)

    celltype_stats = pd.read_csv(
        f'{config.output_path}/stats/celltype_stats.csv'
        )

    for data_type, k in zip(
      ['difference', 'log2(ratio)'], ['dif', 'ratio']):
        for name, group in celltype_stats.groupby(['tissue']):
            print(
                'Plotting mean percent differences between'
                f' celltypes in the {name}.'
                )

            fig, ax = plt.subplots()
            group = (
                group
                .pivot_table(
                    index='cell_class', columns='timepoint', values=k)
                .reindex(heatmap_row_order)
                )
            group[np.isnan(group)] = 0.0

            qvals = celltype_stats[
                    celltype_stats['tissue'] == name].pivot_table(
                        index='cell_class', columns='timepoint',
                        values='qval').reindex(heatmap_row_order)
            qvals[qvals > 0.05] = np.nan

            for col_idx in qvals:
                series = qvals[col_idx]
                for i in series.iteritems():
                    row_idx = i[0]
                    if not i[1] == np.nan:
                        if 0.01 < i[1] <= 0.05:
                            qvals.loc[row_idx, col_idx] = '*'
                        elif 0.001 < i[1] <= 0.01:
                            qvals.loc[row_idx, col_idx] = '**'
                        elif i[1] <= 0.001:
                            qvals.loc[row_idx, col_idx] = '***'
            qvals.replace(to_replace=np.nan, value='', inplace=True)

            g = sns.heatmap(group, square=True, linewidth=0.5,
                            fmt='', cmap='cividis', center=0.0,
                            annot=qvals, xticklabels=1, yticklabels=1,
                            annot_kws={'size': 7})

            row_labels = [
                item.get_text() for item in g.get_yticklabels()]

            row_labels_update = [
                i
                .replace('neg', '$^-$')
                .replace('pos', '$^+$') for i in row_labels
                ]

            g.set_yticklabels(row_labels_update)

            for item in g.get_yticklabels():
                item.set_rotation(0)
                item.set_size(5)
            for item in g.get_xticklabels():
                item.set_rotation(90)
                item.set_size(5)

            g.set_title(name)
            g.set_xlabel('time point')
            g.set_ylabel('cell type')

            save_figure(
                config.figure_path /
                'stats_heatmaps' /
                data_type /
                f'{name}.pdf'
                )

            plt.close('all')

    dashboards_shlf.close()

    return data


@module
def celltype_heatmap(data, config):
    """Plot heatmap of Boolean immunomarker calls."""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)
    heatmap_row_order = sorted(
        dashboards_shlf, key=lambda x: dashboards_shlf[x]['percent'],
        reverse=True)

    plot_input = data[data[('class', 'boolean')] != 'unclassified'].copy()
    plot_input = plot_input[
        [('class', 'boolean'),
         ('metadata', 'fsc'),
         ('metadata', 'ssc')] +
        [('boolean', j) for j in [i for i in data['boolean']]]
         ]

    for name, group in plot_input.groupby([('class', 'boolean')]):

        # binarize fsc
        mean_fsc = group[('metadata', 'fsc')].mean()
        plot_input[('metadata', 'fsc')] = [
            True if i > 35000 else False for i in [mean_fsc]
            ][0]

        # binarize ssc
        mean_ssc = group[('metadata', 'ssc')].mean()
        plot_input[('metadata', 'ssc')] = [
            True if i > 97000 else False for i in [mean_ssc]
            ][0]

    plot_input.drop_duplicates(inplace=True)

    plot_input = plot_input.droplevel(level=0, axis=1)
    plot_input.set_index('boolean', inplace=True)
    plot_input = plot_input.reindex(heatmap_row_order)
    cols = sorted(
        [j for j in [i for i in data['boolean']]]
        ) + ['fsc', 'ssc']
    plot_input.columns = cols

    fig, ax = plt.subplots()
    ax.set_title('protein expression heatmap', y=1.05, weight='normal')

    ax = sns.heatmap(plot_input, cbar=False, square=True,
                     linewidths=1.75, cmap='rocket_r',
                     xticklabels=1, yticklabels=1, ax=ax
                     )

    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axhline(y=plot_input.shape[0], color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.axvline(x=plot_input.shape[1], color='k', linewidth=1.5)

    ylabels = [
        item.get_text() for item in ax.get_yticklabels()]
    ylabels_update = [ylabel.replace(
        'neg', '$^-$').replace('pos', '$^+$') for ylabel in ylabels]
    ax.set_yticklabels(ylabels_update)

    for item in ax.get_yticklabels():
        item.set_rotation(0)
        item.set_size(7)
        item.set_weight('normal')
    for item in ax.get_xticklabels():
        item.set_rotation(70)
        item.set_size(8)
        item.set_weight('normal')

    ax.set_ylabel('')

    plt.tight_layout()

    save_figure(
        config.figure_path /
        'boolean_heatmap.pdf'
        )

    plt.close('all')

    dashboards_shlf.close()

    return data


@module
def tissue_composition_plots(data, config):
    """Plot individual and cumulative percentage of
       tissues accounted for by successively scarce celltypes."""

    plot_input = data[data[('class', 'boolean')] != 'unclassified']

    tissue_color_dict = categorical_cmap(
        numUniqueSamples=len(plot_input[('metadata', 'tissue')].unique()),
        uniqueSampleNames=sorted(plot_input[('metadata', 'tissue')].unique()),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=False,
        cmap='seaborn_default',
        continuous=False,
        )

    for tissue, group in plot_input.groupby([('metadata', 'tissue')]):
        group = (
            group
            .groupby([('class', 'boolean')])
            .size()
            .reindex(plot_input[('class', 'boolean')].unique())
            .fillna(value=0.0)
            .sort_values(ascending=False)
            .to_frame()
            .rename(columns={0: 'count'})
            )

        group.index.name = 'celltype'

        total_cells_in_tissue = group.sum()

        group['percent'] = (group/total_cells_in_tissue)*100

        tally = 0
        accumulation = []
        for i in group['percent']:
            tally += i
            accumulation.append(tally)
        group['accumulation'] = accumulation

        x = np.arange(0, len(group['accumulation']), 1)
        ax = plt.step(
            x, group['accumulation'], where='post',
            color=tissue_color_dict[tissue])

        ax = sns.barplot(
            x=group.index, y='percent', data=group,
            color=tissue_color_dict[tissue])

        xlabels = [
            item.get_text() for item in ax.get_xticklabels()]
        xlabels_update = [
            i.replace('neg', '$^-$').replace('pos', '$^+$') for i in xlabels
            ]
        ax.set_xticklabels(xlabels_update)

        for item in ax.get_xticklabels():
            item.set_rotation(90)
            item.set_weight('normal')

        ax.set_ylim((0, 100))
        ax.set_xlabel(xlabel='celltype', weight='normal')
        ax.set_ylabel(ylabel='% tissue coverage', weight='normal')
        ax.set_axisbelow(True)
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        save_figure(
            config.figure_path /
            'tissue_composition_plots' /
            f'{tissue}.pdf'
            )

        plt.close('all')

    return data


@module
def shannon_entropy_plot(data, config):
    """Plot stacked barchart of celltype distributions across tissues
       together with their Shannon entropy scores."""

    plot_prep = data[data[('class', 'boolean')] != 'unclassified'].copy()

    tissue_color_dict = categorical_cmap(
        numUniqueSamples=len(plot_prep[('metadata', 'tissue')].unique()),
        uniqueSampleNames=sorted(plot_prep[('metadata', 'tissue')].unique()),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=False,
        cmap='seaborn_default',
        continuous=False,
        )

    plot_input = pd.DataFrame(
        columns=sorted(
            list(plot_prep[('metadata', 'tissue')].unique())) + ['entropy'],
        dtype=float
        )

    for celltype, group in sorted(plot_prep.groupby([('class', 'boolean')])):

        counts = (
            group
            .groupby(('metadata', 'tissue'))
            .size()
            .reindex(sorted(plot_prep[('metadata', 'tissue')].unique()))
            )

        total = counts.sum()
        percents = list(counts/total)
        percents = list(np.nan_to_num(percents))
        from scipy.stats import entropy
        entropy = entropy(pk=percents, base=2)
        percents.append(entropy)
        plot_input.loc[celltype] = percents

    plot_input.sort_values(by='entropy', inplace=True)

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    print('Plotting celltype tissue distributions.')

    ax1 = plot_input['entropy'].plot(
            kind='bar', stacked=False, linewidth=0.0,
            figsize=(9, 7), color='grey', ax=ax1
            )
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = plot_input.loc[:, plot_input.columns != 'entropy'].plot(
            kind='bar', stacked=True, linewidth=0.0,
            figsize=(9, 7), ax=ax2
            )

    ax1.set_title(
        'celltype tissue distribution', y=1.05, size=20,
        fontweight='normal'
        )

    xlabels = [
        item.get_text() for item in ax2.get_xticklabels()]
    xlabels_update = [xlabel.replace(
        'neg', '$^-$').replace('pos', '$^+$') for xlabel in xlabels]
    ax2.set_xticklabels(xlabels_update)

    for item in ax2.get_xticklabels():
        item.set_rotation(90)
        item.set_weight('normal')

    ax1.set_ylabel(
        ylabel='entropy (H)', size=11, weight='normal'
        )

    ax2.set_xlabel(xlabel='celltype', size=15, weight='normal')
    ax2.set_ylabel(ylabel='% of tissue', size=11, weight='normal')

    ax2.set_ylim(0.0, 1.0)

    ax2.legend(bbox_to_anchor=[1.0, 1.0])
    plt.tight_layout()

    save_figure(
        config.figure_path /
        'shannon_entropy.pdf'
        )

    plt.close('all')

    return data


@module
def plot_dashboards(data, config):
    """Plot SYLARAS dashboards using data stored in
       the dashboards shelve object."""

    dashboards_shlf = open_dashboards(path=config.dashboards_path)

    dashboards_order = sorted(
        dashboards_shlf, key=lambda x: dashboards_shlf[x]['percent'],
        reverse=True)

    condition_color_dict = categorical_cmap(
        numUniqueSamples=len(data[('metadata', 'status')].unique()),
        uniqueSampleNames=sorted(data[('metadata', 'status')].unique()),
        reverseSampleOrder=True,
        flipColorOrder=True,
        numCatagories=10,
        cmap='seaborn_default',
        continuous=False,
        )

    tissue_color_dict = categorical_cmap(
        numUniqueSamples=len(data[('metadata', 'tissue')].unique()),
        uniqueSampleNames=sorted(data[('metadata', 'tissue')].unique()),
        numCatagories=10,
        reverseSampleOrder=False,
        flipColorOrder=False,
        cmap='seaborn_default',
        continuous=False
        )

    def dash(fig, ss, i):
        sns.set_style('white')

        celltype = dashboards_order[i]
        data_to_consider = dashboards_shlf[celltype]

        radius = math.sqrt(data_to_consider['percent'])/5

        inner = gridspec.GridSpecFromSubplotSpec(
            50, 100, subplot_spec=ss, wspace=0.05, hspace=0.05)

        ax1 = plt.Subplot(fig, inner[0:10, 0:100])

        # dashboard box outline
        ax1.add_patch(Rectangle(
            (0, -4.02), 1, 5.02, fill=None, lw=1.5, color='grey',
            alpha=1, clip_on=False))

        # celltype alias
        ax1.text(
            0.01, 0.92, celltype.replace('neg', '$^-$').replace('pos', '$^+$'),
            horizontalalignment='left', verticalalignment='top',
            fontsize=50, color='k', stretch=0, fontname='Arial',
            fontweight='bold'
            )

        # landmark population indicator
        if dashboards_shlf[celltype]['landmark'] == 'landmark population':
            ax1.text(
                0.013, 0.53, 'landmark population', horizontalalignment='left',
                verticalalignment='top', fontsize=25, fontweight='bold',
                color='blue'
                )
            ax1.text(
                0.013, 0.33, 'signature: ' +
                ', '.join(dashboards_shlf[celltype]['signature']),
                horizontalalignment='left', fontweight='bold',
                verticalalignment='top', fontsize=22, color='k'
                )

        # cell lineage specification
        if dashboards_shlf[celltype]['lineage'] == 'lymphoid':
            cmap = plt.cm.Greens
        elif dashboards_shlf[celltype]['lineage'] == 'myeloid':
            cmap = plt.cm.Blues
        elif dashboards_shlf[celltype]['lineage'] == 'other':
            cmap = plt.cm.Oranges
        ax1.text(
            0.98, 0.88, dashboards_shlf[celltype]['lineage'],
            horizontalalignment='right', verticalalignment='top',
            fontweight='bold', fontsize=33, color='k'
            )

        # make cell lineage banner
        ax2 = plt.Subplot(fig, inner[0:4, 0:100])
        banner_list = []
        for i in list(range(0, 6)):
            banner_list.append(list(range(0, 100)))
        banner = np.array(banner_list)
        ax2.imshow(
            banner, cmap=cmap, interpolation='bicubic',
            vmin=30, vmax=200
            )
        ax2.grid(False)
        ax2.tick_params(axis='both', which='both', length=0)
        for item in ax2.get_xticklabels():
            item.set_visible(False)
        for item in ax2.get_yticklabels():
            item.set_visible(False)
        fig.add_subplot(ax2)

        # make legend for pvalue characters
        ax1.text(
            0.14, -1.92, '(*) 0.01 < q <= 0.05', horizontalalignment='left',
            verticalalignment='top', fontsize=12, color='k', stretch=0,
            fontname='Arial', fontweight='bold'
            )
        ax1.text(
            0.272, -1.92, '(**) 0.001 < q <= 0.01', horizontalalignment='left',
            verticalalignment='top', fontsize=12, color='k', stretch=0,
            fontname='Arial', fontweight='bold'
            )
        ax1.text(
            0.42, -1.92, '(***) q <= 0.001', horizontalalignment='left',
            verticalalignment='top', fontsize=12, color='k', stretch=0,
            fontname='Arial', fontweight='bold'
            )

        ax1.axis('off')
        fig.add_subplot(ax1)

        # pop. size
        ax3 = plt.Subplot(fig, inner[3:15, 82:95])

        patches, texts = ax3.pie(
            [100, 0], radius=radius, shadow=False,
            startangle=90, colors=[(0.34, 0.35, 0.38)]
            )

        for w in patches:
            w.set_linewidth(0.0)
            w.set_edgecolor('k')

        ax3.text(
            -0.8, 0.85, 'pop. size',
            horizontalalignment='right', verticalalignment='bottom',
            fontsize=18, color='k', fontname='Arial',
            fontweight='bold'
            )

        percent_txt = round(data_to_consider['percent'], 2)
        ax3.text(
            -0.8, 0.53, f'{percent_txt}%',
            horizontalalignment='right', verticalalignment='bottom',
            fontsize=16, color='k', stretch=0, fontname='Arial',
            fontweight='bold'
            )
        fig.add_subplot(ax3)

        # tissue distribution
        ax4 = plt.Subplot(fig, inner[3:15, 62:74])
        colors = [tissue_color_dict[x] for x in data_to_consider['data'].index]

        patches, texts = ax4.pie(
            data_to_consider['data'], shadow=False,
            colors=colors, startangle=90, radius=1.0
            )

        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

        ax4.add_patch(
            Rectangle((-1, -1), 2, 2, fill=None, alpha=0)
            )
        ax4.text(
            -0.57, 0.85, 'tissue dist.',
            horizontalalignment='right', verticalalignment='bottom',
            fontsize=18, color='k', fontname='Arial',
            fontweight='bold'
            )

        fig.add_subplot(ax4)

        # scatter boxplots
        ax5 = plt.Subplot(fig, inner[13:27, 3:9])

        plot_input = data_to_consider['scatter_signals']

        g = sns.boxplot(
            x='variable',
            y='value',
            hue='status',
            data=plot_input,
            palette=condition_color_dict,
            linewidth=0.5,
            ax=ax5,
            )

        sns.despine(left=True)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(15)
            item.set_fontweight('normal')
            item.set_position([0, 0])

        for item in g.get_yticklabels():
            item.set_size(9)
            item.set_fontweight('normal')

        g.tick_params(axis='both', which='both', length=0)

        # g.set_ylim(0, 250000)

        # ax5.set_yticklabels(g.get_yticks())
        # labels = ax5.get_yticklabels()
        # ylabels = [float(label.get_text()) for label in labels]
        # ylabels = ['%.1E' % Decimal(s) for s in ylabels if 0 <= s <= 250000]
        # ax5.set_yticklabels(ylabels)

        ax5.set_xlabel('')
        ax5.set_ylabel('')
        ax5.legend(loc=(-0.4, 1.08), prop={'weight': 'normal', 'size': 16})

        ax5.axhline(
            y=0.0, color='darkgray', linewidth=5.0,
            linestyle='-', zorder=1, alpha=1.0
            )
        lines = g.lines[:-1]
        scatter_raster = lines
        fig.add_subplot(ax5)
        g.grid(
            axis='y', color='grey', linestyle='--', linewidth=0.5, alpha=1.0
            )

        # abx boxplots
        ax6 = plt.Subplot(fig, inner[30:46, 3:52])
        plot_input = data_to_consider['abx_signals']

        g = sns.boxplot(
            x='variable',
            y='value',
            hue='status',
            data=plot_input,
            palette=condition_color_dict,
            linewidth=0.5,
            ax=ax6,
            )

        g.grid(
            axis='y', color='grey', linestyle='--',
            linewidth=0.5, alpha=1.0
            )

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(15)
            item.set_fontweight('normal')
            item.set_position([0, 0.06])

        for item in g.get_yticklabels():
            item.set_size(12)
            item.set_fontweight('normal')

        g.tick_params(axis='both', which='both', length=0)

        # ax6.set_ylim(-3.5, 3.5)
        ax6.set_xlabel('')
        ax6.set_ylabel('')
        ax6.legend_.remove()
        ax6.axhline(
            y=0.0, color='darkgray', linewidth=3.0,
            linestyle='-', zorder=1, alpha=1.0
            )
        lines = g.lines[:-1]
        channel_raster = lines
        fig.add_subplot(ax6)

        # difference heatmaps
        ax7 = plt.Subplot(fig, inner[12:27, 10:32])

        g = sns.heatmap(
            data_to_consider['dif_heatmap'], square=True, vmin=None, vmax=None,
            linecolor='w', linewidths=2.0, cbar=True,
            annot=data_to_consider['qval_heatmap'], fmt='', cmap='cividis',
            center=0.0, xticklabels=True, yticklabels=True,
            annot_kws={'size': 20, 'fontweight': 'bold'}, ax=ax7,
            )

        g.tick_params(axis='both', which='both', length=0)

        plt.gcf().axes[-1].yaxis.tick_right()
        for item in plt.gcf().axes[-1].get_yticklabels():
            item.set_size(10)
            item.set_fontweight('normal')

        g.set_xlabel('')
        g.set_ylabel('')

        for item in g.get_yticklabels():
            item.set_color(tissue_color_dict[item.get_text()])
            item.set_rotation(0)
            item.set_size(14)
            item.set_fontweight('bold')
            item.set_position([0.0, 0.0])

        tissue_abbr = {
            'blood': 'Bl',
            'spleen': 'Sp',
            'nodes': 'Nd',
            'marrow': 'Mw',
            'thymus': 'Th'
            }
        ylabels = [
            tissue_abbr[item.get_text()] for item in g.get_yticklabels()
            ]
        g.set_yticklabels(ylabels)

        for item in g.get_xticklabels():
            item.set_rotation(0)
            item.set_size(14)
            item.set_fontweight('normal')
        ax7.set_title(
            'gl261-naive', fontsize=14, fontweight='bold'
            ).set_position([0.5, 1.01])

        fig.add_subplot(ax7)

        # ratio heatmaps
        ax8 = plt.Subplot(fig, inner[12:27, 32:52])

        g = sns.heatmap(
            data_to_consider['ratio_heatmap'], square=True, vmin=None,
            vmax=None, linecolor='w', linewidths=2.0, cbar=True,
            annot=data_to_consider['qval_heatmap'], fmt='', cmap='cividis',
            center=0.0, xticklabels=True, yticklabels=True,
            annot_kws={'size': 20, 'fontweight': 'bold'}, ax=ax8,
            )

        g.tick_params(axis='both', which='both', length=0)

        plt.gcf().axes[-1].yaxis.tick_right()
        for item in plt.gcf().axes[-1].get_yticklabels():
            item.set_size(10)
            item.set_fontweight('normal')

        g.set_xlabel('')
        g.set_ylabel('')
        g.set_yticklabels([])

        for item in g.get_xticklabels():
            item.set_rotation(0)
            item.set_size(14)
            item.set_fontweight('normal')
        ax8.set_title(
            'log' + '$_2$' + '(gl261/naive)', fontsize=14, fontweight='bold'
            ).set_position([0.50, 1.01])
        fig.add_subplot(ax8)

        # replicate percent composition
        ax9 = plt.Subplot(fig, inner[16:21, 61:99])
        ax10 = plt.Subplot(fig, inner[23:28, 61:99])
        ax11 = plt.Subplot(fig, inner[30:35, 61:99])
        ax12 = plt.Subplot(fig, inner[37:42, 61:99])
        ax13 = plt.Subplot(fig, inner[44:49, 61:99])

        # sns.set(style='whitegrid')
        num_conditions = len(data[('metadata', 'status')].unique())
        num_timepoints = len(data[('metadata', 'timepoint')].unique())
        num_replicates = len(data[('metadata', 'replicate')].unique())

        # per condition hue scheme
        list_of_lists = [
            [i]*num_replicates for i in
            range(num_conditions)] * num_timepoints

        # flatten list of lists
        hue_list = [item for sublist in list_of_lists for item in sublist]

        color_dict = categorical_cmap(
            numUniqueSamples=len(set(hue_list)),
            uniqueSampleNames=set(hue_list),
            numCatagories=10,
            reverseSampleOrder=False,
            flipColorOrder=True,
            cmap='seaborn_default',
            continuous=False
            )

        for e, (ax, tissue) in enumerate(zip(
          [ax9, ax10, ax11, ax12, ax13],
          sorted(data[('metadata', 'tissue')].unique()))):
            sns.barplot(
                data_to_consider['replicate_data_xlabels'],
                data_to_consider[f'{tissue}_replicate_data'],
                hue=hue_list, palette=color_dict, linewidth=0.25,
                edgecolor='k', ax=ax,
                )
            ax.legend_.remove()
            ax.grid(
                axis='y', color='grey', linestyle='--',
                linewidth=0.5, alpha=1.0
                )
            ax.set_ylabel('% composition', size=10, fontweight='bold')
            ax.yaxis.set_label_coords(-0.07, .5)
            ax.set_ylim(
                0, data_to_consider['replicate_data_ymax']
                )
            ax.tick_params(axis='y', which='both', length=0)
            ax.tick_params(axis='x', which='both', length=0)

            ax.zorder = 1

            for item in ax.get_yticklabels():
                item.set_rotation(0)
                item.set_size(10)
                item.set_fontweight('normal')

            ax14 = ax.twinx()
            ax14.set_ylim(0, data_to_consider['replicate_data_ymax'])
            ax14.set_yticklabels([])
            ax14.set_ylabel(
                tissue, color=tissue_color_dict[tissue],
                fontweight='bold', size=18
                )
            ax14.yaxis.set_label_coords(-0.15, .5)
            ax14.tick_params(axis='y', which='both', length=0)

            if e == 0:
                for item in ax.get_xticklabels():
                    if 'naive' in str(item):
                        item.set_color('k')
                    if 'gl261' in str(item):
                        item.set_color('k')
                    item.set_rotation(0)
                    item.set_size(8)
                    item.set_fontweight('bold')
                    item.set_visible(True)
                    item.set_position([0, 1.15])
                xlabels = [('' + item.get_text()[-1])
                           for item in ax.get_xticklabels()]
                ax.set_xticklabels(xlabels)

            else:
                for item in ax.get_xticklabels():
                    item.set_visible(False)

            for n, bar in enumerate(ax.patches):

                # customize bar width
                width = (len(data_to_consider['replicate_data_xlabels'])/75)
                bar.set_width(width)

                # adjust misaligned bars
                if 48 < n < 96:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.43)

        fig.add_subplot(ax9)
        fig.add_subplot(ax10)
        fig.add_subplot(ax11)
        fig.add_subplot(ax12)
        fig.add_subplot(ax13)

        for x in [15.45, 31.48]:
            ax9.axvline(
                x=x, ymin=0, ymax=1.48, c='grey',
                linewidth=1, ls='--', zorder=3, clip_on=False
                )
            ax10.axvline(
                x=x, ymin=0, ymax=1.41, c='grey',
                linewidth=1, ls='--', zorder=3, clip_on=False
                )
            ax11.axvline(
                x=x, ymin=0, ymax=1.41, c='grey',
                linewidth=1, ls='--', zorder=3, clip_on=False
                )
            ax12.axvline(
                x=x, ymin=0, ymax=1.41, c='grey',
                linewidth=1, ls='--', zorder=3, clip_on=False
                )
            ax13.axvline(
                x=x, ymin=0, ymax=1.41, c='grey',
                linewidth=1, ls='--', zorder=3, clip_on=False
                )

        # organize timepoint metadata text for replicate counts
        timepoint_offset = 0.0
        for timepoint in sorted(data[('metadata', 'timepoint')].unique()):

            ax1.text(
                (0.675 + timepoint_offset), -0.37, str(timepoint),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16, fontweight='bold', color='k'
                )
            timepoint_offset += 0.125  # 0.15

        # show all four spines around replicate plots
        for ax in [ax9, ax10, ax11, ax12, ax13]:
            ax.spines['top'].set_linewidth(0.2)
            ax.spines['bottom'].set_linewidth(0.2)
            ax.spines['left'].set_linewidth(0.2)
            ax.spines['right'].set_linewidth(0.2)

        # do not show spines around the other axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        return(scatter_raster, channel_raster)

    dashboards_dir = os.path.join(config.figure_path, 'dashboards')
    if not os.path.exists(dashboards_dir):
        os.makedirs(dashboards_dir)

    for i in range(len(dashboards_order)):
        print(f'Plotting the {dashboards_order[i]} dashboard.')
        fig = plt.figure(figsize=(19, 15))
        outer = gridspec.GridSpec(1, 1)
        scatter_raster, channel_raster = dash(fig, outer[0], i)
        rasterize_list = scatter_raster + channel_raster

        rasterize_and_save(
            os.path.join(dashboards_dir,
                         '%s_dashboard.pdf' % dashboards_order[i]),
            rasterize_list, fig=fig, dpi=300,
            savefig_kw={'bbox_inches': 'tight'})
        plt.close('all')

    print('Plotting the combined dashboards.')
    fig = plt.figure(figsize=(190, 45))
    outer = gridspec.GridSpec(4, 8, wspace=0.1, hspace=0.1)
    rasterize_list = []
    for i, oss in enumerate(outer):
        scatter_raster, channel_raster = dash(fig, oss, i)
        rasterize_list += scatter_raster + channel_raster
    rasterize_and_save(
        os.path.join(dashboards_dir, 'combined_dashboards.pdf'),
        rasterize_list, fig=fig, dpi=200, savefig_kw={'bbox_inches': 'tight'})
    plt.close('all')

    dashboards_shlf.close()

    return data
