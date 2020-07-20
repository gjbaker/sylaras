import logging
import functools
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .config import FilterChoice

logger = logging.getLogger(__name__)


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
    data = pd.read_csv(config.data_path)
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
        save_figure(config.figure_path / f"gate_{channel}.pdf")
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

    return data
