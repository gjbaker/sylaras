# SET GLOBAL CONFIGURATIONS
# import pdb; pdb.set_trace()
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from datetime import datetime
from scipy.stats import binom
import pickle

# set map matplotlib color codes to the default seaborn palette
sns.set_color_codes()


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

# required spearman_rho.py dicrectories
save_dir = os.path.join(project_path, 'correlations', 'data',
                        'Spearman_correlation_analysis')

clustermaps = os.path.join(save_dir, 'clustermaps')


def cdf(data_str, key_str):

    print(color.CYAN + 'Computing binomial distributions per cluster'
          ' getting significant k values' + color.END)

    data_dir = os.path.join(clustermaps, data_str)
    key_dir = os.path.join(data_dir, key_str)

    os.chdir(key_dir)
    pi_mapped_clusters = open('mapped_clusters.pickle', 'rb')
    mapped_clusters = pickle.load(pi_mapped_clusters)

    # binom.cdf(k, n, p)
    cdf_dict = {}

    for col in mapped_clusters[
      ['class', 'tissue', 'arm', 'lineage']]:

        p = mapped_clusters.groupby(col).size()/len(
            mapped_clusters)
        n = mapped_clusters.groupby(['cluster']).size()
        k = mapped_clusters.pivot_table(
            index='cluster', columns=col, values='condition',
            fill_value=0, aggfunc='count').unstack()

        fig, axarr = plt.subplots(
            len(n), len(p), sharex=False,
            sharey=True, figsize=(100, 100))

        fig.subplots_adjust(
            left=0.05, right=0.99, bottom=0.1, top=0.9,
            hspace=0.6, wspace=0.1)

        fig.suptitle(
            'Cumulative Distribution Functions', fontsize=15)

        fig.text(
            0.5, 0.04, 'q', ha='center', size=15, weight='bold')
        fig.text(
            0.01, 0.5, 'F(q)', va='center', rotation='vertical',
            size=15, weight='bold')

        for i, cluster in zip(range(len(n)), n):
            clust_idx = n[n == cluster].index[0]
            for j, idx, percent in zip(range(len(p)), p.index, p):
                x_list = []
                y_list = []
                pval_list = []
                for x in range(cluster):
                    x_list.append(x+1)
                    y_list.append(
                        binom.cdf(k=(x+1), n=cluster, p=percent))
                    pval_list.append(
                        1-binom.cdf(k=(x+1), n=cluster, p=percent))

                data = pd.DataFrame(
                    {'ID': idx, 'cluster': clust_idx, 'q': x_list,
                     'F(q)': y_list, 'pval': pval_list})

                cdf_dict[
                    'k=' + str(k[idx][clust_idx]), 'n=' +
                    str(n[clust_idx]), 'p=' + str(p[idx]),
                    idx, 'cluster_' + str(clust_idx), col] = data

                x = np.linspace(0, cluster, 10000)
                axarr[i, j].plot(x, binom.cdf(x, cluster, p[idx]),
                                 color='k', linewidth=0.75)

                axarr[i, j].set_xticks(
                    np.arange(min(x), max(x)+1, 5.0))
                axarr[i, j].tick_params(
                    axis='both', which='major',
                    length=2, color='k', direction='in')

                axarr[i, j].yaxis.set_minor_locator(
                    AutoMinorLocator())
                axarr[i, j].tick_params(
                    axis='y', which='minor', length=0.75,
                    color='k', direction='in')

                axarr[i, j].tick_params(
                    axis='x', which='major', labelsize=7)
                axarr[i, j].tick_params(
                    axis='y', which='major', labelsize=7)

                axarr[i, j].fill_between(
                    x, cdf(x, cluster, p[idx]), zorder=0,
                    color='mediumaquamarine', alpha=0.35)

                axarr[i, j].set_title(
                    idx.replace('neg', '$^-$').replace(
                        'pos', '$^+$').replace(
                        'CD8a', 'CD8' + u'\u03B1') +
                    ', ' + 'cluster ' + str(clust_idx),
                    size=7, weight='bold')

                axarr[i, j].xaxis.grid(False)
                axarr[i, j].yaxis.grid(False)

                fig.savefig(
                    os.path.join(key_dir, col + '_cdfs' + '.pdf'))
                plt.close('all')

    # get cluster enrichments
    clss = []
    cluster_clss = []
    cdfp_clss = []

    tis = []
    cluster_tis = []
    cdfp_tis = []

    lin = []
    cluster_lin = []
    cdfp_lin = []

    ar = []
    cluster_ar = []
    cdfp_ar = []

    for key, value in cdf_dict.items():
        print(key)

        if key[5] == 'class':
            print(key)
            clss.append(key[3])
            clus = int(key[4].split('_', 2)[1])
            cluster_clss.append(clus)
            k = int(key[0].split('=', 2)[1])
            n = int(key[1].split('=', 2)[1])
            p = float(key[2].split('=', 2)[1])
            pval = 1 - binom.cdf(k, n, p)
            cdfp_clss.append(pval)

        elif key[5] == 'tissue':
            print(key)
            tis.append(key[3])
            clus = int(key[4].split('_', 2)[1])
            cluster_tis.append(clus)
            k = int(key[0].split('=', 2)[1])
            n = int(key[1].split('=', 2)[1])
            p = float(key[2].split('=', 2)[1])
            pval = 1 - binom.cdf(k, n, p)
            cdfp_tis.append(pval)

        elif key[5] == 'lineage':
            print(key)
            lin.append(key[3])
            clus = int(key[4].split('_', 2)[1])
            cluster_lin.append(clus)
            k = int(key[0].split('=', 2)[1])
            n = int(key[1].split('=', 2)[1])
            p = float(key[2].split('=', 2)[1])
            pval = 1 - binom.cdf(k, n, p)
            cdfp_lin.append(pval)

        elif key[5] == 'arm':
            print(key)
            ar.append(key[3])
            clus = int(key[4].split('_', 2)[1])
            cluster_ar.append(clus)
            k = int(key[0].split('=', 2)[1])
            n = int(key[1].split('=', 2)[1])
            p = float(key[2].split('=', 2)[1])
            pval = 1 - binom.cdf(k, n, p)
            cdfp_ar.append(pval)

    cdfp_dict = {}
    for name, u in zip(
        ['class', 'tissue', 'lineage', 'arm'], [
            [clss, cluster_clss, cdfp_clss],
            [tis, cluster_tis, cdfp_tis],
            [lin, cluster_lin, cdfp_lin],
            [ar, cluster_ar, cdfp_ar]]):

        cdfp_frame = pd.DataFrame(u)
        cdfp_frame = cdfp_frame.T
        cdfp_frame.rename(
            columns={0: 'classifier', 1: 'cluster', 2: 'pval'},
            inplace=True)

        cdfp_frame_threshold = cdfp_frame[
            cdfp_frame['pval'] <= 0.05].sort_values(by='pval')

        cdfp_dict[name] = cdfp_frame_threshold

    os.chdir(key_dir)
    po_cdfp_dict = open(
        'cdfp_dict.pickle', 'wb')
    pickle.dump(cdfp_dict, po_cdfp_dict)
    po_cdfp_dict.close()

    return cdfp_dict

ex_naive_cdfp = cdf('experimental_data', 'naive_unfiltered')
ex_gl261_cdfp = cdf('experimental_data', 'gl261_unfiltered')
