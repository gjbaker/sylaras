import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gtfparse import read_gtf

# display adjustments
pd.set_option('display.width', None)
pd.options.display.max_rows = 150
pd.options.display.max_columns = 33

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

# specify path to salmon counts
counts_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/data/featureCounts'

# specify path to mouse .gtf file
map_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/transcript_to_gene_mapping'

# specify save path to for .rnk file (for downstream GSEA)
sav_rnk = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/rnk_files'

# import salmon counts
wide = pd.read_csv(
    os.path.join(
        counts_path, 'combined.counts'), sep='\t')

# set index to gene id
wide.set_index('id', inplace=True)

# import mouse .gtf file
gene_set = read_gtf(os.path.join(map_path, 'Mus_musculus.GRCm38.94.gtf'))

# generate an id-to-symbol dictionary
mapping = dict(zip(gene_set['gene_id'], gene_set['gene_name']))

# normalize to read counts
norm = wide.loc[
        :, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']].div(wide.loc[
            :, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']].sum(axis=0))

# transpose DataFrame so that rows are samples and genes are columns
transposed = norm.T

# generate the PCA model
pca = PCA(n_components=min(transposed.shape[0], transposed.shape[1]))

# standardize the feature values (z-score normalization: mean=0, var=1)
X = StandardScaler().fit_transform(transposed)

# fit the standardized data to the PCA model
# transform standardized data into sample component scores
# put into a DataFrame
scores = pd.DataFrame(
    data=pca.fit_transform(X),
    index=transposed.index,
    columns=['PC_' + str(i+1) for i in range(pca.n_components)])

# plot explained variance as a scree plot
sns.set_style('whitegrid')
fig = plt.figure(figsize=(6, 4))
ax = fig.subplots()
plt.bar(
    range(1, len(pca.explained_variance_ratio_)+1),
    pca.explained_variance_ratio_,
    alpha=0.5, align='center',
    label='individual explained variance')

cum_exp_var = np.cumsum(pca.explained_variance_ratio_)

plt.step(
    range(1, len(pca.explained_variance_ratio_)+1),
    cum_exp_var, where='mid', label='cumulative explained variance')

plt.ylabel('explained variance ratio')
plt.xlabel('principal component')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(counts_path, 'featureCounts_scree_plot' + '.pdf'))
plt.close('all')

# plot component scores
sample_dict = {
    'S1': ('b', 's'),
    'S2': ('r', 's'),
    'S3': ('b', '^'),
    'S4': ('r', '^'),
    'S5': ('b', 'o'),
    'S6': ('r', 'o')}

for p, i in zip([('PC_1', 'PC_2'), ('PC_3', 'PC_4')], [(0, 1), (2, 3)]):

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots()

    for k, v in sample_dict.items():
        indicesToKeep = scores.index == k
        plt.scatter(
            scores.loc[indicesToKeep, p[0]],
            scores.loc[indicesToKeep, p[1]],
            c=v[0], s=150, marker=v[1])

    ax.set_xlabel(
        p[0] + '(' + str(
            round(pca.explained_variance_ratio_[i[0]]*100, 2)) + '%)',
        fontsize=15)
    ax.set_ylabel(
        p[1] + '(' + str(
            round(pca.explained_variance_ratio_[i[1]]*100, 2)) + '%)',
        fontsize=15)

    ax.set_title('score plot', fontsize=20)
    ax.legend(transposed.index)
    plt.savefig(
        os.path.join(
            counts_path, 'featureCounts_score_plot_' +
            p[0] + 'v' + p[1] + '.pdf'))
    plt.close('all')

    # plot loadings for first 2 PCs

    # components_ = the eigenvectors, shape:(n_components, n_features)

    # explained_variance_ = eigenvalues, shape:(n_components,)

    # loadings = components_.T * sqrt(explained_variance_)

    # explained_variance_ = np.sum(loadings**2, axis=0)

    # calculate loadings, shape:(n_features, n_components)
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_),
        index=transposed.columns,
        columns=scores.columns)

    pc1and2_loadings = loadings.loc[:, [p[0], p[1]]]

    pc1_sort = pc1and2_loadings.sort_values(by=p[0])
    pc1_filt = pc1_sort.head(100).append(pc1_sort.tail(100))

    pc2_sort = pc1and2_loadings.sort_values(by=p[1])
    pc2_filt = pc2_sort.head(100).append(pc2_sort.tail(100))

    total_filt = pc1_filt.append(pc2_filt)

    sns.set(style='whitegrid')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots()

    for i in total_filt.iterrows():

        gene = i[0]

        pc_1_val = i[1][0]
        pc_2_val = i[1][1]

        plt.scatter(pc_1_val, pc_2_val,
                    marker='o', s=8.0, c='k', alpha=0.5)

        try:
            symbol = mapping[gene]

        except KeyError:
            symbol = np.nan

        plt.annotate(
            symbol, xy=(pc_1_val, pc_2_val),
            xytext=(2, 2), size=8, textcoords='offset points',
            ha='left', va='bottom', weight='bold')

    plt.xlabel(p[0], fontweight='normal', fontsize=15)
    plt.ylabel(p[1], fontweight='normal', fontsize=15)

    plt.axhline(0, ls='dashed', color='gray', linewidth=0.75)
    plt.axvline(0, ls='dashed', color='gray', linewidth=0.75)

    plt.savefig(os.path.join(
        counts_path, 'featureCounts_loadings_plot_' +
        p[0] + 'v' + p[1] + '.pdf'))
    plt.close('all')

# get condition-specific means for each count
cd8 = norm.loc[:, ['S1', 'S3', 'S5']].mean(axis=1) + 0.5
b220 = norm.loc[:, ['S2', 'S4', 'S6']].mean(axis=1) + 0.5

# calculate log2 fold-change column
norm['logFC'] = np.log2(b220/cd8)

# append gene symbol column
norm['Symbol'] = pd.Series(norm.index).map(mapping).tolist()

# sort transcripts by log2 fold-change
norm.sort_values(by='logFC', inplace=True)

# get .rnk file for GSEA on all genes
# slice out gene symbols and fold-change columns
all_genes = norm[['Symbol', 'logFC']]

# average over log2 fold-change in mean reads for each gene id
all_genes = all_genes.groupby(
    by='Symbol').mean().sort_values(by='logFC', ascending=True).reset_index()

# save the .rnk file for all genes from the salmon analysis
all_genes.to_csv(
    os.path.join(sav_rnk, 'featureCounts_allgenes.rnk'),
    index=False, header=False, sep='\t')

# get .rnk file for GSEA on loadings
tag = '# I will be ignored'

PC2_loadings = pd.DataFrame(
    loadings.PC_2.sort_values(ascending=False)).reset_index(
        drop=False)

# ensure no duplicate rankings
if PC2_loadings.duplicated().any() is True:
    print('Some rankings are identical.')

else:
    PC2_loadings.to_csv(
        os.path.join(sav_rnk, 'PC2_loadings_salmon.rnk'),
        index=False, header=False, sep='\t')

# specify path to GSEA output directory
gsea_output_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/data/gsea_output'

if not os.path.exists(gsea_output_path):
    os.mkdir(gsea_output_path)

##############
# get .gct files for GSEA
# tag = '#1.2'

# # choose genes with the top <depth> loadings
# depth = 10000

# genes = loadings.PC_1.sort_values(
#     ascending=False).head(depth).index.tolist() + \
#     loadings.PC_1.sort_values(ascending=False).tail(depth).index.tolist()
# n_genes = len(genes)
# descriptors = ['na'] * n_genes

# pos_loads = loadings.PC_1[genes][loadings.PC_1[genes] > 0].rename(
#     loadings.PC_1.name + '+')
# neg_loads = loadings.PC_1[genes][loadings.PC_1[genes] <= 0].rename(
#     loadings.PC_1.name + '-')

# combo_loads = pd.concat(
#     [pos_loads, neg_loads], axis=1).replace(to_replace=np.nan, value='')

# frame1 = pd.DataFrame(
#     data='', index=range(0, n_genes+3), columns=range(0, 4))

# frame1.iloc[0, 0] = tag
# frame1.iloc[1, 0] = n_genes
# frame1.iloc[1, 1] = len(combo_loads.columns)
# frame1.iloc[2:3, 0:1] = 'NAME'
# frame1.iloc[3:len(genes)+3, 0] = genes
# frame1.iloc[2:3, 1:2] = 'Description'
# frame1.iloc[3:len(descriptors)+3, 1:2] = 'na'
# frame1.iloc[2:3, 2] = combo_loads.columns[0]
# frame1.iloc[3:, 2] = combo_loads['PC_1+'].values
# frame1.iloc[2:3, 3] = combo_loads.columns[1]
# frame1.iloc[3:, 3] = combo_loads['PC_1-'].values
# frame1.to_csv('gsea_input.gct', sep='\t', index=False, header=False)

##############
# get .cls files for GSEA on loadings
# frame2 = pd.DataFrame(data='', index=range(0, 3), columns=range(0, 3))
# frame2.iloc[0, 0] = len(combo_loads.columns)
# n_classes = 2
# frame2.iloc[0, 1] = n_classes
# frame2.iloc[0, 2] = 1
# frame2.iloc[1, 0] = '#'
# frame2.iloc[1, 1] = combo_loads.columns[0]
# frame2.iloc[1, 2] = combo_loads.columns[1]
# frame2.iloc[2, 0] = 0
# frame2.iloc[2, 1] = 1
# frame2.to_csv('my_classes.cls', sep='\t', index=False, header=False)
