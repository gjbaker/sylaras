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
salmon_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/data/salmon_counts'

# specify path to mouse .gtf file
map_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/transcript_to_gene_mapping'

# specify save path to for .rnk file (for downstream GSEA)
sav_rnk = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/rnk_files'

if not os.path.exists(sav_rnk):
    os.mkdir(sav_rnk)

# import salmon counts
long = pd.read_csv(
    os.path.join(
        salmon_path, 'combined.sf'), sep='\t')

# import mouse .gtf file
gene_set = read_gtf(os.path.join(map_path, 'Mus_musculus.GRCm38.94.gtf'))

# generate an id-to-symbol dictionary
mapping = dict(zip(gene_set['transcript_id'], gene_set['gene_name']))

# convert long-table to wide-table on read count
wide = long.pivot(index='Name', columns='Sample', values='TPM')

# normalize to read counts
# norm = wide.loc[
#         :, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']].div(wide.loc[
#             :, ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']].sum(axis=0))

# transpose DataFrame so that rows are samples and genes are columns
transposed = wide.T

# generate the PCA model
pca = PCA(n_components=min(transposed.shape[0], transposed.shape[1]))

# standardize the feature values (z-score widealization: mean=0, var=1)
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
plt.savefig(os.path.join(salmon_path, 'salmon_scree_plot' + '.pdf'))
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
            salmon_path, 'salmon_score_plot_' + p[0] + 'v' + p[1] + '.pdf'))
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

        transcript = i[0]

        pc_1_val = i[1][0]
        pc_2_val = i[1][1]

        plt.scatter(pc_1_val, pc_2_val,
                    marker='o', s=8.0, c='k', alpha=0.5)

        try:
            symbol = mapping[transcript]

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
        salmon_path, 'salmon_loadings_plot_' + p[0] + 'v' + p[1] + '.pdf'))
    plt.close('all')

# get condition-specific means for each TPM
cd8 = wide.loc[:, ['S1', 'S3', 'S5']].mean(axis=1) + 0.5
b220 = wide.loc[:, ['S2', 'S4', 'S6']].mean(axis=1) + 0.5

# calculate log2 fold-change in mean TPM
wide['logFC'] = np.log2(b220/cd8)

# append gene symbol column
wide['Symbol'] = pd.Series(wide.index).map(mapping).tolist()

# sort transcripts by log2 fold-change
wide.sort_values(by='logFC', inplace=True, ascending=True)

# prep cd8 TPM means
cd8 = pd.DataFrame(cd8)
cd8['Symbol'] = pd.Series(cd8.index).map(mapping).tolist()
cd8.rename(columns={0: 'mean'}, inplace=True)
cd8.dropna(inplace=True)
cd8.reset_index(inplace=True)

# prep b220 TPM means
b220 = pd.DataFrame(b220)
b220['Symbol'] = pd.Series(b220.index).map(mapping).tolist()
b220.rename(columns={0: 'mean'}, inplace=True)
b220.dropna(inplace=True)
b220.reset_index(inplace=True)

# get H2 genes
if cd8.index.equals(b220.index) is True:
    print()
    print('cd8 and b220 indices ARE identical, getting all H2 genes.')
    H2_genes = b220['Symbol'][b220['Symbol'].str.startswith('H2-')].tolist()
else:
    print('cd8 and b220 indices ARE NOT identical.')

# filter out H2-Q5, very high transcript number
# indices = [i for i, x in enumerate(H2_genes) if x == "H2-Q5"]
# H2_genes = [i for j, i in enumerate(H2_genes) if j not in indices]

# plot cd8 vs. b220 mean TPMs for all H2 genes
sns.set(style='whitegrid')
fig = plt.figure(figsize=(5, 5))
ax = fig.subplots()

ax.scatter(
    cd8['mean'][cd8['Symbol'].isin(H2_genes)],
    b220['mean'][b220['Symbol'].isin(H2_genes)],
    s=2.0, c='k')

for txt in b220[b220['Symbol'].isin(H2_genes)].iterrows():

    ax.annotate(
        txt[1][2], xy=(cd8['mean'][cd8['Symbol'].isin(H2_genes)][txt[0]],
                       b220['mean'][b220['Symbol'].isin(H2_genes)][txt[0]]),
        xytext=(2, 2), size=8, textcoords='offset points',
        ha='left', va='bottom', weight='bold')

plt.xlabel('mean TPM cd8', fontweight='normal', fontsize=15)
plt.ylabel('mean TPM b220', fontweight='normal', fontsize=15)

plt.savefig(os.path.join(
    salmon_path, 'salmon_H2_genes.pdf'))
plt.close('all')

# get .rnk file for GSEA on all transcripts
# slice out gene symbols and fold-change columns
all_genes = wide[['Symbol', 'logFC']]

# average over log2 fold-change in mean TPMs for each gene transcript
all_genes = all_genes.groupby(
    by='Symbol').mean().sort_values(by='logFC', ascending=True).reset_index()

# drop any symbols set to Nan because they did not appear in the
# Mus_musculus.GRCm38.93.gtf file
# all_genes = all_genes.dropna()

# save the .rnk file for all genes from the salmon analysis
all_genes.to_csv(
    os.path.join(sav_rnk, 'salmon_allgenes.rnk'),
    index=False, header=False, sep='\t')

# get .rnk file for GSEA on loadings
tag = '# I will be ignored'

PC2_loadings = pd.DataFrame(
    loadings.PC_2.sort_values(ascending=False)).reset_index(
        drop=False)

# append gene symbol column
PC2_loadings['Symbol'] = pd.Series(PC2_loadings['Name']).map(mapping).tolist()

# average over PC loadings values for each gene transcript
PC2_loadings = PC2_loadings.groupby(
    by='Symbol').mean().sort_values(by='PC_2', ascending=True).reset_index()

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
