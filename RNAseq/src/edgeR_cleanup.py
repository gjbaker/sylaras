import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# specify path to edgeR differential gene expression analysis results
edge_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/data/edgeR_results'

# specify path to mouse .gtf file (to convert gene ids to symbols)
map_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/transcript_to_gene_mapping'

# specify save path to for .rnk file (for downstream GSEA)
sav_rnk = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/rnk_files'

# import mouse .gtf file
gene_set = read_gtf(os.path.join(map_path, 'Mus_musculus.GRCm38.94.gtf'))

# generate an id-to-symbol dictionary
mapping = dict(zip(gene_set['gene_id'], gene_set['gene_name']))

# import edgeR results
er = pd.read_csv(
    os.path.join(
        edge_path, 'meta_b220_min2and7-Treatment.tsv'), sep='\t')

# create symbols column
er['Symbol'] = er['Gene'].map(mapping)

# filter edgeR results by FDR
sig_er = er[er['FDR'] <= 0.05].sort_values(by='logFC')

# save differentially expressed genes
sig_er.to_csv(
    os.path.join(edge_path, 'edgeR_sig_min2and7.csv'),
    index=False, header=True, sep=',')

# sort all genes by fold-change
all_genes = er.sort_values(by='logFC')

# slice out gene symbols and fold-change columns
all_genes = all_genes[['Symbol', 'logFC']]

# drop any symbols set to Nan because they did not appear in the
# Mus_musculus.GRCm38.94.gtf file
all_genes = all_genes.dropna()

# average over log2 fold-change for each gene id
all_genes = all_genes.groupby(
    by='Symbol').mean().sort_values(by='logFC', ascending=True).reset_index()

# save the .rnk file for all genes from the edgeR analysis
all_genes.to_csv(
    os.path.join(sav_rnk, 'edgeR_allgenes_min2and7.rnk'),
    index=False, header=False, sep='\t')
