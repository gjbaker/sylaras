import os
import pandas as pd
from gtfparse import read_gtf

# display adjustments
pd.set_option('display.width', None)
pd.options.display.max_rows = 150
pd.options.display.max_columns = 33

# specify path to salmon counts
feature_counts_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/data/feature_counts'

salmon_counts_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/data/salmon_counts'

# specify path to mouse .gtf file
map_path = '/Users/gjbaker/projects/gbm_immunosuppression/' + \
    'RNAseq/gsea/transcript_to_gene_mapping'

# import salmon counts
feature = pd.read_csv(
    os.path.join(
        feature_counts_path, 'combined.counts'), sep='\t')

# import salmon counts
salmon = pd.read_csv(
    os.path.join(
        salmon_counts_path, 'combined.sf'), sep='\t')

# import mouse .gtf file
gene_set = read_gtf(os.path.join(map_path, 'Mus_musculus.GRCm38.94.gtf'))

# generate an id-to-symbol dictionary for gene ids
feature_mapping = dict(zip(gene_set['gene_id'], gene_set['gene_name']))

# generate an id-to-symbol dictionary for transcript ids
salmon_mapping = dict(zip(gene_set['transcript_id'], gene_set['gene_name']))

# append gene symbol column
feature['Symbol'] = pd.Series(feature.id).map(feature_mapping).tolist()

# append gene symbol column
salmon['Symbol'] = pd.Series(salmon.Name).map(salmon_mapping).tolist()
