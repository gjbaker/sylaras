# CONFIGURATION
from invoke_init import *

# SYLARAS SPECIFIC IMPORTS
from constants_syl import *
from helpers import *
from data_preprocessing import *
from gate_bias import *
from boolean_classifier import *
from data_disretization import *
from downstream_analysis import *
from gate_bias import *
from lineage_classification import *

# project path for script call
project_path = PROJECT_PATH

# project path error message
if not os.path.exists(project_path):
    print('Project path does not exist')
    sys.exit()

# script run timer
startTime = datetime.now()

# create a directory for pickled objects
os.makedirs(ORTHANT_DIR)
os.makedirs(PICKLE_DIR)


extract_TOTAL()
combine_TOTAL()
random_subset()
gate_bias()
data_discretization()
Boolean_classifier()

# -----------------------------------------------------------------------------
# BEGIN DOWNSTREAM ANALYSIS

TOTAL_and_classified_choices()
split_combine_celltypes()
alpha_vectors = vector_coverage()
dashboard_dict()
overall()
check_celltypes('B220posCD8T', 'CD8T', 'CD4T', 'B', 'b220', 'cd8a')
celltype_barcharts()
celltype_stats()
vector_barcharts()
vector_stats()
replicate_counts()
celltype_pval_mag()
celltype_ratio_mag_split()
celltype_box_vector()
celltype_box_scatter()
celltype_box_channel()
scatter_box_channel()
celltype_piecharts()
TOTAL_mag_heatmap()
TOTAL_ratio_heatmap()
celltype_heatmap()
vector_accumulation()
lineage_classification()
priority_scores()
aggregate_celltype_fig()

print('Postbot analysis completed in ' + str(datetime.now() - startTime))
