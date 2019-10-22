import os


#PATHS

PROJECT_PATH = '/Users/gjbaker/projects/gbm_immunosuppression'
ORTHANT_DIR = os.path.join(PROJECT_PATH, 'postbot', 'data', 'logicle_e20', 'orthant')
PICKLE_DIR= os.path.join(ORTHANT_DIR, 'pickled_global_vars')


#PICKLE FILES

INITIAL_PICKLE_FILE = 'raw_data_dir.pickle'
PO_AGGREGATE_DATA_DIR = 'aggregate_data_dir.pickle'
PO_CHANNEL_LIST = 'channel_list.pickle'
PO_CHANNEL_LIST_UPDATE_DICT = 'channel_list_update_dict.pickle'
PO_CHANNEL_LIST_UPDATE = 'channel_list_update.pickle'
PO_TOTAL_VECTORS = 'total_vectors.pickle'
PO_CLASSIFIED_DIR = 'classified_dir.pickle'
PO_AGGREGATE_CHOICE = 'aggregate_choice.pickle'
PO_CLASSIFIED_CHOICE = 'classified_choice.pickle'
PO_AVE = 'ave.pickle'
PO_AVE_VEC = 'ave_vec.pickle'
PO_TOTAL_SAVE_PLOT = 'TOTAL_save_plot.pickle'
PO_CLASSIFIED_CHOICE_COPY = 'classified_choice_copy.pickle'
PO_TOTAL_SIG_DIR = 'TOTAL_sig_dir.pickle'
PO_TOTAL_SIG_DIF_ALL = 'TOTAL_sig_dif_all.pickle'
PO_TOTAL_SIG_DIF = 'TOTAL_sig_dif.pickle'
PO_SIG_DIF_FDRCORRECTED = 'sig_dif_FDRcorrected.pickle'
PO_CHANNEL_LIST_CD8A = 'channel_list_cd8a.pickle'
PO_TOTAL_SIG_DIF_ALL_MAG = 'TOTAL_sig_dif_all_mag.pickle'
PO_SIG_DIF_FDRCORRECTED_RATIO = 'sig_dif_FDRcorrected_ratio.pickle'
PO_VECTOR_CLASSIFICATION = 'vector_classification.pickle'
PO_LANDMARK_POPS = 'landmark_pops.pickle'
PO_PAIR_GRID_INPUT = 'pair_grid_input.pickle'
PO_COLOR_DICT = 'color_dict.pickle'
PO_OV = 'ov.pickle'


#SHELVES

FINAL_FRAMES_DICT = 'final_frames_dict.shelve'
BOO_FRAMES_DICT_SHLF = 'Boo_frames_dict.shelve'
DASHBOARD_SHLF = 'dashboard.shelve'


#CSV FILES

TOTAL_DATA = 'TOTAL_data.csv'
TOTAL_SAMPLE = 'TOTAL_sample.csv'
DUPE_REPORT_CSV = 'duplicate_report.csv'
OVERALL_CSV = 'overall.csv'
TOTAL_SIG_DIF_CSV = 'classifier_sig_dif.csv'
SIG_DIF_FDRCORRECTED_CSV = 'FDR_corrected_classifier_sig_dif.csv'
TOTAL_SIG_DIF_VEC_CSV = 'vector_sig_dif.csv'






