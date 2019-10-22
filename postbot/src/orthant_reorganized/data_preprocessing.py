from invoke_init import *
from constants_syl import *
from helpers import *

# -----------------------------------------------------------------------------
def extract_TOTAL():
    banner('RUNNING MODULE: extract_TOTAL')
    raw_data_dir = os.path.join(ORTHANT_DIR, 'raw')
    os.makedirs(raw_data_dir)
    projectdir = '/Volumes/SysBio/SORGER PROJECTS/gbmi'
    TOTAL_rootdir = projectdir + '/data/input/phenobot/1bhtAe0/logicle'
    for TOTAL_dirpath, TOTAL_dirnames, TOTAL_filenames in os.walk(
      TOTAL_rootdir):
        if '/00/' not in TOTAL_dirpath:  # avoid baseline data
            for file in TOTAL_filenames:
                if file.endswith('data.tsv'):
                    print('Extracting_TOTAL_raw_data_file_' +
                          TOTAL_dirpath + '.')
                    TOTAL_path_list = TOTAL_dirpath.split(os.sep)
                    df9 = pd.DataFrame(TOTAL_path_list)
                    df10 = df9.T
                    df10.columns = ['', 'Volumes', 'SysBio', 'SORGER PROJECTS',
                                    'gbmi', 'data', 'input', 'phenobot',
                                    '1bhtAe0', 'logicle', 'time_point',
                                    'tissue', 'status', 'replicate']
                    df11 = df10.drop(['', 'Volumes', 'SysBio',
                                      'SORGER PROJECTS', 'gbmi',
                                      'data', 'input', 'phenobot', '1bhtAe0',
                                      'logicle'], axis=1)
                    TOTAL_cols = df11.columns.tolist()
                    df12 = df11[TOTAL_cols].iloc[0]
                    os.chdir(TOTAL_dirpath)
                    TOTAL_data = pd.read_csv('data.tsv', sep='\t')
                    for ci, c in enumerate(TOTAL_cols):
                        TOTAL_data.insert(ci, c, df12[c])
                    TOTAL_filename = df12.values.tolist()
                    TOTAL_data.to_csv(os.path.join(
                        raw_data_dir, '%s.csv' %
                        (TOTAL_filename)), index=False)
    print()
    os.chdir(PICKLE_DIR)
    po_raw_data_dir = open(INITIAL_PICKLE_FILE, 'wb')
    pickle.dump(raw_data_dir, po_raw_data_dir)
    po_raw_data_dir.close()


# combine individual FULL .tsv files into one aggregate dataframe
def combine_TOTAL():
    banner('RUNNING MODULE: combine_TOTAL')
    os.chdir(PICKLE_DIR)
    pi_raw_data_dir = open(INITIAL_PICKLE_FILE, 'rb')
    raw_data_dir = pickle.load(pi_raw_data_dir)

    aggregate_data_dir = os.path.join(ORTHANT_DIR, 'aggregate_datasets')
    os.makedirs(aggregate_data_dir)

    os.chdir(raw_data_dir)
    TOTAL_filenames = glob.glob(raw_data_dir + '/*.csv')

    TOTAL_dfs1 = []
    for filename in TOTAL_filenames:
        print('Adding_' + filename + ' to TOTAL DataFrame.')
        TOTAL_dfs1.append(pd.read_csv(filename))
    del TOTAL_filenames
    print()
    print('Aggregating TOTAL dataset.')
    TOTAL_data = pd.concat(TOTAL_dfs1, ignore_index=True)
    del TOTAL_dfs1

    # merge 31, 35, and 36DPI data to 30DPI in TOTAL data
    TOTAL_data.loc[(TOTAL_data['time_point'] == 31) &
                   (TOTAL_data['replicate'] == 1), 'replicate'] = 2
    TOTAL_data.loc[(TOTAL_data['time_point'] == 31), 'time_point'] = 30

    TOTAL_data.loc[(TOTAL_data['time_point'] == 35) &
                   (TOTAL_data['replicate'] == 1), 'replicate'] = 4
    TOTAL_data.loc[(TOTAL_data['time_point'] == 35) &
                   (TOTAL_data['replicate'] == 2), 'replicate'] = 5
    TOTAL_data.loc[(TOTAL_data['time_point'] == 35), 'time_point'] = 30

    TOTAL_data.loc[(TOTAL_data['time_point'] == 36) &
                   (TOTAL_data['replicate'] == 1), 'replicate'] = 7
    TOTAL_data.loc[(TOTAL_data['time_point'] == 36) &
                   (TOTAL_data['replicate'] == 2), 'replicate'] = 8
    TOTAL_data.loc[(TOTAL_data['time_point'] == 36), 'time_point'] = 30

    TOTAL_data['time_point'].replace(to_replace=23, value=30, inplace=True)

    channel_list = TOTAL_data.columns.tolist()[7:18]
    channel_list.sort()

    TOTAL_data.to_csv(os.path.join(
        aggregate_data_dir, TOTAL_DATA), index=False)

    os.chdir(PICKLE_DIR)
    po_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'wb')
    pickle.dump(aggregate_data_dir, po_aggregate_data_dir)
    po_aggregate_data_dir.close()

    po_channel_list = open(PO_CHANNEL_LIST, 'wb')
    pickle.dump(channel_list, po_channel_list)
    po_channel_list.close()
    print()

# take a random subset of TOTAL_data
def random_subset():

    banner('RUNNING MODULE: random_subset')

    os.chdir(PICKLE_DIR)
    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    os.chdir(aggregate_data_dir)
    print('Reading TOTAL_data.')
    TOTAL_data = pd.read_csv(TOTAL_DATA)

    weight_dict = {'blood': 2.173700e-06,
                   'marrow': 4.850331e-07,
                   'nodes': 3.770597e-07,
                   'spleen': 5.381480e-07,
                   'thymus': 3.894210e-07}

    weights = pd.Series([weight_dict[i] for i in TOTAL_data['tissue']])

    print()
    print('Subsampling TOTAL_data.')
    TOTAL_sample = TOTAL_data.sample(n=10000000, replace=False,
                                     weights=weights, random_state=1, axis=0)

    TOTAL_sample.reset_index(drop=True, inplace=True)

    channel_list = TOTAL_sample.columns.tolist()[6:18]
    channel_list.sort()

    TOTAL_sample['cluster'] = np.random.randint(
        42, size=len(TOTAL_sample))
    TOTAL_sample['row'] = TOTAL_sample.index.tolist()
    TOTAL_sample['index'] = TOTAL_sample.index.tolist()

    TOTAL_cols = ['time_point', 'tissue', 'status', 'replicate',
                  'cluster', 'fsc', 'ssc', 'ly6c', 'cd3e', 'cd11c',
                  'cd45', 'cd11b', 'cd4', 'ly6g', 'f480', 'cd49b',
                  'cd8', 'b220', 'row', 'index']

    TOTAL_sample = TOTAL_sample[TOTAL_cols]

    TOTAL_sample.to_csv(os.path.join(
        aggregate_data_dir, TOTAL_SAMPLE), index=False)

    os.chdir(PICKLE_DIR)
    po_channel_list = open(PO_CHANNEL_LIST, 'wb')
    pickle.dump(channel_list, po_channel_list, 2)
    po_channel_list.close()
    print()