from invoke_init import *
from constants_syl import *
from helpers import *


# map celltype to lineage
def lineage_classification():

    banner('RUNNING MODULE: lineage_classification')

    os.chdir(PICKLE_DIR)

    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    lin1 = classified_choice[channel_list_update + ['cell_type']]

    lin2 = lin1.groupby(
        channel_list_update).size().sort_values(ascending=False)
    lin2 = pd.DataFrame(lin2).rename(columns={0: 'count'})

    lin3 = lin1.join(lin2, on=channel_list_update, how='left')

    lineage_frame = lin3.drop_duplicates().sort_values(
        by='count', ascending=False).reset_index(drop=True)

    lineage_dict = {'Mono': 'myeloid', 'Eo': 'myeloid', 'Mac': 'myeloid',
                    'PMN': 'myeloid', 'DC': 'myeloid', 'NK': 'lymphoid',
                    'T': 'lymphoid', 'CD4T': 'lymphoid', 'CD8T': 'lymphoid',
                    'DPT': 'lymphoid', 'DNT': 'lymphoid',
                    'LTi': 'lymphoid', 'B': 'lymphoid',
                    'Precursor': 'other', 'unspecified': 'other'}

    lineage_dict_regex = {'^.*' + k + '$': v for k, v in lineage_dict.items()}
    lineage_frame['lineage'] = lineage_frame['cell_type'].replace(
        lineage_dict_regex, regex=True)

    lineage_dict_regex_abr = {k: 'Y' for k, v in lineage_dict.items()
                              if k not in ['unspecified', 'other']}

    lineage_frame['landmark'] = lineage_frame['cell_type'] \
        .replace(lineage_dict_regex_abr)
    lineage_frame['landmark'] = lineage_frame['landmark'].replace(
        list(set(lineage_frame['cell_type'])), 'N')

    vector_classification = {}
    landmark_pops = []
    for index, row in lineage_frame.iterrows():
        if row['cell_type'] != 'unspecified':
            vector_classification[row['cell_type']] = {}
            vector_classification[row['cell_type']]['lineage'] = row['lineage']
            vector_classification[row['cell_type']]['signature'] = []
            if row['landmark'] == 'Y':
                landmark_pops.append(row['cell_type'])
            for i, num in enumerate(row[:-4]):
                if num != 0:
                    vector_classification[row['cell_type']]['signature'] \
                        .append(list(lineage_frame)[:-4][i])
    for key, value in vector_classification.items():
        print(key, value)
    print()

    os.chdir(PICKLE_DIR)
    po_vector_classification = open(PO_VECTOR_CLASSIFICATION, 'wb')
    pickle.dump(vector_classification, po_vector_classification)
    po_vector_classification.close()

    po_landmark_pops = open(PO_LANDMARK_POPS, 'wb')
    pickle.dump(landmark_pops, po_landmark_pops)
    po_landmark_pops.close()
