from invoke_init import *
from constants_syl import *
from helpers import *


# -----------------------------------------------------------------------------
# VECTORIZATION


# get Boolean representations of FULL, kernel, jitter, and kernel_jitter
# datasets into a Python dictionary
def data_discretization():

    banner('RUNNING MODULE: data_discretization')

    os.chdir(PICKLE_DIR)
    final_frames_dict_shlf = shelve.open(FINAL_FRAMES_DICT)

    pi_channel_list = open(PO_CHANNEL_LIST, 'rb')
    channel_list = pickle.load(pi_channel_list)

    TOTAL_unique_vector_dir = os.path.join(ORTHANT_DIR, 'unique_vectors')
    os.makedirs(TOTAL_unique_vector_dir)

    Boo_frames_dict = {}
    for name, frame in final_frames_dict_shlf.items():
        channel_columns = {}
        for channel in frame.iloc[:, 7:18]:
            channel_columns[channel] = frame.loc[:, channel].values
        print()
        for key, value in channel_columns.items():
            print('Converting ' + key + ' protein expression data into its '
                  'Boolean represenation in the ' + name + ' DataFrame.')
            for i, v in enumerate(value):
                if v > 0:
                    value[i] = 1
                elif v <= 0:
                    value[i] = 0
        Boo_data = frame.iloc[:, 7:18].astype(int)
        channel_list.sort()
        Boo_data = Boo_data[channel_list]
        channel_list_update_dict = {
            'b220': 'B220', 'cd45': 'CD45', 'cd11b': 'CD11b', 'cd11c': 'CD11c',
            'cd3e': 'CD3e', 'cd4': 'CD4', 'cd49b': 'CD49b', 'cd8': 'CD8a',
            'f480': 'F480', 'ly6c': 'Ly6C', 'ly6g': 'Ly6G'}
        Boo_data1 = Boo_data.rename(columns=channel_list_update_dict)
        Boo_data2 = pd.concat([frame.iloc[:, 0:7], Boo_data1,
                              frame.iloc[:, 18:20]], axis=1)

        # correct for CD49b in blood
        NK = Boo_data2['CD49b'][
            ((Boo_data2['B220'] == 0) & (Boo_data2['CD11b'] == 1) &
             (Boo_data2['CD11c'] == 0) & (Boo_data2['CD3e'] == 0) &
             (Boo_data2['CD4'] == 0) & (Boo_data2['CD45'] == 1) &
             (Boo_data2['CD49b'] == 1) & (Boo_data2['CD8a'] == 0) &
             (Boo_data2['F480'] == 0) & (Boo_data2['Ly6C'] == 0) &
             (Boo_data2['Ly6G'] == 0))]

        non_NK = Boo_data2['CD49b'][
            ~((Boo_data2['B220'] == 0) & (Boo_data2['CD11b'] == 1) &
              (Boo_data2['CD11c'] == 0) & (Boo_data2['CD3e'] == 0) &
              (Boo_data2['CD4'] == 0) & (Boo_data2['CD45'] == 1) &
              (Boo_data2['CD49b'] == 1) & (Boo_data2['CD8a'] == 0) &
              (Boo_data2['F480'] == 0) & (Boo_data2['Ly6C'] == 0) &
              (Boo_data2['Ly6G'] == 0))]

        non_NK[:] = 0

        new_cd49b_col = non_NK.append(NK).sort_index()

        del NK
        del non_NK

        Boo_data2['CD49b'] = new_cd49b_col

        Boo_frames_dict[name] = Boo_data2

        channel_list_update = list(channel_list_update_dict.values())
        channel_list_update.sort()
        unique_vectors = Boo_data2.drop_duplicates(channel_list_update)

        g = sns.heatmap(unique_vectors.loc[:, channel_list_update])
        for item in g.get_yticklabels():
            item.set_rotation(90)
        plt.savefig(os.path.join(TOTAL_unique_vector_dir, name +
                    '_unique_vectors' + '.pdf'))
        plt.close('all')
    print()

    os.chdir(PICKLE_DIR)
    Boo_frames_dict_shlf = shelve.open(BOO_FRAMES_DICT_SHLF)
    Boo_frames_dict_shlf.update(Boo_frames_dict)
    Boo_frames_dict_shlf.close()

    po_channel_list_update_dict = open(PO_CHANNEL_LIST_UPDATE_DICT, 'wb')
    pickle.dump(channel_list_update_dict, po_channel_list_update_dict)
    po_channel_list_update_dict.close()

    po_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'wb')
    pickle.dump(channel_list_update, po_channel_list_update)
    po_channel_list_update.close()