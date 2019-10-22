from invoke_init import *
from constants_syl import *
from helpers import *


# -----------------------------------------------------------------------------
# BOOLEAN CLASSIFIER

# classify Boolean vectors as celltypes
# vector = orthant without a biologically meaningful name
# celltype = orthant assigned a biologically meaningful name
def Boolean_classifier():

    banner('RUNNING MODULE: Boolean_classifier')

    os.chdir(PICKLE_DIR)
    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    Boo_frames_dict_shlf = shelve.open(BOO_FRAMES_DICT_SHLF)

    # define the input variable names
    B220, CD3e, CD4, CD8a, Ly6G, Ly6C, F480, CD11b, CD11c, CD49b, CD45 = map(
        exprvar, ['B220', 'CD3e', 'CD4', 'CD8a', 'Ly6G', 'Ly6C', 'F480',
                  'CD11b', 'CD11c', 'CD49b', 'CD45'])

    space = list(iter_points([B220, CD3e, CD4, CD8a, Ly6G, Ly6C, F480, CD11b,
                              CD11c, CD49b, CD45]))

    # Care dictionaries
    dict_yes = {}

    # dict_yes['test_yes'] = {'B220', 'CD11b', 'CD11c', 'CD3e', 'CD4', 'CD45',
    # 'CD49b', 'CD8a', 'F480', 'Ly6C', 'Ly6G'}

    # B cells
    dict_yes['B_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}
    dict_yes['CD45negB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['F480posB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD8aposB_yes'] = {
        'B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # cytotoxic T cells
    dict_yes['CD8T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposCD8T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', 'Ly6C', '~Ly6G'}
    dict_yes['B220posCD8T_yes'] = {
        'B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # helper T cells
    dict_yes['CD4T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', 'CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposCD4T_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', 'CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}

    # myeloid DCs
    dict_yes['DC_yes'] = {
        '~B220', 'CD11b', 'CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', 'Ly6C', '~Ly6G'}

    # NK cells
    dict_yes['NK_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', 'CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # PMN cells
    dict_yes['PMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', 'Ly6G'}
    dict_yes['CD45negPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', 'Ly6G'}
    dict_yes['Ly6CnegPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', 'Ly6G'}
    dict_yes['F480posPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', 'Ly6C', 'Ly6G'}
    dict_yes['F480posLy6CnegPMN_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', 'Ly6G'}

    # Monocytes
    dict_yes['Mono_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}
    dict_yes['Ly6CnegMono_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD11bnegMono_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}

    # Macrophages
    dict_yes['Mac_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposMac_yes'] = {
        '~B220', 'CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', 'Ly6C', '~Ly6G'}
    dict_yes['CD11bnegMac_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', 'F480', '~Ly6C', '~Ly6G'}

    # Double positive T cells
    dict_yes['DPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', 'CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD45negDPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', 'CD4', '~CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD3eposDPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', 'CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # Immature single positive T cells
    dict_yes['ISPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD45negISPT_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        'CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # Lymphoid tissue inducer cells
    dict_yes['LTi_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', 'CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}

    # Double negative T cells
    dict_yes['DNT_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['Ly6CposDNT_yes'] = {
        '~B220', '~CD11b', '~CD11c', 'CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', 'Ly6C', '~Ly6G'}

    # Precursor immune cells
    dict_yes['Precursor_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', 'CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}
    dict_yes['CD45negPrecursor_yes'] = {
        '~B220', '~CD11b', '~CD11c', '~CD3e', '~CD4', '~CD45', '~CD49b',
        '~CD8a', '~F480', '~Ly6C', '~Ly6G'}

    dict_yes = collections.OrderedDict(sorted(dict_yes.items()))

    # Don't care dictionaries
    dict_no = {}

    # dict_no['test_no'] = {}

    # B cells
    dict_no['B_no'] = {}
    dict_no['Ly6CposB_no'] = {}
    dict_no['CD45negB_no'] = {}
    dict_no['F480posB_no'] = {}
    dict_no['CD8aposB_no'] = {}

    # cytotoxic T cells
    dict_no['CD8T_no'] = {}
    dict_no['Ly6CposCD8T_no'] = {}
    dict_no['B220posCD8T_no'] = {}

    # helper T cells
    dict_no['CD4T_no'] = {}
    dict_no['Ly6CposCD4T_no'] = {}

    # myeloid DCs
    dict_no['DC_no'] = {}

    # NK cells
    dict_no['NK_no'] = {}

    # PMN cells
    dict_no['PMN_no'] = {}
    dict_no['CD45negPMN_no'] = {}
    dict_no['Ly6CnegPMN_no'] = {}
    dict_no['F480posPMN_no'] = {}
    dict_no['F480posLy6CnegPMN_no'] = {}

    # Monocytes
    dict_no['Mono_no'] = {}
    dict_no['Ly6CnegMono_no'] = {}
    dict_no['CD11bnegMono_no'] = {}

    # Macrophages
    dict_no['Mac_no'] = {}
    dict_no['Ly6CposMac_no'] = {}
    dict_no['CD11bnegMac_no'] = {}

    # Double positive T cells
    dict_no['DPT_no'] = {}
    dict_no['CD45negDPT_no'] = {}
    dict_no['CD3eposDPT_no'] = {}

    # Immature single positive T cells
    dict_no['ISPT_no'] = {}
    dict_no['CD45negISPT_no'] = {}

    # Lymphoid tissue inducer cells
    dict_no['LTi_no'] = {}

    # Double negative T cells
    dict_no['DNT_no'] = {}
    dict_no['Ly6CposDNT_no'] = {}

    # Precursor immune cells
    dict_no['Precursor_no'] = {}
    dict_no['CD45negPrecursor_no'] = {}

    dict_no = collections.OrderedDict(sorted(dict_no.items()))

    # Boolean expression generator
    exp = {}
    truth = {}
    vectors = pd.DataFrame()
    for (k1, v1), (k2, v2) in zip(dict_no.items(), dict_yes.items()):
        cell_type = '%s' % ((k1.rsplit('_', 2)[0]))
        current_dont_cares = []
        current_cares = []
        q = len(v1)
        s = 'Or('
        for x1 in v1:
            current_dont_cares.append(x1)
            a = '~' + x1
            current_dont_cares.append(a)
        for x2 in v2:
            current_cares.append(x2)

        for i, v in enumerate(list(itertools.combinations
                                   (current_dont_cares, q))):
            c = list(v)
            d = [x[-3:] for x in c]
            if len(d) == len(set(d)):
                f = str(current_cares + c)[1:-1].replace("'", "")
                f = 'And(' + f + '), '
                s += f
        w = s + ')'
        w = w[:-3] + w[-3:-2].replace(",", "") + \
            w[-2:-1].replace(" ", "") + w[-1:]
        exp[cell_type] = expr(w)
        truth[cell_type] = expr2truthtable(exp[cell_type])
        v = len(list(exp[cell_type].satisfy_all()))
        vector = pd.DataFrame(exp[cell_type].satisfy_all())
        vector.columns = [str(u) for u in vector.columns]
        vector['cell_type'] = cell_type
        print(vector['cell_type'].to_string(index=False))
        print((vector.loc[:, vector.columns != 'cell_type'])
              .to_string(index=False))
        print()
        vectors = vectors.append(vector)
    vectors = vectors.reset_index(drop=True)

    # show duplicate vector report
    vector_counts = vectors.sort_values(channel_list_update) \
        .groupby(channel_list_update).count()
    dupe_vectors = vector_counts[vector_counts.cell_type > 1]
    dupe_vectors = dupe_vectors.reset_index() \
        .rename(columns={'cell_type': 'count'})
    dupe_report = pd.merge(vectors, dupe_vectors, on=channel_list_update)
    if not dupe_report.empty:
        print('Duplicate vector report:')
        print(dupe_report)
        dupe_report.to_csv(DUPE_REPORT_CSV)
        print()

    # conflict resolution
    if not dupe_report.empty:
        vectors = pd.DataFrame()
        channels = tuple([col for col in dupe_report.columns
                         if col not in ['count']])

        # specify vector assignments to drop from the classifier
        # (put into cons sqaure brackets) dupe_report.loc[1:1, cols2].values,
        # dupe_report.loc[2:2, cols2].values
        conflicts = []

        conflicts = [val for sublist in conflicts for val in sublist]
        for c in range(len(conflicts)):
            j = vectors[(vectors['B220'] != conflicts[c][0]) |
                        (vectors['CD11b'] != conflicts[c][1]) |
                        (vectors['CD11c'] != conflicts[c][2]) |
                        (vectors['CD3e'] != conflicts[c][3]) |
                        (vectors['CD4'] != conflicts[c][4]) |
                        (vectors['CD49b'] != conflicts[c][5]) |
                        (vectors['CD8a'] != conflicts[c][6]) |
                        (vectors['F480'] != conflicts[c][7]) |
                        (vectors['Ly6C'] != conflicts[c][8]) |
                        (vectors['Ly6G'] != conflicts[c][9]) |
                        (vectors['cell_type'] != conflicts[c][10])]
            vectors = j
            vectors = j

    # print classifier statistics
    count = vectors['cell_type'].value_counts().tolist()
    total_vectors = sum(count)
    name = vectors['cell_type'].value_counts().index.tolist()
    print('Classifier report:')
    print(str(total_vectors) +
          ' unique vectors are specified under the current classifer.')
    print()
    print('Of which,...')
    for count, name in zip(count, name):
        print(str(count) + ' satisfies the ' + name + ' cell phenotype.')
    print()

    classified_dir = os.path.join(ORTHANT_DIR, 'classified_data')
    os.makedirs(classified_dir)

    unspecified_dir = os.path.join(ORTHANT_DIR, 'unspecified_data')
    os.makedirs(unspecified_dir)

    for zero_name, zero_frame in Boo_frames_dict_shlf.items():
        classified = pd.merge(zero_frame, vectors, how='left',
                              on=channel_list_update)
        classified = classified.fillna(value='unspecified')
        classified.to_csv(os.path.join(classified_dir, zero_name +
                          '_classified_data.csv'), index=False)
        count2 = classified['cell_type'].value_counts()
        percent_coverage = (sum(count2) - count2['unspecified']) \
            / sum(count2) * 100
        print('The current classifier covers ' + str(percent_coverage) +
              ' percent of the cells in the ' + zero_name + ' dataset,'
              ' which contains ' +
              str(len(zero_frame.iloc[:, 7:18].drop_duplicates())) +
              ' unique vectors.')
        print()

        # check residual, unclassified single-cell data
        unspecified = classified[classified['cell_type'] == 'unspecified']
        unspecified = unspecified.groupby(channel_list_update).size() \
            .reset_index().rename(columns={0: 'count'})
        unspecified = unspecified.sort_values(by='count', ascending=False)
        if not unspecified.empty:
            print(zero_name + ' unspecified vector report:')
            print(unspecified)
            print('The sum of the unspecified cells is: ' +
                  str(unspecified['count'].sum()))
            unspecified.to_csv(os.path.join(unspecified_dir, zero_name +
                               '_unspecified_report.csv'), index=False)
            print()

    os.chdir(PICKLE_DIR)
    po_total_vectors = open(PO_TOTAL_VECTORS, 'wb')
    pickle.dump(total_vectors, po_total_vectors)
    po_total_vectors.close()

    po_classified_dir = open(PO_CLASSIFIED_DIR, 'wb')
    pickle.dump(classified_dir, po_classified_dir)
    po_classified_dir.close()