from invoke_init import *
from constants_syl import *
from helpers import *


# plot data (x vs. y) for specific celltypes (celltype1 vs. celltype2)
def check_celltypes(celltype1, celltype2, celltype3, celltype4, x, y):

    banner('RUNNING MODULE: check_celltypes')

    os.chdir(PICKLE_DIR)
    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    os.chdir(aggregate_data_dir)
    overall = pd.read_csv(OVERALL_CSV)

    test = overall[[x, y, 'cell_type']][
        (overall['tissue'] == 'blood') &
        (overall['status'] == 'naive') &
        ((overall['cell_type'] == celltype1) |
        (overall['cell_type'] == celltype2) |
        (overall['cell_type'] == celltype3) |
        (overall['cell_type'] == celltype4))]

    color_dict = {
        'B220posCD8T': 'black',
        'CD8T': 'darkgray', 'CD4T': 'orange',
        'B': 'darkcyan'}

    color_list = [color_dict[i] for i in test['cell_type']]

    sns.set_style('whitegrid')
    plt.scatter(test[x], test[y], c=color_list, s=0.25)
    plt.close('all')



# plot celltype bar charts
def celltype_barcharts():

    banner('RUNNING MODULE: celltype_barcharts')

    os.chdir(PICKLE_DIR)
    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    cnt = classified_choice.groupby(['time_point', 'tissue', 'status',
                                    'replicate', 'cell_type']).size()
    cnt1 = cnt.unstack()
    cnt2 = cnt1.replace(to_replace='NaN', value=0.0)
    cnt3 = cnt2.stack().astype(int)
    ave = cnt3.groupby(level=['time_point', 'tissue', 'status',
                       'replicate']).apply(lambda x: (x / x.sum())*100)
    ave_group = ave.groupby(level=['time_point', 'tissue',
                            'status', 'cell_type'])
    mean = ave_group.mean()
    sem = ave_group.sem()

    Boo_bar_plot_TOTAL_t_input = pd.concat([mean, sem], axis=1)
    Boo_bar_plot_TOTAL_t_input.rename(columns={0: 'mean', 1: 'sem'},
                                      inplace=True)
    Boo_bar_plot_TOTAL_t_input = Boo_bar_plot_TOTAL_t_input.reset_index()
    TOTAL_save_plot = os.path.join(ORTHANT_DIR, 'Boolean_plots')
    os.mkdir(TOTAL_save_plot)
    for name, group in Boo_bar_plot_TOTAL_t_input.groupby(['time_point',
                                                          'tissue']):
        print('Plotting ' + str(name) + ' Boolean bar chart.')
        data = group.pivot_table(index='cell_type', columns='status',
                                 values=['mean', 'sem'])

        # invert status order
        data = data.sort_index(level=0, axis=1, ascending=False)

        data = data.reindex(
                labels=sorted(classified_choice['cell_type'].unique()),
                axis=0)
        data.fillna(value=0.0, inplace=True)

        sns.set_style('whitegrid')
        ax = data['mean'].plot(yerr=data['sem'], kind='bar', grid=False,
                               width=0.78, linewidth=1, figsize=(20, 10),
                               color=['b', 'g'], alpha=0.6, title=str(name))

        xlabels = [
            item.get_text() for item in ax.get_xticklabels()]

        xlabels_update = [xlabel.replace(
            'neg', '$^-$').replace('pos', '$^+$').replace(
            'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]

        ax.set_xticklabels(xlabels_update)

        for item in ax.get_xticklabels():
            item.set_size(20)
            item.set_weight('bold')

        for item in ax.get_yticklabels():
            item.set_size(15)

        ax.set_xlabel(xlabel='immunophenotype', size=18, weight='bold')
        ax.set_ylabel(ylabel='% of tissue', size=18, weight='bold')

        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.xaxis.grid(False)

        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel('% tissue composition')

        title = ax.get_title()
        title1 = title.split(',', 2)[0]
        title2 = title.split(',', 2)[1]
        title2 = title2.strip(')')
        title2 = title2[2:-1]
        title = title1 + 'dpi' + ', ' + title2 + ')'

        ax.set_title(title, size=30, weight='bold', y=1.02)

        legend_text_properties = {'size': 20, 'weight': 'bold'}
        legend = plt.legend(prop=legend_text_properties)
        for legobj in legend.legendHandles:
            legobj.set_linewidth(5.0)

        plt.tight_layout()
        plt.savefig(os.path.join(TOTAL_save_plot, str(name) +
                                 'bar_plot' + '.pdf'))
        plt.close('all')
    print()

    os.chdir(PICKLE_DIR)
    po_ave = open(PO_AVE, 'wb')
    pickle.dump(ave, po_ave)
    po_ave.close()

    po_TOTAL_save_plot = open(PO_TOTAL_SAVE_PLOT, 'wb')
    pickle.dump(TOTAL_save_plot, po_TOTAL_save_plot)
    po_TOTAL_save_plot.close()

# plot vector bar charts
def vector_barcharts():

    banner('RUNNING MODULE: vector_barcharts')

    os.chdir(PICKLE_DIR)
    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    pi_TOTAL_save_plot = open(PO_TOTAL_SAVE_PLOT, 'rb')
    TOTAL_save_plot = pickle.load(pi_TOTAL_save_plot)

    classified_choice_copy = classified_choice.copy()
    classified_choice_copy.index = pd.MultiIndex.from_arrays(
        classified_choice[channel_list_update].values.T)
    classified_choice_copy.reset_index(drop=False, inplace=True)

    classified_choice_copy['vector'] = (
        classified_choice_copy['level_0'].map(str) +
        classified_choice_copy['level_1'].map(str) +
        classified_choice_copy['level_2'].map(str) +
        classified_choice_copy['level_3'].map(str) +
        classified_choice_copy['level_4'].map(str) +
        classified_choice_copy['level_5'].map(str) +
        classified_choice_copy['level_6'].map(str) +
        classified_choice_copy['level_7'].map(str) +
        classified_choice_copy['level_8'].map(str) +
        classified_choice_copy['level_9'].map(str) +
        classified_choice_copy['level_10'].map(str))
    classified_choice_copy.drop(['level_0', 'level_1', 'level_2', 'level_3',
                                'level_4', 'level_5', 'level_6', 'level_7',
                                 'level_8', 'level_9', 'level_10'],
                                axis=1, inplace=True)

    cnt_vec = classified_choice_copy.groupby(['time_point', 'tissue', 'status',
                                             'replicate', 'vector']).size()
    cnt1_vec = cnt_vec.unstack()
    cnt2_vec = cnt1_vec.replace(to_replace='NaN', value=0.0)
    cnt3_vec = cnt2_vec.stack().astype(int)
    ave_vec = cnt3_vec.groupby(level=['time_point', 'tissue', 'status',
                               'replicate']).apply(lambda x: (x / x.sum())*100)
    ave_group_vec = ave_vec.groupby(level=['time_point', 'tissue', 'status',
                                    'vector'])
    mean_vec = ave_group_vec.mean()
    sem_vec = ave_group_vec.sem()

    Boo_bar_plot_TOTAL_t_input_vec = pd.concat([mean_vec, sem_vec], axis=1)
    Boo_bar_plot_TOTAL_t_input_vec.rename(
        columns={0: 'mean', 1: 'sem'}, inplace=True)
    Boo_bar_plot_TOTAL_t_input_vec = Boo_bar_plot_TOTAL_t_input_vec \
        .reset_index()

    TOTAL_save_plot2 = os.path.join(TOTAL_save_plot, 'total_Boolean_plots')
    os.mkdir(TOTAL_save_plot2)

    sns.set(style='whitegrid')
    for name, group in Boo_bar_plot_TOTAL_t_input_vec.groupby(['time_point',
                                                              'tissue']):

        print('Plotting ' + str(name) + ' Boolean bar chart.')
        data = group.pivot_table(index='vector', columns='status',
                                 values=['mean', 'sem'])

        # invert status order
        data = data.sort_index(level=0, axis=1, ascending=False)

        ax = data['mean'].plot(yerr=data['sem'], kind='bar', grid=False,
                               width=0.78, linewidth=1, figsize=(20, 10),
                               color=['b', 'g'], alpha=0.6, title=str(name))

        xlabels = [
            item.get_text() for item in ax.get_xticklabels()]

        xlabels_update = [xlabel.replace(
            'neg', '$^-$').replace('pos', '$^+$').replace(
            'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]

        ax.set_xticklabels(xlabels_update)

        for item in ax.get_xticklabels():
            item.set_weight('normal')

        plt.ylim(0.0, 100.0)
        ax.set_xlabel('Boolean vector')
        ax.set_ylabel('% tissue composition')
        plt.tight_layout()
        plt.savefig(os.path.join(TOTAL_save_plot2, str(name) +
                    'bar_plot' + '.pdf'))
        plt.close('all')

    os.chdir(PICKLE_DIR)
    po_classified_choice_copy = open(PO_CLASSIFIED_CHOICE_COPY, 'wb')
    pickle.dump(classified_choice_copy, po_classified_choice_copy)
    po_classified_choice_copy.close()

    po_ave_vec = open(PO_AVE_VEC, 'wb')
    pickle.dump(ave_vec, po_ave_vec)
    po_ave_vec.close()


# plot pvalue vs. magnitude scatter plots on Boolean vector data
def celltype_pval_mag():

    banner('RUNNING MODULE: celltype_pval_mag')

    os.chdir(PICKLE_DIR)
    pi_TOTAL_sig_dif = open(PO_TOTAL_SIG_DIF, 'rb')
    TOTAL_sig_dif = pickle.load(pi_TOTAL_sig_dif)

    pi_color_dict = open(PO_COLOR_DICT, 'rb')
    color_dict = pickle.load(pi_color_dict)

    # map Phenograph clusters to Boolean classifier on TOTAL sig_dif data
    TOTAL_sig_dif['dif'] = (TOTAL_sig_dif['gl261_percent'] -
                            TOTAL_sig_dif['naive_percent']).astype(float)

    TOTAL_mag_dir = os.path.join(ORTHANT_DIR, 'pvalue_vs_mag_plots')
    os.makedirs(TOTAL_mag_dir)
    os.chdir(TOTAL_mag_dir)

    sns.set(style='whitegrid')
    TOTAL_plot_dict = {}
    for name, group in TOTAL_sig_dif.groupby(['time_point']):
        print('Plotting p-value vs. magnitude for Boolean vector data at ' +
              str(name) + ' day time point.')
        fig, ax = plt.subplots()
        for tissue, color in color_dict.items():
            tissue_data = ('%s') % (tissue)
            tissue_data = [i for i in group.cell_type[group.tissue == tissue]]
            TOTAL_plot_dict[tissue] = plt.scatter(
                group.dif[group.tissue == tissue],
                -(group.pvalue[group.tissue == tissue].apply(np.log10)),
                c=color)
            for label, x, y in zip(
              tissue_data, group.dif[group.tissue == tissue],
              -(group.pvalue[group.tissue == tissue].apply(np.log10))):
                if x > 0 or x < -0:
                    plt.annotate(
                        label, size=5,
                        xy=(x, y), xytext=(-5, 5),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', fc='yellow',
                                  alpha=0.0),
                        arrowprops=dict(arrowstyle='->', color='k',
                                        connectionstyle='arc3,rad=0'))
        legend_list = []
        for key, value in color_dict.items():
            line = mlines.Line2D(
                [], [], color=value, linestyle='None', marker='o',
                markersize=10, label=key)
            legend_list.append(line)
        legend_text_properties = {'weight': 'bold'}
        plt.legend(handles=legend_list, prop=legend_text_properties)
        plt.axvline(x=0.0, ymin=0.0, ymax=5.00, linewidth=0.5,
                    linestyle='dashed', color='k', alpha=0.7)
        axes = plt.gca()

        for spine in ax.spines.values():
            spine.set_edgecolor('k')

        # axes.set_xlim([-22.4, 22.4])
        axes.set_ylim(ymin=1.3010299956639813)
        ax.set_title(str(name) + 'dpi', size=20, y=1.02, weight='bold')
        ax.set_xlabel('difference (%)', size=15, weight='bold')
        ax.set_ylabel('-log10(p-value)', size=15, weight='bold')
        ax.grid(linewidth=0.5, linestyle='--', alpha=1.0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            axis='both', which='minor', length=2.5,
            color='k', direction='in', left=True,
            right=True, bottom=True, top=True)
        plt.savefig(os.path.join(TOTAL_mag_dir, 'mag_vs_sig' +
                    str(name) + '.pdf'))
        plt.close('all')
    print()

# with split x-axis
def celltype_ratio_mag_split():

    banner('RUNNING MODULE: celltype_ratio_mag')

    os.chdir(PICKLE_DIR)

    # plot all p-values
    # pi_TOTAL_sig_dif = open(PO_TOTAL_SIG_DIF, 'rb')
    # TOTAL_sig_dif = pickle.load(pi_TOTAL_sig_dif)

    # plot only FDR-corrected q-values
    pi_sig_dif_FDRcorrected = open(
        PO_SIG_DIF_FDRCORRECTED, 'rb')
    sig_dif_FDRcorrected = pickle.load(
        pi_sig_dif_FDRcorrected)

    pi_color_dict = open(PO_COLOR_DICT, 'rb')
    color_dict = pickle.load(pi_color_dict)

    TOTAL_mag2_dir = os.path.join(ORTHANT_DIR, 'ratio_vs_mag_plots')
    os.makedirs(TOTAL_mag2_dir)
    os.chdir(TOTAL_mag2_dir)

    TOTAL_ratio_v_mag_input = sig_dif_FDRcorrected.copy()
    TOTAL_ratio_v_mag_input['dif'] = (
        TOTAL_ratio_v_mag_input['gl261_percent'] -
        TOTAL_ratio_v_mag_input['naive_percent']).astype(float)

    TOTAL_ratio_v_mag_input['ratio'] = np.log2(
        ((0.01 + TOTAL_ratio_v_mag_input['gl261_percent']) /
         (0.01 + TOTAL_ratio_v_mag_input['naive_percent'])).astype(float))

    sns.set(style='whitegrid')

    fig = plt.figure(figsize=(8, 5.7))

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1], left=0.3, wspace=0.1)

    ax = plt.subplot(gs[0])

    ax2 = plt.subplot(gs[1], sharey=ax)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax3 = plt.subplot(gs[2], sharey=ax)
    plt.setp(ax3.get_yticklabels(), visible=False)

    group_copy = TOTAL_ratio_v_mag_input.sort_values(by='dif', ascending=False)

    group_copy['cond'] = group_copy['cell_type'].map(str) + '_' + \
        group_copy['tissue'] + '_' + group_copy['time_point'].map(str)

    time_dict = {7: '#00b3a3', 14: '#000000', 30: '#a7abab'}
    group_copy['hue'] = group_copy['time_point'].map(time_dict)

    ax.barh(
        group_copy['cond'], group_copy['dif'],
        color=group_copy['hue'], linewidth=0.0)

    ax2.barh(
        group_copy['cond'], group_copy['dif'],
        color=group_copy['hue'], linewidth=0.0)

    ax3.barh(
        group_copy['cond'], group_copy['dif'],
        color=group_copy['hue'], linewidth=0.0)

    fig.canvas.draw()

    ylabels = [item.get_text() for item in ax.get_yticklabels()]

    ylabels_update = [ylabel.replace('neg', '$^-$').replace(
        'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')
                      for ylabel in ylabels]

    ax.set_yticklabels(ylabels_update)

    for x in [ax, ax2, ax3]:
        for item in x.get_xticklabels():
            item.set_rotation(0)
            item.set_size(10)

    ax.set_xlim([-20.0, -15.0])
    ax2.set_xlim([-4.5, 4.5])
    ax3.set_xlim([15.0, 20.0])

    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    legend_list = []
    for key, value in time_dict.items():
        line = mlines.Line2D(
            [], [], color=value, linestyle='-', lw=6,
            markersize=15, label=key)
        legend_list.append(line)

    legend_text_properties = {'weight': 'bold'}
    plt.legend(handles=legend_list, prop=legend_text_properties)

    plt.savefig(os.path.join(TOTAL_mag2_dir, 'mag_bar.pdf'))
    plt.close('all')

    TOTAL_plot2_dict = {}
    for name, group in TOTAL_ratio_v_mag_input.groupby(['time_point']):
        print('Plotting fold-change vs. magnitude for Boolean vector data at '
              + str(name) + ' day time point.')

        fig = plt.figure()

        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1])

        ax = plt.subplot(gs[0])

        ax2 = plt.subplot(gs[1], sharey=ax)
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax3 = plt.subplot(gs[2], sharey=ax)
        plt.setp(ax3.get_yticklabels(), visible=False)

        for tissue, color in color_dict.items():
            tissue_data = ('%s') % (tissue)
            tissue_data = [i for i in group.cell_type[group.tissue == tissue]]

            TOTAL_plot2_dict[tissue] = ax.scatter(
                group.dif[group.tissue == tissue],
                group.ratio[group.tissue == tissue], c=color, s=100)

            TOTAL_plot2_dict[tissue] = ax2.scatter(
                group.dif[group.tissue == tissue],
                group.ratio[group.tissue == tissue], c=color, s=100)

            TOTAL_plot2_dict[tissue] = ax3.scatter(
                group.dif[group.tissue == tissue],
                group.ratio[group.tissue == tissue], c=color, s=100)

            for label, x, y in zip(
              tissue_data, group.dif[group.tissue == tissue],
              group.ratio[group.tissue == tissue]):

                label = label.replace(
                    'neg', '$^-$').replace('pos', '$^+$').replace(
                    'CD8a', 'CD8' + u'\u03B1')

                for a in [ax, ax2, ax3]:
                    if x > 0 or x < -0:

                        a.annotate(
                            label, size=5,
                            xy=(x, y), xytext=(-5, 5),
                            textcoords='offset points',
                            weight='bold', ha='right', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.1', fc='yellow',
                                      alpha=0.0),
                            arrowprops=dict(arrowstyle='->', color='k',
                                            connectionstyle='arc3,rad=0',
                                            shrinkA=-0.1))

                    for spine in a.spines.values():
                        spine.set_edgecolor('k')

        legend_list = []
        for key, value in color_dict.items():
            line = mlines.Line2D(
                [], [], color=value, linestyle='None', marker='o',
                markersize=10, label=key)
            legend_list.append(line)

        legend_text_properties = {'weight': 'bold'}
        plt.legend(handles=legend_list, prop=legend_text_properties)

        ax2.axvline(x=0.0, ymin=-6.0, ymax=6.0, linewidth=0.5,
                    linestyle='dashed', color='k', alpha=0.7)
        ax.axhline(y=0.0, xmin=-25.0, xmax=25.0, linewidth=0.5,
                   linestyle='dashed', color='k', alpha=0.7)
        ax3.axhline(y=0.0, xmin=-25.0, xmax=25.0, linewidth=0.5,
                    linestyle='dashed', color='k', alpha=0.7)
        ax2.axhline(y=0.0, xmin=-25.0, xmax=25.0, linewidth=0.5,
                    linestyle='dashed', color='k', alpha=0.7)

        ax.set_xlim([-20.0, -15.0])
        ax2.set_xlim([-4.1, 2.0])
        ax3.set_xlim([15.0, 20.0])
        ax.set_ylim([-2.0, 2.0])

        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(False)

        ax2.set_title(str(name) + 'dpi', size=20, y=1.02, weight='bold')
        ax2.set_xlabel('difference (%)', size=15, weight='bold')
        ax.set_ylabel('weighted log2(fold-change)', size=15, weight='bold')
        ax.grid(linewidth=0.5, linestyle='--', alpha=1.0)
        ax2.grid(linewidth=0.5, linestyle='--', alpha=1.0)
        ax3.grid(linewidth=0.5, linestyle='--', alpha=1.0)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            axis='both', which='minor', length=2.5,
            color='k', direction='in', left=True,
            right=True, bottom=True, top=True)

        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(
            axis='both', which='minor', length=2.5,
            color='k', direction='in', left=False,
            right=False, bottom=True, top=True)

        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        ax3.tick_params(
            axis='both', which='minor', length=2.5,
            color='k', direction='in', left=False,
            right=True, bottom=True, top=True)

        plt.savefig(os.path.join(TOTAL_mag2_dir, 'ratio_vs_mag' +
                    str(name) + '.pdf'))
        plt.close('all')
    print()


# plot box plots per Boolean vector across all protein expression channels
def celltype_box_vector():

    banner('RUNNING MODULE: celltype_box_vector')

    os.chdir(PICKLE_DIR)
    pi_channel_list = open(PO_CHANNEL_LIST, 'rb')
    channel_list = pickle.load(pi_channel_list)

    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    os.chdir(aggregate_data_dir)
    overall = pd.read_csv(OVERALL_CSV)

    TOTAL_box_vector_dir = os.path.join(ORTHANT_DIR, 'box_vector_plots')
    os.makedirs(TOTAL_box_vector_dir)
    os.chdir(TOTAL_box_vector_dir)

    channel_list_cd8a = channel_list.copy()
    channel_list_cd8a[7] = 'cd8a'
    expression_frames = []
    ov = overall[channel_list_cd8a + ['fsc', 'ssc', 'cell_type',
                                      'status']][overall['cell_type'] !=
                                                 'unspecified']
    ov = ov.sort_values(['cell_type', 'status'], ascending=[True, False])

    del overall

    for name, group in ov.groupby(['cell_type']):
        print('Extracting protein expression data for the ' + name +
              ' cell Boolean vector.')
        group.columns = [str(col) + '_' + name for col in group.columns]
        data = pd.melt(group)
        for channel in channel_list_cd8a:
            x = data['variable'][data['variable'] == channel +
                                 '_' + name].reset_index(drop=True)
            y = data['value'][data['variable'] == channel + '_'
                              + name].reset_index(drop=True)
            hue = data['value'][data['variable'] == 'status' + '_'
                                + name].reset_index(drop=True)
            m = pd.concat([x, y, hue], axis=1)
            m.columns = ['marker', 'value', 'status']
            expression_frames.append(m)
            overall_dataframe_TOTAL = pd.concat(expression_frames)
    print()

    print('Splitting marker column into channel and cell_type columns'
          ' in overall_dataframe_TOTAL')
    split1 = overall_dataframe_TOTAL['marker'].str.split('_', expand=True)
    to_concat = [split1, overall_dataframe_TOTAL.iloc[:, 1:]]
    foo = pd.concat(to_concat,  axis=1)
    foo.columns = ['channel', 'cell_type', 'value', 'status']
    print()

    sns.set(style='whitegrid')
    for cell in sorted(foo['cell_type'].unique()):
        print('Plotting protein expression data across all channels for the '
              + str(cell) + ' cell Boolean vector.')
        data = foo[foo['cell_type'] == cell].copy()
        data['value'] = data['value'].astype('float')
        dashboard_shlf[cell]['box'] = data
        dashboard_shlf[cell]['box-fsc-ssc'] = ov[ov['cell_type'] == cell]

        g = sns.boxplot(
            x='channel', y='value', hue='status', data=data, linewidth=0.5,
            palette={'gl261': 'mediumaquamarine', 'naive': 'b'},
            order=['cd45', 'b220', 'cd11b', 'cd11c', 'cd3e', 'cd4', 'cd49b',
                   'cd8a', 'f480', 'ly6c', 'ly6g'])
        sns.despine(left=True)

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
        g.xaxis.grid(False)
        g.yaxis.grid(True)

        legend_text_properties = {'size': 10, 'weight': 'bold'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')

        g.set_xlabel('channel', size=15, weight='bold')

        g.set_ylabel('intensity', size=15, weight='bold')

        plt.ylim(-3.0, 3.0)
        g.set_title(str(cell), fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(TOTAL_box_vector_dir, str(cell) + '.pdf'))
        plt.close('all')
    print()

    del split1
    del to_concat
    del foo
    os.chdir(PICKLE_DIR)
    po_ov = open(PO_OV, 'wb')
    pickle.dump(ov, po_ov)
    po_ov.close()

    po_channel_list_cd8a = open(PO_CHANNEL_LIST_CD8A, 'wb')
    pickle.dump(channel_list_cd8a, po_channel_list_cd8a)
    po_channel_list_cd8a.close()

    del ov

    dashboard_shlf.close()





# plot violin plots per Boolean vector across both scatter channels
def celltype_box_scatter():

    banner('RUNNING MODULE: celltype_box_scatter')

    os.chdir(PICKLE_DIR)
    pi_ov = open(PO_OV, 'rb')
    ov = pickle.load(pi_ov)

    TOTAL_box_vector_scatter_dir = os.path.join(
        ORTHANT_DIR, 'box_vector_plots_scatter')
    os.makedirs(TOTAL_box_vector_scatter_dir)
    os.chdir(TOTAL_box_vector_scatter_dir)

    TOTAL_scatter_frames = []
    for name, group in ov[['fsc', 'ssc', 'cell_type',
                          'status']].groupby(['cell_type']):
        print('Extracting fsc/ssc data for the ' + name +
              ' cell Boolean vector.')
        group.columns = [str(col) + '_' + name for col in group.columns]
        data = pd.melt(group)
        for channel in ['fsc', 'ssc']:
            x = data['variable'][data['variable'] == channel + '_'
                                 + name].reset_index(drop=True)
            y = data['value'][data['variable'] == channel + '_'
                              + name].reset_index(drop=True)
            hue = data['value'][data['variable'] == 'status' + '_'
                                + name].reset_index(drop=True)
            m = pd.concat([x, y, hue], axis=1)
            m.columns = ['marker', 'value', 'status']
            TOTAL_scatter_frames.append(m)
            overall_dataframe_TOTAL_scatter = pd.concat(TOTAL_scatter_frames)
    print()

    print('Splitting marker column into channel and cell_type columns'
          ' in overall_dataframe_TOTAL')
    split1_scatter = overall_dataframe_TOTAL_scatter['marker'] \
        .str.split('_', expand=True)
    to_concat_scatter = [split1_scatter,
                         overall_dataframe_TOTAL_scatter.iloc[:, 1:]]
    foo_scatter = pd.concat(to_concat_scatter,  axis=1)
    foo_scatter.columns = ['channel', 'cell_type', 'value', 'status']
    print()

    sns.set(style='whitegrid')
    for cell in sorted(foo_scatter['cell_type'].unique()):
        fig, ax = plt.subplots(figsize=(3, 10))
        fig.subplots_adjust(left=0.2, right=0.95, bottom=0.1,
                            top=0.95)
        print('Plotting fsc/ssc data across all channels for the '
              + str(cell) + ' cell Boolean vector.')
        data = foo_scatter[foo_scatter['cell_type'] == cell].copy()
        data['value'] = data['value'].astype('float')
        g = sns.boxplot(
            x='channel', y='value', hue='status', data=data,
            palette={'gl261': 'mediumaquamarine', 'naive': 'b'},
            order=['fsc', 'ssc'], zorder=3)

        name = cell.replace('neg', '$^-$').replace('pos', '$^+$').replace(
            'CD8a', 'CD8' + u'\u03B1')

        sns.despine(left=True)
        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(15)
            item.set_fontweight('bold')
        plt.ylim(-0.0, 250000)
        ax.set_xlabel('channel', size=15, weight='bold', labelpad=12)
        g.set_title(name, fontweight='bold')

        plt.savefig(os.path.join(TOTAL_box_vector_scatter_dir,
                    str(cell) + '.pdf'))
        plt.close('all')
    print()

    os.chdir(PICKLE_DIR)





# plot box plots per channel across all Boolean vectors
def celltype_box_channel():

    banner('RUNNING MODULE: celltype_box_channel')

    os.chdir(PICKLE_DIR)
    pi_channel_list_cd8a = open(PO_CHANNEL_LIST_CD8A, 'rb')
    channel_list_cd8a = pickle.load(pi_channel_list_cd8a)

    pi_ov = open(PO_OV, 'rb')
    ov = pickle.load(pi_ov)

    os.chdir(ORTHANT_DIR)
    if os.path.isdir(ORTHANT_DIR + '/box_channel_plots') is False:
        TOTAL_box_channel_dir = os.path.join(ORTHANT_DIR, 'box_channel_plots')
        os.makedirs(TOTAL_box_channel_dir)

    else:
        TOTAL_box_channel_dir = os.path.join(ORTHANT_DIR, 'box_channel_plots')

    sns.set(style='whitegrid')
    for channel in channel_list_cd8a:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        print('Plotting ' + channel +
              ' protein expression across all Boolean vectors.')

        order = []

        for name, group in ov.groupby(['cell_type', 'status']):
            if name[1] == 'naive':

                order.append((name[0], group[channel].median()))

        order.sort(key=lambda x: x[1], reverse=True)
        order = [i[0] for i in order]

        # define cell_type subset
        # ov = ov[
        #     (ov['cell_type'] == 'CD4posT') |
        #     (ov['cell_type'] == 'CD8aposT') |
        #     (ov['cell_type'] == 'CD8aposISPT') |
        #     (ov['cell_type'] == 'DNT') |
        #     (ov['cell_type'] == 'CD3eposDPT')|
        #     (ov['cell_type'] == 'CD4posT') |
        #     (ov['cell_type'] == 'DPT')]

        g = sns.boxplot(
            x='cell_type', y=channel, hue='status', data=ov, linewidth=0.5,
            palette={'gl261': 'mediumaquamarine', 'naive': 'b'},
            order=order)
        sns.despine(left=True)

        xlabels = [item.get_text() for item in g.get_xticklabels()]

        xlabels_update = [xlabel.replace('neg', '$^-$').replace(
            'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')
                          for xlabel in xlabels]

        g.set_xticklabels(xlabels_update)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')

        legend_text_properties = {'size': 10, 'weight': 'bold'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 1.0))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
        g.xaxis.grid(False)
        g.yaxis.grid(True)

        g.set_xlabel('cell class', size=15, weight='bold')

        g.set_ylabel('intensity', size=15, weight='bold', labelpad=15)

        channel_update = channel.replace(
            'cd8a', 'CD8' + u'\u03B1').replace('cd3e', 'CD3' + u'\u03B5')

        g.set_title(str(channel_update), size=20, fontweight='bold', y=1.2)

        plt.ylim(-3.0, 3.0)

        plt.tight_layout()
        plt.savefig(os.path.join(TOTAL_box_channel_dir,
                                 'vector_vs_' + str(channel) + '.pdf'))
        plt.close('all')
    print()





# plot box plots per scatter channel across all Boolean vectors
def scatter_box_channel():

    banner('RUNNING MODULE: scatter_box_channel')

    os.chdir(PICKLE_DIR)
    pi_channel_list_cd8a = open(PO_CHANNEL_LIST_CD8A, 'rb')
    channel_list_cd8a = pickle.load(pi_channel_list_cd8a)

    pi_ov = open(PO_OV, 'rb')
    ov = pickle.load(pi_ov)

    if os.path.isdir(ORTHANT_DIR + '/box_channel_plots') is False:
        TOTAL_box_channel_dir = os.path.join(ORTHANT_DIR, 'box_channel_plots')
        os.makedirs(TOTAL_box_channel_dir)

    else:
        TOTAL_box_channel_dir = os.path.join(ORTHANT_DIR, 'box_channel_plots')

    sns.set(style='whitegrid')

    for channel in ['fsc', 'ssc']:

        plt.rcParams['figure.figsize'] = 8, 5.5
        print('Plotting ' + channel +
              ' protein expression across all Boolean vectors.')

        order = []

        # define cell_type subset
        # ov = ov[
        #     (ov['cell_type'] == 'CD4posT') |
        #     (ov['cell_type'] == 'CD8aposT') |
        #     (ov['cell_type'] == 'CD8aposISPT') |
        #     (ov['cell_type'] == 'DNT') |
        #     (ov['cell_type'] == 'CD3eposDPT')|
        #     (ov['cell_type'] == 'CD4posT') |
        #     (ov['cell_type'] == 'DPT')]

        for name, group in ov.groupby(['cell_type', 'status']):
            if name[1] == 'naive':
                if channel == 'fsc':
                    order.append((name[0], group[channel].median()))

                elif channel == 'ssc':
                    order.append((name[0], group[channel].median()))

        order.sort(key=lambda x: x[1])
        order = [i[0] for i in order]

        g = sns.boxplot(
            x='cell_type', y=channel, hue='status', data=ov, linewidth=0.5,
            palette={'gl261': 'mediumaquamarine', 'naive': 'b'}, order=order)
        sns.despine(left=True)
        # order=natsorted(ov['cell_type'].unique())

        xlabels = [item.get_text() for item in g.get_xticklabels()]

        xlabels_update = [xlabel.replace('neg', '$^-$').replace(
            'pos', '$^+$').replace('CD8a', 'CD8' + u'\u03B1')
                          for xlabel in xlabels]

        g.set_xticklabels(xlabels_update)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_fontweight('normal')

        legend_text_properties = {'size': 10, 'weight': 'bold'}
        legend = plt.legend(prop=legend_text_properties, loc=(0, 0.97))

        for legobj in legend.legendHandles:
            legobj.set_linewidth(0)

        g.grid(color='grey', linestyle='--', linewidth=0.5, alpha=1.0)
        g.xaxis.grid(False)
        g.yaxis.grid(True)

        g.set_xlabel('immunophenotype', size=15, weight='bold')

        ylabel = g.get_ylabel()
        g.set_ylabel(ylabel, size=30, weight='bold', labelpad=15)

        g.set_title(str(channel), size=20, fontweight='bold', y=1.002)
        plt.tight_layout()
        plt.savefig(os.path.join(TOTAL_box_channel_dir,
                                 'vector_vs_' + str(channel) + '.pdf'))
        plt.close('all')
    print()





# plot tissue pie charts per Boolean vector
def celltype_piecharts():

    banner('RUNNING MODULE: TOTAL_piecharts')

    # define factor generator
    def factors(n):
        flatten_iter = itertools.chain.from_iterable
        return set(flatten_iter((i, n//i)
                   for i in range(1, int(n**0.5)+1) if n % i == 0))


    os.chdir(PICKLE_DIR)
    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_color_dict = open(PO_COLOR_DICT, 'rb')
    color_dict = pickle.load(pi_color_dict)

    TOTAL_tissue_pie_dir = os.path.join(ORTHANT_DIR, 'tissue_pie_plots')
    os.makedirs(TOTAL_tissue_pie_dir)
    os.chdir(TOTAL_tissue_pie_dir)

    classified_choice['cell_type'] = classified_choice['cell_type'] \
        .str.replace('/', '-')
    for name, group in classified_choice.groupby(['cell_type']):
        print('Plotting Boolean vector ' + str(name) + ' piechart.')
        data = group['tissue'].value_counts()
        dashboard_shlf[str(name)]['data'] = data
        fig, ax = plt.subplots()
        labels = data.index.tolist()
        colors = [color_dict[x] for x in labels]
        patches, texts, autotexts = ax.pie(
            data, shadow=False, colors=colors, autopct='%1.1f%%',
            startangle=90, radius=0.1)
        plt.axis('equal')
        plt.legend(labels, loc='upper right')
        ax.set_title(str(name))

        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

        plt.savefig(os.path.join(TOTAL_tissue_pie_dir, str(name) + '.pdf'))
        plt.close('all')
    print()

    # get best factorization of total number of classified Boolean vectors (+1;
    # unspecified)
    TOTAL_n = len(classified_choice.groupby(['cell_type'])) + 1

    TOTAL_factors = factors(TOTAL_n)
    TOTAL_factors = list(TOTAL_factors)

    TOTAL_tuple_list = []
    for i, v in enumerate(list(itertools.combinations(TOTAL_factors, 2))):
        if v[0] * v[1] == TOTAL_n:
            TOTAL_tuple_list.append(v)

    TOTAL_dif_list = []
    for pair in TOTAL_tuple_list:
        TOTAL_dif_list.append(abs(pair[0] - pair[1]))

    TOTAL_tuple_dict = dict(zip(TOTAL_tuple_list, TOTAL_dif_list))
    TOTAL_target_tuple = min(TOTAL_tuple_dict, key=TOTAL_tuple_dict.get)

    # plot tissue pie charts per Boolean vector where piechart radius is
    # proportional to population size
    the_grid = GridSpec(TOTAL_target_tuple[0], TOTAL_target_tuple[1])
    the_grid.update(hspace=30, wspace=30, left=0.1,
                    right=0.93, bottom=0.08, top=0.83)
    coordinates = [(x, y) for x in range(
        TOTAL_target_tuple[0]) for y in range(TOTAL_target_tuple[1])]

    dif = len(coordinates) - len(classified_choice.groupby(['cell_type']))

    if dif > 0:
        coordinates = coordinates[:-dif]
    else:
        pass

    for coordinate, (name, group) in itertools.zip_longest(
      coordinates, classified_choice.groupby(['cell_type'])):
        print(coordinate, name)
        data = group['tissue'].value_counts()
        total = group['tissue'].value_counts().sum()
        percent = (total/(len(classified_choice.index))*100)
        dashboard_shlf[str(name)]['percent'] = percent
        print('Plotting population size proportioned Boolean vector '
              + str(name) + ' (' + str(percent) + ')' + ' piechart.')
        radius = math.sqrt(percent)*10
        ax = plt.subplot(the_grid[coordinate], aspect=1)
        name = name.replace('neg', '$^-$').replace('pos', '$^+$').replace(
            'CD8a', 'CD8' + u'\u03B1')
        ax.set_title(str(name), y=(radius/2.5), loc='left',
                     fontsize=6.0, weight='normal')
        labels = data.index.tolist()
        colors = [color_dict[x] for x in labels]
        patches, texts = ax.pie(data, shadow=False, radius=radius,
                                colors=colors, startangle=90)
        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

    plt.savefig(os.path.join(TOTAL_tissue_pie_dir,
                'TOTAL_pie_charts' + '.pdf'))
    plt.close('all')
    print()

    dashboard_shlf.close()





# plot time difference heatmap plots on TOTAL data
def TOTAL_mag_heatmap():

    banner('RUNNING MODULE: TOTAL_mag_heatmap')

    os.chdir(PICKLE_DIR)
    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    pi_TOTAL_sig_dif_all = open(PO_TOTAL_SIG_DIF_ALL, 'rb')
    TOTAL_sig_dif_all = pickle.load(pi_TOTAL_sig_dif_all)

    pi_sig_dif_FDRcorrected = open(
        PO_SIG_DIF_FDRCORRECTED, 'rb')
    sig_dif_FDRcorrected = pickle.load(
        pi_sig_dif_FDRcorrected)

    TOTAL_time_mag_heatmap_dir = os.path.join(
        ORTHANT_DIR, 'time_mag_heatmap_plots')
    os.makedirs(TOTAL_time_mag_heatmap_dir)

    TOTAL_sig_dif_all_mag = TOTAL_sig_dif_all.copy()
    TOTAL_sig_dif_all_mag['dif'] = (
        TOTAL_sig_dif_all_mag['gl261_percent'] -
        TOTAL_sig_dif_all_mag['naive_percent']).astype(float)
    plt.rcParams['font.weight'] = 'normal'

    cmap = sns.color_palette('RdBu_r', n_colors=1000000)

    order_of_dict = sorted(
        dashboard_shlf, key=lambda x: dashboard_shlf[x]['percent'],
        reverse=True)

    for name, group in TOTAL_sig_dif_all_mag.groupby(['cell_type']):
        group = group.pivot_table(
            index='tissue', columns='time_point', values='dif')
        dashboard_shlf[name]['sig_dif_all'] = group

    for name, group in TOTAL_sig_dif_all_mag.groupby(['tissue']):
        print('Plotting ' + name + ' Boolean vector difference heatmap.')

        fig, ax = plt.subplots()

        group = group.pivot_table(index='cell_type', columns='time_point',
                                  values='dif')

        # sort each group object according to cell_type population size
        group['cat'] = pd.Categorical(
            group.index, categories=order_of_dict, ordered=True)
        group.sort_values('cat', inplace=True)
        group.drop('unspecified', axis=0, inplace=True)
        group.drop('cat', axis=1, inplace=True)

        # grab FDR-corrected q-values
        pv = sig_dif_FDRcorrected[
            sig_dif_FDRcorrected['tissue'] == name].pivot_table(
                index='cell_type', columns='time_point',
                values='corrected_pvalue')

        # reshape pv to look like group and replace with significance icons
        if not pv.empty:

            for i in group.columns:
                if i not in pv.columns:
                    pv[i] = np.nan

            cols = sorted(pv.columns)
            pv = pv[cols]

            pv = group.merge(
                pv, how='left', left_index=True, right_index=True)

            for c in pv.columns:
                if 'x' in c:
                    pv.drop(c, axis=1, inplace=True)

            cols_dict = {i: int(i[:-2]) for i in pv.columns}

            pv.rename(cols_dict, axis=1, inplace=True)

            for col in pv:
                s = pv[col]
                for i in s.iteritems():
                    idx = i[0]
                    if not i[1] == np.nan:
                        if 0.01 < i[1] <= 0.05:
                            pv.loc[idx, col] = '*'
                        elif 0.001 < i[1] <= 0.01:
                            pv.loc[idx, col] = '**'
                        elif i[1] <= 0.001:
                            pv.loc[idx, col] = '***'

            pv.replace(to_replace=np.nan, value='', inplace=True)

            annot = pv
            kws = {'size': 7}

        else:
            annot = None
            kws = None

        g = sns.heatmap(group, square=True, linewidth=0.5,
                        vmin=-10, vmax=10, fmt='', cmap='cividis', center=0.0,
                        annot=annot, xticklabels=1, yticklabels=1,
                        annot_kws=kws)

        ylabels = [
            item.get_text() for item in g.get_yticklabels()]

        ylabels_update = [i.replace(
                          'neg', '$^-$').replace('pos', '$^+$').replace(
                          'CD8a', 'CD8' + u'\u03B1') for i
                          in ylabels]

        g.set_yticklabels(ylabels_update)

        for item in g.get_yticklabels():
            item.set_rotation(0)
            item.set_size(5)
        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(5)

        ax.set_title(str(name))
        plt.savefig(os.path.join(TOTAL_time_mag_heatmap_dir,
                    str(name) + '.pdf'))
        plt.close('all')
    print()

    os.chdir(PICKLE_DIR)
    po_TOTAL_sig_dif_all_mag = open(
        PO_TOTAL_SIG_DIF_ALL_MAG, 'wb')
    pickle.dump(TOTAL_sig_dif_all_mag,
                po_TOTAL_sig_dif_all_mag)
    po_TOTAL_sig_dif_all_mag.close()

    dashboard_shlf.close()





# plot time ratio heatmap plots on TOTAL data
# (switch between dashboard p- and q-values)
def TOTAL_ratio_heatmap():

    banner('RUNNING MODULE: TOTAL_ratio_heatmap')

    os.chdir(PICKLE_DIR)
    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    pi_TOTAL_sig_dif_all_mag = open(
        PO_TOTAL_SIG_DIF_ALL_MAG, 'rb')
    TOTAL_sig_dif_all_mag = pickle.load(
        pi_TOTAL_sig_dif_all_mag)

    pi_sig_dif_FDRcorrected = open(
        PO_SIG_DIF_FDRCORRECTED, 'rb')
    sig_dif_FDRcorrected = pickle.load(
        pi_sig_dif_FDRcorrected)

    TOTAL_time_ratio_heatmap_dir = os.path.join(
        ORTHANT_DIR, 'time_ratio_heatmap_plots')
    os.makedirs(TOTAL_time_ratio_heatmap_dir)
    os.chdir(TOTAL_time_ratio_heatmap_dir)

    TOTAL_sig_dif_all_mag['ratio'] = np.log2(
        ((0.01 + TOTAL_sig_dif_all_mag['gl261_percent']) /
         (0.01 + TOTAL_sig_dif_all_mag['naive_percent'])).astype(float))

    TOTAL_sig_dif_all_mag['ratio'].replace(
        to_replace=['Nan'], value=0.0, inplace=True)

    # prep FDR-corrected conditions for dashboard plot
    sig_dif_FDRcorrected_ratio = sig_dif_FDRcorrected.copy()
    sig_dif_FDRcorrected_ratio.reset_index(drop=True, inplace=True)
    sig_dif_FDRcorrected_ratio = sig_dif_FDRcorrected_ratio.sort_values(
        ['time_point', 'tissue', 'cell_type', 'pvalue'])
    sig_dif_FDRcorrected_ratio.reset_index(drop=True, inplace=True)

    # if plotting uncorrected p-values in dashboard plot
    # filter to get only statistically significant conditions
    # for dashboard plot
    # TOTAL_sig_dif_all_mag_ratio = TOTAL_sig_dif_all_mag.copy()
    # TOTAL_sig_dif_mag_ratio = TOTAL_sig_dif_all_mag_ratio[
    #     TOTAL_sig_dif_all_mag_ratio['pvalue'] <= 0.05]
    # TOTAL_sig_dif_mag_ratio = TOTAL_sig_dif_mag_ratio[
    #     abs(TOTAL_sig_dif_mag_ratio['statistic']) > 2.131]
    # TOTAL_sig_dif_mag_ratio.reset_index(drop=True, inplace=True)
    # TOTAL_sig_dif_mag_ratio = TOTAL_sig_dif_mag_ratio.sort_values(
    #     ['time_point', 'tissue', 'cell_type', 'pvalue'])
    # TOTAL_sig_dif_mag_ratio.reset_index(drop=True, inplace=True)

    order_of_dict = sorted(
        dashboard_shlf, key=lambda x: dashboard_shlf[x]['percent'],
        reverse=True)

    for name, group in TOTAL_sig_dif_all_mag.groupby(['cell_type']):
        group = group.pivot_table(
            index='tissue', columns='time_point', values='ratio')
        dashboard_shlf[name]['sig_ratio_all'] = group

    for name, group in TOTAL_sig_dif_all_mag.groupby(['tissue']):
        print('Plotting ' + name + ' Boolean vector ratio heatmap.')

        fig, ax = plt.subplots()

        group = group.pivot_table(index='cell_type', columns='time_point',
                                  values='ratio')

        # sort each group object according to cell_type population size
        group['cat'] = pd.Categorical(
            group.index, categories=order_of_dict, ordered=True)
        group.sort_values('cat', inplace=True)
        group.drop('unspecified', axis=0, inplace=True)
        group.drop('cat', axis=1, inplace=True)

        # grab FDR-corrected q-values
        pv = sig_dif_FDRcorrected_ratio[
            sig_dif_FDRcorrected_ratio['tissue'] == name].pivot_table(
                index='cell_type', columns='time_point',
                values='corrected_pvalue')

        # reshape pv to look like group and replace with significance icons
        if not pv.empty:

            for i in group.columns:
                if i not in pv.columns:
                    pv[i] = np.nan
            cols = sorted(pv.columns)
            pv = pv[cols]

            pv = group.merge(pv, how='left', left_index=True, right_index=True)

            for c in pv.columns:
                if 'x' in c:
                    pv.drop(c, axis=1, inplace=True)

            cols_dict = {i: int(i[:-2]) for i in pv.columns}

            pv.rename(cols_dict, axis=1, inplace=True)

            for col in pv:
                s = pv[col]
                for i in s.iteritems():
                    idx = i[0]
                    if not i[1] == np.nan:
                        if 0.01 < i[1] <= 0.05:
                            pv.loc[idx, col] = '*'
                        elif 0.001 < i[1] <= 0.01:
                            pv.loc[idx, col] = '**'
                        elif i[1] <= 0.001:
                            pv.loc[idx, col] = '***'

            pv.replace(to_replace=np.nan, value='', inplace=True)

            annot = pv
            kws = {'size': 7}

        else:
            annot = None
            kws = None

        g = sns.heatmap(group, square=True, linewidth=0.5,
                        vmin=-4.0, vmax=4.0, fmt='', cmap='cividis',
                        center=0.0, annot=annot, xticklabels=1, yticklabels=1,
                        annot_kws=kws)

        ylabels = [
            item.get_text() for item in g.get_yticklabels()]

        ylabels_update = [i.replace(
                          'neg', '$^-$').replace('pos', '$^+$').replace(
                          'CD8a', 'CD8' + u'\u03B1') for i
                          in ylabels]

        g.set_yticklabels(ylabels_update)

        for item in g.get_yticklabels():
            item.set_rotation(0)
            item.set_size(5)
        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(5)

        ax.set_title(str(name))
        plt.savefig(os.path.join(TOTAL_time_ratio_heatmap_dir,
                    str(name) + '.pdf'))
        plt.close('all')
    print()

    os.chdir(PICKLE_DIR)
    po_sig_dif_FDRcorrected_ratio = open(
        PO_SIG_DIF_FDRCORRECTED_RATIO, 'wb')
    pickle.dump(sig_dif_FDRcorrected_ratio,
                po_sig_dif_FDRcorrected_ratio)
    po_sig_dif_FDRcorrected_ratio.close()

    # if plotting uncorrected p-values in dashboard plot
    # po_TOTAL_sig_dif_mag_ratio = open(
    #     'TOTAL_sig_dif_mag_ratio.pickle', 'wb')
    # pickle.dump(TOTAL_sig_dif_mag_ratio,
    #             po_TOTAL_sig_dif_mag_ratio)
    # po_TOTAL_sig_dif_mag_ratio.close()

    dashboard_shlf.close()





# plot single heatmap for celltypes
def celltype_heatmap():

    banner('RUNNING MODULE: celltype_heatmap')

    os.chdir(PICKLE_DIR)
    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    pi_TOTAL_save_plot = open(PO_TOTAL_SAVE_PLOT, 'rb')
    TOTAL_save_plot = pickle.load(pi_TOTAL_save_plot)

    os.chdir(aggregate_data_dir)
    overall = pd.read_csv(OVERALL_CSV)

    print('Plotting Boolean classifier heatmap.')
    print()
    Boo_heatmap_input = overall[overall['cell_type'] != 'unspecified']
    Boo_heatmap_input = Boo_heatmap_input[
        channel_list_update + ['fsc', 'ssc', 'cell_type']]

    for name, group in Boo_heatmap_input.groupby(['cell_type']):

        mean_fsc = group['fsc'].mean()
        mean_ssc = group['ssc'].mean()
        print(
            name + ':', 'mean_fsc: ' + str(mean_fsc),
            'mean_ssc: ' + str(mean_ssc))

        if mean_fsc > 35000:
            Boo_heatmap_input.loc[
                Boo_heatmap_input.cell_type == name, 'fsc'] = 1

        else:
            Boo_heatmap_input.loc[
                Boo_heatmap_input.cell_type == name, 'fsc'] = 0

        if mean_ssc > 97000:
            Boo_heatmap_input.loc[
                Boo_heatmap_input.cell_type == name, 'ssc'] = 1
        else:
            Boo_heatmap_input.loc[
                Boo_heatmap_input.cell_type == name, 'ssc'] = 0

    Boo_heatmap_input['fsc'] = np.int64(Boo_heatmap_input['fsc'])
    Boo_heatmap_input['ssc'] = np.int64(Boo_heatmap_input['ssc'])

    Boo_heatmap_input = Boo_heatmap_input.drop_duplicates()
    Boo_heatmap_input.rename(
        {'fsc': 'FSC', 'ssc': 'SSC'}, axis=1, inplace=True)

    Boo_heatmap_input['percent'] = [
        dashboard_shlf[i]['percent'] for i in
        Boo_heatmap_input['cell_type'] if i != 'unspecified']

    Boo_heatmap_input = Boo_heatmap_input.sort_values(
        by='percent', ascending=False).reset_index(drop=True)
    Boo_heatmap_input.drop('percent', axis=1, inplace=True)

    Boo_heatmap_input = Boo_heatmap_input.set_index(
        'cell_type', drop=True, append=False, inplace=False,
        verify_integrity=False)

    plt.rcParams['font.weight'] = 'normal'
    plt.figure()
    fig, ax = plt.subplots()
    ax.set_title('Boolean immunophenotypes', y=1.05, weight='normal')

    ax = sns.heatmap(Boo_heatmap_input, ax=ax, cbar=False, square=True,
                     linewidths=1.75, cmap='rocket_r',
                     xticklabels=1, yticklabels=1)

    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axhline(y=Boo_heatmap_input.shape[0], color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.axvline(x=Boo_heatmap_input.shape[1], color='k', linewidth=1.5)

    ylabels = [
        item.get_text() for item in ax.get_yticklabels()]

    ylabels_update = [ylabel.replace(
        'neg', '$^-$').replace('pos', '$^+$').replace(
        'CD8a', 'CD8' + u'\u03B1') for ylabel in ylabels]

    ax.set_yticklabels(ylabels_update)

    for item in ax.get_yticklabels():
        item.set_rotation(0)
        item.set_size(7)
        item.set_weight('normal')
    for item in ax.get_xticklabels():
        item.set_rotation(70)
        item.set_size(8)
        item.set_weight('normal')

    ax.set_ylabel('')

    xlabels = [
        item.get_text() for item in ax.get_xticklabels()]

    xlabels_update = [xlabel.replace(
        'CD3e', 'CD3' + u'\u03B5').replace(
        'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]

    ax.set_xticklabels(xlabels_update)

    plt.tight_layout()
    plt.savefig(os.path.join(TOTAL_save_plot, 'Boolean_heatmap' + '.pdf'))
    plt.close('all')
    print()





# plot vector percentages and accumulation
def vector_accumulation():

    banner('RUNNING MODULE: vector_accumulation')

    os.chdir(PICKLE_DIR)
    pi_classified_choice = open(PO_CLASSIFIED_CHOICE, 'rb')
    classified_choice = pickle.load(pi_classified_choice)

    pi_channel_list_update = open(PO_CHANNEL_LIST_UPDATE, 'rb')
    channel_list_update = pickle.load(pi_channel_list_update)

    pi_color_dict = open(PO_COLOR_DICT, 'rb')
    color_dict = pickle.load(pi_color_dict)

    complete_data = classified_choice[channel_list_update
                                      + ['tissue', 'cell_type']]

    accumulation_plots = os.path.join(ORTHANT_DIR, 'accumulation_plots')
    os.makedirs(accumulation_plots)

    percentage_plots = os.path.join(ORTHANT_DIR, 'percentage_plots')
    os.makedirs(percentage_plots)

    agg_cnt = []
    for tissue in sorted(complete_data['tissue'].unique()):
        tissue_data = complete_data[complete_data['tissue'] == tissue]
        cnt = tissue_data.groupby(['tissue', 'cell_type']).size()
        cnt_frame = pd.DataFrame(cnt).reset_index()
        cnt_frame = cnt_frame.rename(columns={0: 'count'})
        cnt_frame.sort_values(by='count', ascending=False, inplace=True)

        percentage = []
        for index, row in cnt_frame.iterrows():
            percentage.append(row['count']/(cnt_frame['count'].sum())*100)
        cnt_frame['percentage'] = percentage

        # drop 'unspecified' cell count so accumulation only goes to cutoff
        # pre-specified in vector_coverage function)
        cnt_frame_drop = cnt_frame[cnt_frame['cell_type'] != 'unspecified']

        cnt_frame_drop = cnt_frame_drop.copy()
        cnt_frame_drop['accumulation'] = pd.DataFrame(
            np.cumsum(cnt_frame_drop['percentage']))

        agg_cnt.append(cnt_frame_drop)
        agg_cnt_frame = pd.concat(agg_cnt, axis=0)
        agg_cnt_frame.drop(['count', 'accumulation'], axis=1, inplace=True)
        percentage_stacked = agg_cnt_frame.pivot_table(
            values=['percentage'], index='cell_type', columns='tissue')

        percentage_stacked.columns = percentage_stacked.columns.droplevel(0)
        percentage_stacked['sum'] = percentage_stacked.sum(axis=1)

        percentage_stacked_normal = percentage_stacked.loc[
            :, 'blood':'sum'].div(percentage_stacked['sum'], axis=0)

        # compute Shannon Entropy
        entropy_list = []
        for row in percentage_stacked_normal.iterrows():
            p_data = row[1][:-1]

            # ensure no NaNs
            p_data[pd.isnull(p_data)] = 0.0
            entropy = scipy.stats.entropy(p_data, base=2)
            entropy_list.append(entropy)

        percentage_stacked_normal['entropy'] = entropy_list
        percentage_stacked_normal.sort_values(
            by='entropy', ascending=True, inplace=True)

        percentage_tissue_dir = os.path.join(percentage_plots, tissue)
        os.makedirs(percentage_tissue_dir)

        cnt_frame_drop.to_csv(os.path.join(
            percentage_tissue_dir, tissue + '_vec_freq.csv'))

        sns.set(style='whitegrid')
        plt.rcParams['font.weight'] = 'normal'
        print('Plotting ' + tissue + ' Boolean vector accumulation plot.')

        # g = sns.factorplot(
        #     x='cell_type', y='accumulation', data=cnt_frame_drop, kind='bar',
        #     size=7, aspect=1.75, color=color_dict[tissue])

        x = np.arange(1, len(cnt_frame_drop['accumulation'])+1, 1)
        g = plt.step(
            x, cnt_frame_drop['accumulation'], where='pre',
            color=color_dict[tissue])

        plt.xticks(x)

        ax = plt.gca()

        ax.set_ylim([0, 100])
        ax.set_xlabel(xlabel='immunophenotype', weight='bold')
        ax.set_ylabel(ylabel='% tissue coverage', weight='bold')

        ax.set_xticklabels(cnt_frame_drop['cell_type'])

        xlabels = [
            item.get_text() for item in ax.get_xticklabels()]

        xlabels_update = [i.replace(
                          'neg', '$^-$').replace('pos', '$^+$').replace(
                          'CD8a', 'CD8' + u'\u03B1') for i
                          in xlabels]

        ax.set_xticklabels(xlabels_update)

        for item in ax.get_xticklabels():
            item.set_rotation(90)
            item.set_weight('normal')

        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(accumulation_plots, tissue +
                    '_vector_accumulation' + '.pdf'))
        plt.close('all')

        sns.set(style='whitegrid')
        plt.rcParams['font.weight'] = 'normal'
        print('Plotting ' + tissue +
              ' Boolean vector percentage of total cells plot.')
        g = sns.barplot(
            x='cell_type', y='percentage', data=cnt_frame_drop,
            color=color_dict[tissue])

        xlabels = [
            item.get_text() for item in g.get_xticklabels()]

        xlabels_update = [xlabel.replace(
            'neg', '$^-$').replace('pos', '$^+$').replace(
            'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]

        g.set_xticklabels(xlabels_update)

        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_weight('normal')

        g.set()
        ax = plt.gca()
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, alpha=0.5)
        ax.set_xlabel(xlabel='immunophenotype', weight='bold')
        ax.set_ylabel(ylabel='% of tissue', weight='bold')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(percentage_tissue_dir, tissue +
                    '_vector_percentage' + '.pdf'))
        plt.close('all')
        print()

    sns.set(style='whitegrid')
    plt.rcParams['font.weight'] = 'normal'

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    print('Plotting stacked celltype tissue distribution.')
    colors = list(islice(cycle(
        list(color_dict.values())), None, len(percentage_stacked)))
    g = percentage_stacked_normal.iloc[:, :-2].plot(
        kind='bar', stacked=True, linewidth=0.0,
        figsize=(9, 7), color=colors, ax=ax1)
    ax1.set_title('immunophenotype tissue distribution',
                  y=1.05, size=20, fontweight='bold')
    xlabels = [
        item.get_text() for item in ax1.get_xticklabels()]

    xlabels_update = [xlabel.replace(
        'neg', '$^-$').replace('pos', '$^+$').replace(
        'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]

    ax1.set_xticklabels(xlabels_update)

    for item in ax1.get_xticklabels():
        item.set_rotation(90)
        item.set_weight('normal')

    ax1.set_xlabel(xlabel='immunophenotype',
                   size=15, weight='bold')
    ax1.set_ylabel(ylabel='% of tissue',
                   size=15, weight='bold')

    g.xaxis.grid(False)
    g.yaxis.grid(False)

    ax1.get_legend().remove()
    ax1.set_ylim(0.0, 1.0)

    # plot Shannon entropy
    print('Superimposing Shannon diversity index.')
    g = percentage_stacked_normal['entropy'].plot(ls='--', c='k', ax=ax2)

    xlabels = [
        item.get_text() for item in ax2.get_xticklabels()]

    xlabels_update = [xlabel.replace(
        'neg', '$^-$').replace('pos', '$^+$').replace(
        'CD8a', 'CD8' + u'\u03B1') for xlabel in xlabels]

    ax2.set_xticklabels(xlabels_update)

    for item in ax2.get_xticklabels():
        item.set_rotation(90)
        item.set_weight('normal')

    ax2.set_ylabel(ylabel='Shannon diversity index (H)',
                   size=15, weight='bold')

    g.xaxis.grid(False)
    g.yaxis.grid(False)

    for spine in ax2.spines.values():
        spine.set_edgecolor('k')

    # make custom legend
    legend_list = []
    for key, value in color_dict.items():
        line = mlines.Line2D([], [], color=value, marker='', markersize=30,
                             label=key)
        legend_list.append(line)
    legend_text_properties = {'weight': 'bold'}
    legend = plt.legend(handles=legend_list, prop=legend_text_properties)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(5.0)

    plt.tight_layout()
    plt.savefig(os.path.join(percentage_plots,
                'Shannon_entropy' + '.pdf'))
    plt.close('all')
    print()

    print()





# plot priority scores
def priority_scores():

    banner('RUNNING MODULE: priority_scores')

    os.chdir(PICKLE_DIR)
    # plot all p-values
    # pi_TOTAL_sig_dif = open(PO_TOTAL_SIG_DIF, 'rb')
    # TOTAL_sig_dif = pickle.load(pi_TOTAL_sig_dif)

    # plot only FDR-corrected q-values
    pi_sig_dif_FDRcorrected = open(
        PO_SIG_DIF_FDRCORRECTED, 'rb')
    sig_dif_FDRcorrected = pickle.load(
        pi_sig_dif_FDRcorrected)

    pi_vector_classification = open(PO_VECTOR_CLASSIFICATION, 'rb')
    vector_classification = pickle.load(pi_vector_classification)

    pair_grid_input = sig_dif_FDRcorrected.copy()

    pair_grid_input['dif'] = (pair_grid_input['gl261_percent'] -
                              pair_grid_input['naive_percent']).astype(float)

    pair_grid_input['weighted_fold_change'] = np.log2(
        (0.01 + pair_grid_input['gl261_percent']) /
        (0.01 + pair_grid_input['naive_percent']))

    pair_grid_input['name'] = pair_grid_input['tissue'].map(str) + \
        '_' + pair_grid_input['cell_type']

    time_sig = pair_grid_input.groupby(['tissue', 'cell_type']).size()
    time_sig = pd.DataFrame(time_sig)
    time_sig = time_sig.reset_index()
    time_sig['name'] = time_sig['tissue'].map(str) + \
        '_' + time_sig['cell_type']
    time_sig = time_sig.drop(['tissue', 'cell_type'], axis=1)
    time_sig.columns = ['time_sig', 'name']

    pair_grid_input = pd.merge(pair_grid_input, time_sig, on='name')

    pair_grid_input['name'] = pair_grid_input['time_point'].map(str) + \
        '_' + pair_grid_input['name']
    pair_grid_input = pair_grid_input.drop(
        ['time_point', 'tissue', 'cell_type', 'gl261_percent',
         'naive_percent'], axis=1)

    pair_grid_input['priority_score'] = (
        ((1-pair_grid_input['corrected_pvalue'])**3) *
        (abs(pair_grid_input['weighted_fold_change'] +
         pair_grid_input['dif']))) + pair_grid_input['time_sig']

    pair_grid_col_order = ['name', 'priority_score', 'weighted_fold_change',
                           'dif', 'corrected_pvalue', 'time_sig', 'statistic']
    pair_grid_input = pair_grid_input[pair_grid_col_order]
    pair_grid_input.sort_values('priority_score', ascending=False,
                                inplace=True)

    pair_grid_input['corrected_pvalue'] = -np.log10(
        pair_grid_input['corrected_pvalue'])

    color_list = []
    for name in pair_grid_input['name']:
        name = name.split('_', 2)[2]
        if name == 'unspecified':
            color_list.append('k')
        else:
            if vector_classification[name]['lineage'] == 'lymphoid':
                color_list.append('g')
            elif vector_classification[name]['lineage'] == 'myeloid':
                color_list.append('b')
            elif vector_classification[name]['lineage'] == 'other':
                color_list.append('k')
    pair_grid_input['color'] = color_list

    sns.set(style='whitegrid')
    plt.rcParams['font.weight'] = 'normal'
    print('Plotting priority score plot.')

    pair_grid_input['priority_score'] = np.log10(
        pair_grid_input['priority_score'])

    g = sns.PairGrid(
        pair_grid_input.sort_values('priority_score', ascending=False),
        x_vars=pair_grid_input.columns[1:6], y_vars='name',
        size=20, aspect=.17)

    g.map(sns.stripplot, size=17, orient='h',
          palette='Reds_r', edgecolor='gray', linewidth=1)

    g.set(xlabel='', ylabel='')
    titles = ['log10(priority_score)', 'weighted_fold-change',
              'difference (%)', '-log10(q-value)', 'time_significance rank']

    axes = g.axes

    # for item in g.axes[0, 0].get_yticklabels():
    #
    #     name = item.get_text()
    #     name = name.split('_', 2)[2]
    #     if name != 'unspecified':
    #         if vector_classification[name]['lineage'] == 'lymphoid':
    #             item.set_color('g')
    #         elif vector_classification[name]['lineage'] == 'myeloid':
    #             item.set_color('b')
    #         elif vector_classification[name]['lineage'] == 'other':
    #             item.set_color('k')

    ax_list = np.ndarray.tolist(axes)
    ax_list = [item for sublist in ax_list for item in sublist]
    for ax in ax_list:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.1f'))
        for item in ax.get_xticklabels():
                item.set_size(15)

    ylabels = [
        item.get_text() for item in g.axes[0, 0].get_yticklabels()]

    ylabels_update = [ylabel.replace(
        'neg', '$^-$').replace('pos', '$^+$').replace(
        'CD8a', 'CD8' + u'\u03B1').replace('7', 'early').replace(
            '14', 'middle').replace('30', 'late') for ylabel in ylabels]

    ylabels_update = [ylabel.rsplit('_', 3)[2] + '_' +
                      ylabel.rsplit('_', 3)[1] + '_' +
                      ylabel.rsplit('_', 3)[0] for ylabel in ylabels_update]

    g.axes[0, 0].set_yticklabels(ylabels_update, size=22, weight='normal')

    axes[0, 0].set_xlim()
    axes[0, 1].set_xlim()
    axes[0, 1].axvline(x=0.0, linewidth=1.5, linestyle='dashed',
                       color='k', alpha=0.8)
    axes[0, 2].set_xlim()
    axes[0, 2].axvline(x=0.0, linewidth=1.5, linestyle='dashed',
                       color='k', alpha=0.8)
    axes[0, 3].set_xlim()
    axes[0, 4].set_xlim()

    for ax, title in zip(g.axes.flat, titles):
        ax.set(title=title)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        l = ax.get_title()
        ax.set_title(l, size=18, y=1.005, fontweight='normal')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(ORTHANT_DIR, 'priority_scores' + '.pdf'))
    plt.close('all')

    os.chdir(PICKLE_DIR)
    po_pair_grid_input = open(PO_PAIR_GRID_INPUT, 'wb')
    pickle.dump(pair_grid_input, po_pair_grid_input)
    po_pair_grid_input.close()
    print()



# plot aggregate figure
def aggregate_celltype_fig():

    banner('RUNNING MODULE: aggregate_celltype_fig')

    os.chdir(PICKLE_DIR)
    dashboard_shlf = shelve.open(DASHBOARD_SHLF, writeback=True)

    pi_aggregate_data_dir = open(PO_AGGREGATE_DATA_DIR, 'rb')
    aggregate_data_dir = pickle.load(pi_aggregate_data_dir)

    # use to plot uncorrected p-values
    # pi_TOTAL_sig_dif_mag_ratio = open('TOTAL_sig_dif_mag_ratio.pickle', 'rb')
    # TOTAL_sig_dif_mag_ratio = pickle.load(pi_TOTAL_sig_dif_mag_ratio)

    # use to plot FDR-corrected q-values
    pi_sig_dif_FDRcorrected_ratio = open(PO_SIG_DIF_FDRCORRECTED_RATIO, 'rb')
    sig_dif_FDRcorrected_ratio = pickle.load(pi_sig_dif_FDRcorrected_ratio)

    pi_pair_grid_input = open(PO_PAIR_GRID_INPUT, 'rb')
    pair_grid_input = pickle.load(pi_pair_grid_input)

    pi_color_dict = open(PO_COLOR_DICT, 'rb')
    color_dict = pickle.load(pi_color_dict)

    pi_channel_list_update_dict = open(PO_CHANNEL_LIST_UPDATE_DICT, 'rb')
    channel_list_update_dict = pickle.load(pi_channel_list_update_dict)

    pi_landmark_pops = open(PO_LANDMARK_POPS, 'rb')
    landmark_pops = pickle.load(pi_landmark_pops)

    pi_vector_classification = open(PO_VECTOR_CLASSIFICATION, 'rb')
    vector_classification = pickle.load(pi_vector_classification)

    os.chdir(aggregate_data_dir)
    overall = pd.read_csv(OVERALL_CSV)

    aggregate_fig_dir = os.path.join(ORTHANT_DIR, 'aggregate_figure')
    os.makedirs(aggregate_fig_dir)
    os.chdir(aggregate_fig_dir)

    def sort_preserve_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def make_ticklabels_invisible(fig):
        for i, ax in enumerate(fig.axes):
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_visible(False)

    def pvalue_chars(sig_chr):
        if 0.01 < sig_chr['corrected_pvalue'] <= 0.05:
            val = '*'
        elif 0.001 < sig_chr['corrected_pvalue'] <= 0.01:
            val = '**'
        elif sig_chr['corrected_pvalue'] <= 0.001:
            val = '***'
        return val

    temp_sorted_grid = list(pair_grid_input['name'])
    rank_sorted_grid = sort_preserve_order(
        [a.split('_')[-1] for a in temp_sorted_grid])

    order_of_dict = sorted(
        dashboard_shlf, key=lambda x: dashboard_shlf[x]['percent'],
        reverse=True)
    order_of_dict.remove('unspecified')
    # rank_sorted_grid.remove('unspecified')

    # vertical orientation(95, 150)
    # horizontal orientation(190, 75)

    # vertical orientation(10, 5)
    # horizontal orientation(5, 10)

    max_radius = math.sqrt(dashboard_shlf[order_of_dict[0]]['percent'])/5
    channel_list_update_dict['cd8a'] = channel_list_update_dict['cd8']
    del channel_list_update_dict['cd8']

    ax_dict = {'ax_tbuf': [], 'ax_bbuf': [], 'ax_lbuf': [], 'ax_rbuf': [],
               'ax0_1': [], 'ax0_2': [], 'ax0_3': [], 'ax0': [], 'ax1': [],
               'ax2': [], 'ax3': [], 'ax4': [], 'ax5': [], 'ax6': [],
               'ax7': [], 'ax8': [], 'ax9': [], 'ax10': [],
               'ax11': [], 'ax12': []}
    cbar_dict_mag = {}
    cbar_dict_ratio = {}

    # alternative order_of_dict
    # order_of_dict = ['B', 'DPT', 'CD4posT', 'CD8aposT', 'PMN', 'CD8aposISPT',
    #                  'Precursor', 'Ly6CposCD8aposT', 'Mono', 'DNT',
    #                  'Ly6CposMac', 'Ly6CposCD4posT', 'F480posPMN', 'Mac',
    #                  'F480posB', 'CD11bnegMono', 'Ly6CposDNT', 'CD45negPMN',
    #                  'Ly6CnegPMN', 'Ly6CposB', 'CD3eposDPT', 'Ly6CnegMono',
    #                  'Non-immune', 'CD45negB', 'F480posLy6CnegPMN',
    #                  'CD11bnegMac', 'NK', 'CD45negDPT',
    #                  'CD45negCD8aposISPT', 'B220posCD8aposT', 'DC']

    def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                           savefig_kw={}):
        """Save a figure with raster and vector components

        This function lets you specify which objects to rasterize at the export
        stage, rather than within each plotting call. Rasterizing certain
        components of a complex figure can significantly reduce file size.

        Inputs
        ------
        fname : str
            Output filename with extension
        rasterize_list : list (or object)
            List of objects to rasterize (or a single object to rasterize)
        fig : matplotlib figure object
            Defaults to current figure
        dpi : int
            Resolution (dots per inch) for rasterizing
        savefig_kw : dict
            Extra keywords to pass to matplotlib.pyplot.savefig

        If rasterize_list is not specified, then all contour, pcolor, and
        collects objects (e.g., ``scatter, fill_between`` etc) will be
        rasterized

        Note: does not work correctly with round=True in Basemap

        Example
        -------
        Rasterize the contour, pcolor, and scatter plots, but not the line

        >>> import matplotlib.pyplot as plt
        >>> from numpy.random import random
        >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
        >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
        >>> cax1 = ax1.contourf(Z)
        >>> cax2 = ax2.scatter(X, Y, s=Z)
        >>> cax3 = ax3.pcolormesh(Z)
        >>> cax4 = ax4.plot(Z[:, 0])
        >>> rasterize_list = [cax1, cax2, cax3]
        >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
        """

        # Behave like pyplot and act on current figure if
        # no figure is specified
        fig = plt.gcf() if fig is None else fig

        # Need to set_rasterization_zorder in order for rasterizing to work
        zorder = -5  # Somewhat arbitrary, just ensuring less than 0

        if rasterize_list is None:
            # Have a guess at stuff that should be rasterised
            types_to_raster = ['QuadMesh', 'Contour', 'collections']
            rasterize_list = []

            print("""
            No rasterize_list specified, so the following objects will
            be rasterized: """)
            # Get all axes, and then get objects within axes
            for ax in fig.get_axes():
                for item in ax.get_children():
                    if any(x in str(item) for x in types_to_raster):
                        rasterize_list.append(item)
            print('\n'.join([str(x) for x in rasterize_list]))
        else:
            # Allow rasterize_list to be input as an object to rasterize
            if type(rasterize_list) != list:
                rasterize_list = [rasterize_list]

        for item in rasterize_list:

            # Whether or not plot is a contour plot is important
            is_contour = (isinstance(item, matplotlib.contour.QuadContourSet)
                          or
                          isinstance(item, matplotlib.tri.TriContourSet))

            # Whether or not collection of lines
            # This is commented as we seldom want to rasterize lines
            # is_lines = isinstance(
            #    item, matplotlib.collections.LineCollection)

            # Whether or not current item is list of patches
            all_patch_types = tuple(
                x[1] for x in getmembers(matplotlib.patches, isclass))
            try:
                is_patch_list = isinstance(item[0], all_patch_types)
            except TypeError:
                is_patch_list = False

            # Convert to rasterized mode and then change zorder properties
            if is_contour:
                curr_ax = item.ax.axes
                curr_ax.set_rasterization_zorder(zorder)
                # For contour plots, need to set each part of the contour
                # collection individually
                for contour_level in item.collections:
                    contour_level.set_zorder(zorder + 100)
                    contour_level.set_rasterized(True)
            elif is_patch_list:
                # For list of patches, need to set zorder for each patch
                for patch in item:
                    curr_ax = patch.axes
                    curr_ax.set_rasterization_zorder(zorder)
                    patch.set_zorder(zorder + 100)
                    patch.set_rasterized(True)
            else:
                # For all other objects, we can just do it all at once
                curr_ax = item.axes
                curr_ax.set_rasterization_zorder(zorder)
                item.set_rasterized(True)
                item.set_zorder(zorder + 100)

        # dpi is a savefig keyword argument, but treat it as special]
        # since it is important to this function
        if dpi is not None:
            savefig_kw['dpi'] = dpi

        # Save resulting figure
        fig.savefig(fname, **savefig_kw)

    def dash(fig, ss, i):

        celltype = order_of_dict[i]

        sig_chr = sig_dif_FDRcorrected_ratio[
            sig_dif_FDRcorrected_ratio['cell_type'] == celltype]
        if not sig_chr.empty:
            sig_chr_2 = sig_chr.copy()
            sig_chr_2['character'] = sig_chr_2.apply(pvalue_chars, axis=1)
        else:
            sig_chr_2 = None
            sig_chr_dif_dict = {}

        data_to_consider = dashboard_shlf[celltype]
        radius = math.sqrt(data_to_consider['percent'])/5
        inner = gridspec.GridSpecFromSubplotSpec(
            50, 100, subplot_spec=ss, wspace=0.05, hspace=0.05)

        ax0 = plt.Subplot(fig, inner[0:10, 0:100])

        # dashboard box outline
        ax0.add_patch(matplotlib.patches.Rectangle(
            (0, -4.02), 1, 5.02, fill=None, lw=1.5, color='grey',
            alpha=1, clip_on=False))

        # celltype name
        ax0.text(0.01, 0.92, celltype.replace(
            'neg', '$^-$').replace('pos', '$^+$'),
             horizontalalignment='left', verticalalignment='top',
             fontsize=50, color='k', stretch=0, fontname='Arial',
             fontweight='bold')

        for tl in ax0.get_xticklabels() + ax0.get_yticklabels():
            tl.set_visible(False)

        # priority score
        ax0.text(0.97, 0.48, 'priority', horizontalalignment='right',
                 verticalalignment='center', fontweight='bold',
                 fontsize=25, color='k')
        ax0.text(0.97, 0.34, 'rank', horizontalalignment='right',
                 verticalalignment='center', fontweight='bold',
                 fontsize=25, color='k')
        if celltype in rank_sorted_grid:
            ax0.text(0.98, -0.05, str(
                rank_sorted_grid.index(celltype)+1),
                horizontalalignment='right', verticalalignment='center',
                fontweight='bold', fontsize=90, color='teal')
        else:
            ax0.text(0.98, -0.05, 'n/s',
                     horizontalalignment='right', verticalalignment='center',
                     fontweight='bold', fontsize=72, color='teal')

        how_to_sort = {'CD45': 0, 'B220': 1, 'CD11b': 2, 'CD11c': 3, 'CD3e': 4,
                       'CD4': 5, 'CD49b': 6, 'CD8a': 7, 'F480': 8, 'Ly6C': 9,
                       'Ly6G': 10}

        vector_classification[celltype]['signature'] = sorted(
            vector_classification[celltype]['signature'],
            key=lambda d: how_to_sort[d])

        # landmark info
        if celltype in landmark_pops:
            ax0.text(0.013, 0.53, 'landmark population',
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=25, fontweight='bold', color='blue')
            ax0.text(0.013, 0.33, 'signature: ' +
                     ', '.join(
                        vector_classification[celltype]['signature']),
                     horizontalalignment='left', fontweight='bold',
                     verticalalignment='top', fontsize=22, color='k')

        # lineage info
        ax0_1 = plt.Subplot(fig, inner[0:4, 0:100])
        ax0_2 = plt.Subplot(fig, inner[0:4, 0:100])
        ax0_3 = plt.Subplot(fig, inner[0:4, 0:100])

        if vector_classification[celltype]['lineage'] == 'myeloid':
            ax0.text(0.98, 0.88, 'myeloid', horizontalalignment='right',
                     verticalalignment='top', fontweight='bold', fontsize=33,
                     color='k')
            bar_list = []
            for i in list(range(0, 6)):
                bar_list.append(list(range(0, 100)))
            bar = np.array(bar_list)
            ax0_1.imshow(bar, cmap=plt.cm.Blues, interpolation='bicubic',
                         vmin=30, vmax=200)
            ax0_1.grid(False)
            for item in ax0_1.get_xticklabels():
                item.set_visible(False)
            for item in ax0_1.get_yticklabels():
                item.set_visible(False)
            fig.add_subplot(ax0_1)
            ax_dict['ax0_1'].append(ax0_1)

        elif vector_classification[celltype]['lineage'] == 'lymphoid':
            ax0.text(0.98, 0.88, 'lymphoid', horizontalalignment='right',
                     verticalalignment='top', fontweight='bold', fontsize=33,
                     color='k')
            bar_list = []
            for i in list(range(0, 6)):
                bar_list.append(list(range(0, 100)))
            bar = np.array(bar_list)
            ax0_2.imshow(bar, cmap=plt.cm.Greens, interpolation='bicubic',
                         vmin=30, vmax=200)
            ax0_2.grid(False)
            for item in ax0_2.get_xticklabels():
                item.set_visible(False)
            for item in ax0_2.get_yticklabels():
                item.set_visible(False)
            fig.add_subplot(ax0_2)
            ax_dict['ax0_2'].append(ax0_2)

        elif vector_classification[celltype]['lineage'] == 'other':
            ax0.text(0.98, 0.88, 'other', horizontalalignment='right',
                     verticalalignment='top', fontweight='bold', fontsize=33,
                     color='k')
            bar_list = []
            for i in list(range(0, 6)):
                bar_list.append(list(range(0, 100)))
            bar = np.array(bar_list)
            ax0_3.imshow(bar, cmap=plt.cm.Oranges, interpolation='bicubic',
                         vmin=30, vmax=200)
            ax0_3.grid(False)
            for item in ax0_3.get_xticklabels():
                item.set_visible(False)
            for item in ax0_3.get_yticklabels():
                item.set_visible(False)
            fig.add_subplot(ax0_3)
            ax_dict['ax0_3'].append(ax0_3)

        # make titles for replicate data
        ax0.text(0.657, -0.345, 'early',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=16, fontweight='bold', color='k')
        naive_7 = ax0.text(0.627, -0.47, 'naive',
                           horizontalalignment='left',
                           verticalalignment='center',
                           fontsize=13, fontweight='bold')
        naive_7.set_color('b')
        gl261_7 = ax0.text(0.69, -0.47, 'gl261',
                           horizontalalignment='left',
                           verticalalignment='center',
                           fontsize=13, fontweight='bold')
        gl261_7.set_color('mediumaquamarine')

        ax0.text(0.773, -0.345, 'middle',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=16, fontweight='bold', color='k')
        naive_14 = ax0.text(0.754, -0.47, 'naive',
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=13, fontweight='bold')
        naive_14.set_color('b')
        gl261_14 = ax0.text(0.815, -0.47, 'gl261',
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=13, fontweight='bold')
        gl261_14.set_color('mediumaquamarine')

        ax0.text(0.912, -0.345, 'late',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=16, fontweight='bold', color='k')
        naive_30 = ax0.text(0.88, -0.47, 'naive',
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=13, fontweight='bold')
        naive_30.set_color('b')
        gl261_30 = ax0.text(0.94, -0.47, 'gl261',
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=13, fontweight='bold')
        gl261_30.set_color('mediumaquamarine')

        # make legend for pvalue characters
        ax0.text(0.14, -1.92, '(*) 0.01 < p <= 0.05',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=12, color='k', stretch=0, fontname='Arial',
                 fontweight='bold')

        ax0.text(0.272, -1.92, '(**) 0.001 < p <= 0.01',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=12, color='k', stretch=0, fontname='Arial',
                 fontweight='bold')

        ax0.text(0.42, -1.92, '(***) p <= 0.001',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=12, color='k', stretch=0, fontname='Arial',
                 fontweight='bold')

        ax0.axis('off')
        fig.add_subplot(ax0)
        ax_dict['ax0'].append(ax0)

        # pop. size
        ax1 = plt.Subplot(fig, inner[2:15, 72:85])
        patches, texts = ax1.pie(
            [100, 0], radius=radius, shadow=False,
            startangle=90, colors=[(0.34, 0.35, 0.38)])

        for w in patches:
            w.set_linewidth(0.0)
            w.set_edgecolor('k')

        ax1.add_patch(matplotlib.patches.Rectangle(
            (-max_radius*1.2, -max_radius*1.2), max_radius*1.2*2,
            max_radius*1.2*2, fill=None, alpha=1))
        ttl_pop = ax1.set_title(
            'pop. size', fontsize=18, fontweight='bold')
        ttl_pop.set_position([-0.31, 0.69])
        ax1.text(-1.5, 0.50, str(round(data_to_consider['percent'], 2)) + '%',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=16, color='k', stretch=0, fontname='Arial',
                 fontweight='bold')

        ax1.axis('equal')
        fig.add_subplot(ax1)
        ax_dict['ax1'].append(ax1)

        # tissue dist.
        ax2 = plt.Subplot(fig, inner[2:14, 52:64])
        labels = data_to_consider['data'].index.tolist()
        colors = [color_dict[x] for x in labels]
        patches, texts = ax2.pie(
            data_to_consider['data'], shadow=False,
            colors=colors, startangle=90, radius=1.0)

        for w in patches:
            w.set_linewidth(0.25)
            w.set_edgecolor('k')

        ax2.add_patch(matplotlib.patches.Rectangle(
            (-1, -1), 2, 2, fill=None, alpha=0))
        ttl_dist = ax2.set_title('tissue dist.',
                                 fontsize=18, fontweight='bold')
        ttl_dist.set_position([-0.27, 0.67])
        ax2.axis('equal')
        fig.add_subplot(ax2)
        ax_dict['ax2'].append(ax2)

        # scatter boxplots
        sns.set(style='whitegrid')
        ax3 = plt.Subplot(fig, inner[13:27, 5:11])
        df = data_to_consider['box-fsc-ssc']
        temp = df[df.columns[-4:]].copy(deep='True')
        temp_ssc = temp[['ssc', 'cell_type', 'status']].copy(deep='True')
        temp_fsc = temp[['fsc', 'cell_type', 'status']].copy(deep='True')
        temp_ssc['channel_name'] = 'ssc'
        temp_fsc['channel_name'] = 'fsc'
        temp_ssc = temp_ssc.rename(index=str, columns={"ssc": "data"})
        temp_fsc = temp_fsc.rename(index=str, columns={"fsc": "data"})
        final_temp = temp_fsc.append(temp_ssc, ignore_index=True)
        final_temp['data'] = final_temp['data'].astype('float')
        final_temp['channel_name'] = final_temp['channel_name'].replace(
            {'fsc': 'FSC', 'ssc': 'SSC'})
        g = sns.boxplot(
            x='channel_name', y='data', hue='status', data=final_temp, ax=ax3,
            palette={'gl261': 'mediumaquamarine', 'naive': 'b'})
        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(15)
            item.set_fontweight('bold')
            item.set_position([0, 0])
        for item in g.get_yticklabels():
            item.set_size(9)
            item.set_fontweight('bold')
        ax3.set_ylim(0, 250000)
        fig.canvas.draw()
        ax3.set_yticklabels(ax3.get_yticks())
        labels = ax3.get_yticklabels()
        ylabels = [float(label.get_text()) for label in labels]
        ylabels = ['%.1E' % Decimal(s) for s in ylabels if 0 <= s <= 250000]
        g.set_yticklabels(ylabels)
        ax3.set_xlabel('')
        ax3.set_ylabel('')
        legend_properties = {'weight': 'bold', 'size': 16}
        ax3.legend(loc=(-0.80, 1.04), prop=legend_properties)
        ax3.axhline(y=0.0, color='darkgray', linewidth=5.0,
                    linestyle='-', zorder=1, alpha=1.0)
        lines = g.lines[:-1]
        scatter_raster = lines
        fig.add_subplot(ax3)
        ax_dict['ax3'].append(ax3)

        # channel boxplots
        sns.set(style='whitegrid')
        ax4 = plt.Subplot(fig, inner[30:46, 3:52])
        data_to_consider['box'] = data_to_consider['box'].replace(
            {'channel': channel_list_update_dict})
        g = sns.boxplot(
            x='channel', y='value', hue='status',
            data=data_to_consider['box'], ax=ax4,
            palette={'gl261': 'mediumaquamarine',
                     'naive': 'b'},
            order=how_to_sort)
        for item in g.get_xticklabels():
            item.set_rotation(90)
            item.set_size(15)
            item.set_fontweight('bold')
            item.set_position([0, 0.06])
        for item in g.get_yticklabels():
            item.set_size(12)
            item.set_fontweight('bold')
        ax4.set_ylim(-3.5, 3.5)
        ax4.set_xlabel('')
        ax4.set_ylabel('')
        ax4.legend_.remove()
        ax4.axhline(y=0.0, color='darkgray', linewidth=3.0,
                    linestyle='-', zorder=1, alpha=1.0)
        lines = g.lines[:-1]
        channel_raster = lines
        fig.add_subplot(ax4)
        ax_dict['ax4'].append(ax4)

        # get t-n difference significance characters
        character_dif_frame = data_to_consider['sig_dif_all'].copy()
        if sig_chr_2 is not None:
            sig_chr_dif_dict = {
                i: ('{0:.6g}'.format(i)) for i in sig_chr_2['dif']}

            for i in data_to_consider['sig_dif_all'].unstack():
                f = ('{0:.6g}'.format(i))
                if f in sig_chr_dif_dict.values():
                    h = [k for k, v in sig_chr_dif_dict.items() if v == f]
                    char = sig_chr_2.loc[
                        sig_chr_2['dif'] == h[0], 'character']
                    char.reset_index(drop=True, inplace=True)
                    char = char[0]
                    character_dif_frame.replace(
                        to_replace=character_dif_frame[
                            character_dif_frame == h[0]],
                        value=char, inplace=True)
                else:
                    character_dif_frame.replace(
                        to_replace=character_dif_frame[
                            character_dif_frame == i],
                        value='', inplace=True)
        else:
            for i in data_to_consider['sig_dif_all'].unstack():
                character_dif_frame.replace(
                    to_replace=character_dif_frame[character_dif_frame == i],
                    value='', inplace=True)

        # t-n difference
        ax5 = plt.Subplot(fig, inner[12:27, 11:33])
        g = sns.heatmap(data_to_consider['sig_dif_all'], square=True, ax=ax5,
                        vmin=-10, vmax=10, linecolor='w', linewidths=2.0,
                        cbar=True, annot=character_dif_frame, fmt='',
                        cmap='cividis', center=0.0,
                        xticklabels=True, yticklabels=True,
                        annot_kws={'size': 20, 'fontweight': 'bold'})

        cbar_ax_mag = plt.gcf().axes[-1]
        cbar_dict_mag[celltype] = cbar_ax_mag
        g.set_xlabel('')
        g.set_ylabel('')
        ylabels = [item.get_text() for item in g.get_yticklabels()]
        for t in range(len(ylabels)):
            if ylabels[t] == 'blood':
                ylabels[t] = 'Bl'
            elif ylabels[t] == 'spleen':
                ylabels[t] = 'Sp'
            elif ylabels[t] == 'nodes':
                ylabels[t] = 'Nd'
            elif ylabels[t] == 'marrow':
                ylabels[t] = 'Mw'
            elif ylabels[t] == 'thymus':
                ylabels[t] = 'Th'
        g.set_yticklabels(ylabels)
        for item in g.get_yticklabels():
            tissue_name = str(item)
            item.set_rotation(0)
            item.set_size(14)
            item.set_fontweight('bold')
            item.set_position([0.02, 0.0])
            if 'Bl' in tissue_name:
                item.set_color('r')
            if 'Mw' in tissue_name:
                item.set_color('b')
            if 'Nd' in tissue_name:
                item.set_color('g')
            if 'Sp' in tissue_name:
                item.set_color('m')
            if 'Th' in tissue_name:
                item.set_color('y')
        xlabels = [item.get_text() for item in g.get_xticklabels()]
        for t in range(len(xlabels)):
            if xlabels[t] == '7':
                xlabels[t] = 'early'
            elif xlabels[t] == '14':
                xlabels[t] = 'middle'
            if xlabels[t] == '30':
                xlabels[t] = 'late'
        g.set_xticklabels(xlabels)
        for item in g.get_xticklabels():
            item.set_rotation(0)
            item.set_size(14)
            item.set_fontweight('bold')
        ttl_dif = ax5.set_title('gl261-naive', fontsize=20, fontweight='bold')
        ttl_dif.set_position([0.35, 1.01])
        fig.add_subplot(ax5)
        ax_dict['ax5'].append(ax5)

        # get log2(t/n) ratio significance characters
        character_ratio_frame = data_to_consider['sig_ratio_all'].copy()
        if sig_chr_2 is not None:
            sig_chr_ratio_dict = {
                i: ('{0:.6g}'.format(i)) for i in
                sig_chr_2['ratio']}

            for i in data_to_consider['sig_ratio_all'].unstack():
                f = ('{0:.6g}'.format(i))
                if f in sig_chr_ratio_dict.values():
                    h = [k for k, v in sig_chr_ratio_dict.items() if v == f]
                    char = sig_chr_2.loc[
                        sig_chr_2['ratio'] == h[0], 'character']
                    char.reset_index(drop=True, inplace=True)
                    char = char[0]
                    character_ratio_frame.replace(
                        to_replace=character_ratio_frame[
                            character_ratio_frame == h[0]],
                        value=char, inplace=True)
                else:
                    character_ratio_frame.replace(
                        to_replace=character_ratio_frame[
                            character_ratio_frame == i],
                        value='', inplace=True)
        else:
            for i in data_to_consider['sig_ratio_all'].unstack():
                character_ratio_frame.replace(
                    to_replace=character_ratio_frame[
                        character_ratio_frame == i],
                    value='', inplace=True)

        # log2(t/n) ratio
        ax6 = plt.Subplot(fig, inner[12:27, 33:53])
        g = sns.heatmap(data_to_consider['sig_ratio_all'], square=True, ax=ax6,
                        vmin=-4, vmax=4, linecolor='w', linewidths=2.0,
                        cbar=True, annot=character_ratio_frame, fmt='',
                        cmap='cividis', center=0.0,
                        xticklabels=True, yticklabels=True,
                        annot_kws={'size': 20, 'fontweight': 'bold'})

        cbar_ax_ratio = plt.gcf().axes[-1]
        cbar_dict_ratio[celltype] = cbar_ax_ratio
        g.set_xlabel('')
        g.set_ylabel('')
        g.set_yticklabels([])
        xlabels = [item.get_text() for item in g.get_xticklabels()]
        for t in range(len(xlabels)):
            if xlabels[t] == '7':
                xlabels[t] = 'early'
            elif xlabels[t] == '14':
                xlabels[t] = 'middle'
            if xlabels[t] == '30':
                xlabels[t] = 'late'
        g.set_xticklabels(xlabels)
        for item in g.get_xticklabels():
            item.set_rotation(0)
            item.set_size(14)
            item.set_fontweight('bold')
        ttl_ratio = ax6.set_title('log' + '$_2$' + '(gl261/naive)',
                                  fontsize=20, fontweight='bold')
        ttl_ratio.set_position([0.55, 1.01])
        fig.add_subplot(ax6)
        ax_dict['ax6'].append(ax6)

        # replicate percent composition
        ax7 = plt.Subplot(fig, inner[16:21, 61:99])
        ax8 = plt.Subplot(fig, inner[23:28, 61:99])
        ax9 = plt.Subplot(fig, inner[30:35, 61:99])
        ax10 = plt.Subplot(fig, inner[37:42, 61:99])
        ax11 = plt.Subplot(fig, inner[44:49, 61:99])

        sns.set(style='whitegrid')

        hue_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5, 5, 5]

        colors = {0: 'b', 1: 'mediumaquamarine', 2: 'b',
                  3: 'mediumaquamarine', 4: 'b', 5: 'mediumaquamarine'}

        sns.barplot(data_to_consider['x_final_rep_data'],
                    data_to_consider['blood_rep_data'], hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax7)
        ax7.legend_.remove()

        sns.barplot(data_to_consider['x_final_rep_data'],
                    data_to_consider['marrow_rep_data'], hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax8)
        ax8.legend_.remove()

        sns.barplot(data_to_consider['x_final_rep_data'],
                    data_to_consider['nodes_rep_data'], hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax9)
        ax9.legend_.remove()

        sns.barplot(data_to_consider['x_final_rep_data'],
                    data_to_consider['spleen_rep_data'], hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax10)
        ax10.legend_.remove()

        sns.barplot(data_to_consider['x_final_rep_data'],
                    data_to_consider['thymus_rep_data'], hue=hue_list,
                    palette=colors, linewidth=0.25, edgecolor='b', ax=ax11)
        ax11.legend_.remove()

        for ax, tissue in zip([ax7, ax8, ax9, ax10, ax11],
                              sorted(overall['tissue'].unique())):
            ax.set_ylabel('% composition', size=10, fontweight='bold')
            # ax.set_ylim(0, 0.5 * math.ceil(
            #     2.0 * max(data_to_consider['y_overall_rep_data'])))
            ax.set_ylim(
                0, max(data_to_consider['y_overall_rep_data']))
            ax.tick_params(axis='y', which='both', length=0)
            ax.zorder = 1
            for item in ax.get_yticklabels():
                item.set_rotation(0)
                item.set_size(10)
                item.set_fontweight('bold')
            for item in ax.get_xticklabels():
                item.set_visible(False)
            ax12 = ax.twinx()
            ax12.set_ylim(0, max(data_to_consider['y_overall_rep_data']))
            ax12.set_yticklabels([])
            ax12.set_ylabel(tissue, color=color_dict[tissue],
                            fontweight='bold', size=18)
            ax12.yaxis.set_label_coords(-0.13, .5)
            ax12.tick_params(axis='y', which='both', length=0)

            if tissue == 'blood':
                for item in ax.get_xticklabels():
                    if 'naive' in str(item):
                        item.set_color('k')
                    if 'gl261' in str(item):
                        item.set_color('k')
                    item.set_rotation(0)
                    item.set_size(8)
                    item.set_fontweight('bold')
                    item.set_visible(True)
                    item.set_position([0, 1.2])
                xlabels = [('' + item.get_text()[-1])
                           for item in ax.get_xticklabels()]
                ax.set_xticklabels(xlabels)

            for n, bar in enumerate(ax.patches):
                width = bar.get_width()
                bar.set_width(width*5)
                if 48 < n < 96:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.13)
                elif 96 < n < 144:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.23)
                elif 144 < n < 192:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.38)
                elif 192 < n < 240:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.45)
                elif 240 < n < 288:
                    bar_coord = bar.get_x()
                    bar.set_x(bar_coord - 0.58)

        fig.add_subplot(ax7)
        ax_dict['ax7'].append(ax7)
        fig.add_subplot(ax8)
        ax_dict['ax8'].append(ax8)
        fig.add_subplot(ax9)
        ax_dict['ax9'].append(ax9)
        fig.add_subplot(ax10)
        ax_dict['ax10'].append(ax10)
        fig.add_subplot(ax11)
        ax_dict['ax11'].append(ax11)

        for x in [15.45, 31.48]:
            ax7.axvline(x=x, ymin=0, ymax=1.48, c='grey',
                        linewidth=1, ls='--', zorder=3, clip_on=False)
            ax8.axvline(x=x, ymin=0, ymax=1.41, c='grey',
                        linewidth=1, ls='--', zorder=3, clip_on=False)
            ax9.axvline(x=x, ymin=0, ymax=1.41, c='grey',
                        linewidth=1, ls='--', zorder=3, clip_on=False)
            ax10.axvline(x=x, ymin=0, ymax=1.41, c='grey',
                         linewidth=1, ls='--', zorder=3, clip_on=False)
            ax11.axvline(x=x, ymin=0, ymax=1.41, c='grey',
                         linewidth=1, ls='--', zorder=3, clip_on=False)

        # save individual dashboard
        ax_list_replicates = []
        ax_list_else = []
        for key, value in ax_dict.items():
            if key in ['ax7', 'ax8', 'ax9', 'ax10', 'ax11']:
                for j in value:
                    ax_list_replicates.append(j)
            elif key in ['ax0', 'ax0_1', 'ax0_2', 'ax0_3', 'ax1', 'ax2', 'ax3',
                         'ax4', 'ax5', 'ax6']:
                for j in value:
                    ax_list_else.append(j)

        # show all four spines around replicate plots
        for ax in ax_list_replicates:
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)

        # do not show spines around the other axes
        for ax in ax_list_else:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for value in cbar_dict_mag.values():
            value.yaxis.tick_right()
            for item in value.get_yticklabels():
                item.set_size(13)
                item.set_fontweight('bold')

        for value in cbar_dict_ratio.values():
            value.yaxis.tick_right()
            for item in value.get_yticklabels():
                item.set_size(13)
                item.set_fontweight('bold')

        ax_list_replicates = []
        ax_list_else = []
        for key, value in ax_dict.items():
            if key in ['ax7', 'ax8', 'ax9', 'ax10', 'ax11']:
                for j in value:
                    ax_list_replicates.append(j)
            elif key in ['ax0', 'ax0_1', 'ax0_2', 'ax0_3', 'ax1', 'ax2', 'ax3',
                         'ax4', 'ax5', 'ax6']:
                for j in value:
                    ax_list_else.append(j)

        # show all four spines around replicate plots
        for ax in ax_list_replicates:
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)

        # do not show spines around the other axes
        for ax in ax_list_else:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for value in cbar_dict_mag.values():
            value.yaxis.tick_right()
            for item in value.get_yticklabels():
                item.set_size(13)
                item.set_fontweight('bold')

        for value in cbar_dict_ratio.values():
            value.yaxis.tick_right()
            for item in value.get_yticklabels():
                item.set_size(13)
                item.set_fontweight('bold')

        return(scatter_raster, channel_raster)

    for i in range(len(order_of_dict)):
        print('Plotting the ' + order_of_dict[i] + ' dashboard.')
        fig = plt.figure(figsize=(19, 15))
        outer = gridspec.GridSpec(1, 1)
        scatter_raster, channel_raster = dash(fig, outer[0], i)
        rasterize_list = scatter_raster + channel_raster
        rasterize_and_save(
            '%s_dashboard.pdf' % order_of_dict[i],
            rasterize_list, fig=fig, dpi=300,
            savefig_kw={'bbox_inches': 'tight'})
        plt.close('all')

    print('Plotting the combined dashboards.')
    fig = plt.figure(figsize=(190, 45))
    outer = gridspec.GridSpec(3, 10, wspace=0.1, hspace=0.1)
    rasterize_list = []
    for i, oss in enumerate(outer):
        scatter_raster, channel_raster = dash(fig, oss, i)
        rasterize_list += scatter_raster + channel_raster
    rasterize_and_save(
        'combined_dashboards.pdf',
        rasterize_list, fig=fig, dpi=200, savefig_kw={'bbox_inches': 'tight'})
    plt.close('all')


    dashboard_shlf.close()
