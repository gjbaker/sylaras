import os
import sys
import re
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import collections

import flowkit as fk

from sklearn.preprocessing import MinMaxScaler
from seaborn import kdeplot

from selenium.webdriver import Firefox, FirefoxOptions

from bokeh.models import ColumnDataSource, Band, Span
from bokeh.io import export_svgs
from bokeh.plotting import figure, show

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas

from datetime import datetime

# read gating_config.yaml
config = yaml.safe_load(open(sys.argv[1]))

raw_path = config['raw_path']
cmpvs_path = config['cmpvs_path']
webdriver_path = config['webdriver_path']
webbrowser_path = config['webbrowser_path']
channel_metadata = config['channel_metadata']

# add geckodriver and firefox to system path for saving bokeh html plots
sys.path.append(webdriver_path)
sys.path.append(webbrowser_path)

# initialize webdriver instance
options = FirefoxOptions()
options.add_argument('--headless')
web_driver = Firefox(
    executable_path=webdriver_path,
    options=options
    )

# define path to save directory
save_dir = 'histograms'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# use common y-axis scale for all channel/tissue specific histograms
sharey = False

# read zeros tables
zeros = pd.read_csv(sys.argv[2])

# generate tissue- and channel- specific pdf pages
# for their corresponding histograms
for channel in sorted(channel_metadata.keys()):

    # generate pdf canvas
    my_canvas = canvas.Canvas(os.path.join(save_dir, f'{channel}.pdf'))

    if channel not in ['fsc', 'ssc']:

        # define logicle transformation model
        param_w = channel_metadata[channel][1]
        xform = fk.transforms.LogicleTransform(
            'logicle',
            param_t=262144.0,
            param_w=param_w,
            param_m=4.5,
            param_a=0
            )

    for tissue in sorted(zeros['tissue'].unique()):
        print()
        print(f'Processing {channel}-{tissue} pdf page:')
        print()

        # specify x, y origin coordinates for first histogram on the page
        x_start, y_start = (1500, 7700)

        # set x, y scaling parameters for histograms
        my_canvas.scale(0.10, 0.10)

        # add channel_tissue
        my_canvas.setFont('Helvetica', 175)
        my_canvas.drawString(300, 8100, f'{channel}_{tissue}')

        # get total cmpvs files for current tissue
        fcs_group = [
            i for i in os.listdir(cmpvs_path)
            if (i.startswith('measurement'))
            & ((i.endswith('.fcs')))
            & (i.split('_')[2] == tissue)
            ]

        # choose for development
        # fcs_group = [fcs_group[0]]

        # loop over tissue-specific fcs files and store sample data in dict
        data_dict = {}
        for fname in fcs_group:
            print(f'Storing data for {fname}')

            # extract sample metadata
            sample_metadata = fname.split('.fcs')[0]
            status = sample_metadata.split('_')[3]
            timepoint = sample_metadata.split('_')[1]
            replicate = sample_metadata.split('_')[4]

            # store cmpvs and tp-specific unstained fcs files as variables
            sample = fk.Sample(os.path.join(cmpvs_path, fname))
            fiducial = fk.Sample(
                os.path.join(raw_path, f'control_{timepoint}_unstained.fcs')
                )

            if channel not in ['fsc', 'ssc']:

                # apply logicle transformation to fcs files
                sample.apply_transform(xform)
                fiducial.apply_transform(xform)

            # get indices for current channel
            sample_channel_label = channel_metadata[channel][0]
            sample_channel_idx = sample.get_channel_index(sample_channel_label)

            if channel not in ['fsc', 'ssc']:
                fiducial_channel_label = (
                    sample_channel_label.split('FJComp-')[1]
                    )
            else:
                fiducial_channel_label = sample_channel_label

            fiducial_channel_idx = fiducial.get_channel_index(
                fiducial_channel_label
                )

            # store xform data for current channel as variables
            if channel not in ['fsc', 'ssc']:
                source = 'xform'
            else:
                source = 'raw'
            sample_xform_data = sample.get_channel_data(
                channel_index=sample_channel_idx,
                source=source, subsample=False)
            fiducial_xform_data = fiducial.get_channel_data(
                channel_index=fiducial_channel_idx,
                source=source, subsample=False)

            # trim extreme outliers
            lower_cutoff, upper_cutoff = (0.1, 99.9)

            cutoff_values = np.percentile(
                sample_xform_data, [lower_cutoff, upper_cutoff], axis=0
                )
            sample_xform_data = sample_xform_data[
                (sample_xform_data > cutoff_values[0])
                & (sample_xform_data < cutoff_values[1])
                ]
            cutoff_values = np.percentile(
                fiducial_xform_data, [lower_cutoff, upper_cutoff], axis=0
                )
            fiducial_xform_data = fiducial_xform_data[
                (fiducial_xform_data > cutoff_values[0])
                & (fiducial_xform_data < cutoff_values[1])
                ]

            if channel in ['fsc', 'ssc']:
                sample_xform_data = (
                    MinMaxScaler(feature_range=(0, 1), copy=True)
                    .fit_transform(sample_xform_data.reshape(-1, 1))
                    )
                sample_xform_data = sample_xform_data[:, 0]

                fiducial_xform_data = (
                    MinMaxScaler(feature_range=(0, 1), copy=True)
                    .fit_transform(fiducial_xform_data.reshape(-1, 1))
                    )
                fiducial_xform_data = fiducial_xform_data[:, 0]

            # get xform histogram data from fcs samples
            sample_hist, sample_edges = np.histogram(
                sample_xform_data, density=False, bins='fd'
                )
            fiducial_hist, fiducial_edges = np.histogram(
                fiducial_xform_data, density=False, bins='fd'
                )

            data_dict[f'{status}_tp{timepoint}_rep{replicate}'] = (
                fiducial_xform_data,
                sample_hist,
                sample_edges,
                fiducial_hist,
                fiducial_edges,
                )
        print()

        # identify max hist value among tissue-specific fcs files
        print('Finding max histogram count value.')
        counts_list = []
        for k, v in data_dict.items():
            counts_list.extend(v[1].tolist())
        if sharey:
            max_count = max(counts_list)
        else:
            max_count = None
        print()

        # loop over data_dict, generate corresponding svg plots,
        # convert to reportlab graphics, and store in reportlab_graphics dict
        reportlab_graphics = {}
        for k, v in data_dict.items():
            print(f'Plotting data for {k}')

            title = k
            status = title.split('_')[0]
            timepoint = int(re.findall('\d+', title.split('_')[1])[0])
            replicate = int(re.findall('\d+', title.split('_')[2])[0])

            fiducial_xform_data = v[0]
            sample_hist = v[1]
            sample_edges = v[2]
            fiducial_hist = v[3]
            fiducial_edges = v[4]

            # initialize Bokeh figure
            p = figure(
                title=title,
                title_location='above',
                tools='',
                background_fill_color='white'
                )
            p.title.text_font_size = '27pt'
            p.grid.grid_line_color = 'silver'
            p.grid.grid_line_width = 0.3
            p.yaxis.axis_label = 'event count'
            p.yaxis.axis_label_text_font_size = '21pt'
            p.yaxis.axis_label_standoff = 20
            p.xaxis.major_label_text_font_size = '17pt'
            p.xaxis.major_label_standoff = 28
            p.yaxis.major_label_text_font_size = '17pt'
            p.yaxis.major_label_standoff = 18

            # plot sample histogram
            p.quad(
                bottom=0,
                top=sample_hist,
                left=sample_edges[:-1],
                right=sample_edges[1:],
                fill_color='steelblue',
                line_color=None,
                alpha=1.0
                )

            # rescale fiducial histogram y-dim to sample histogram y-dim
            ymin = sample_hist.min()
            ymax = sample_hist.max()
            fiducial_hist_rescaled = (
                MinMaxScaler(feature_range=(ymin, ymax), copy=True)
                .fit_transform(fiducial_hist.reshape(-1, 1))
                )

            # plot fiducial histogram
            # p.quad(
            #     bottom=0,
            #     top=fiducial_hist_rescaled[:, 0],
            #     left=fiducial_edges[:-1],
            #     right=fiducial_edges[1:],
            #     fill_color='orange',
            #     line_color=None,
            #     alpha=0.5
            #     )

            # plot rescaled fiducial histogram as outline
            # p.line(
            #     fiducial_edges[:-1], fiducial_hist_rescaled[:, 0],
            #     line_color='crimson',
            #     line_width=2,
            #     alpha=1.0,
            #     )

            # shade area under the fiducial curve
            # d = {'x': fiducial_edges[:-1], 'y': fiducial_hist_rescaled[:, 0]}
            # df = pd.DataFrame(data=d)
            # plot_source = ColumnDataSource(
            #     data=dict(x=df['x'], y=df['y'])
            #     )
            # band = Band(
            #     base='x',
            #     upper='y',
            #     source=plot_source,
            #     level='overlay',
            #     fill_alpha=0.2,
            #     fill_color='crimson'
            #     )
            # p.add_layout(band)

            # extract fiducial kde data from seaborn kdeplot output
            x, y = (
                kdeplot(data=fiducial_xform_data)
                .get_lines()[0]
                .get_data()
                )
            plt.close('all')

            # rescale kde y-dimension to sample histogram
            y_rescaled = (
                MinMaxScaler(feature_range=(ymin, ymax), copy=True)
                .fit_transform(y.reshape(-1, 1))
                )

            # plot rescaled fiducial kde
            p.line(
                x, y_rescaled[:, 0],
                line_color='orange',
                line_width=6,
                alpha=1.0,
                legend_label='unstained kde'
                )

            # generate gate annoatation and add to plot
            gate = zeros['bias'][
                (zeros['channel'] == channel)
                & (zeros['tissue'] == tissue)
                & (zeros['status'] == status)
                & (zeros['timepoint'] == timepoint)
                & (zeros['replicate'] == replicate)
                ]

            gate = Span(
                location=gate.iloc[0],
                dimension='height',
                line_color='dimgrey',
                line_dash='solid',
                line_width=4.0
                )
            p.add_layout(gate)

            # assign common min and max values for x and y axes
            p.x_range.start = 0.0
            p.x_range.end = 1.0
            p.y_range.start = 0.0
            p.y_range.end = max_count

            # format legend
            p.legend.location = 'top_right'
            p.legend.background_fill_color = 'white'
            p.legend.label_text_font_size = '15pt'

            # show bokeh plot in Firefox browser
            # show(p)

            # convert bokeh plot to svg format
            p.output_backend = 'svg'

            # export bokeh plot
            plot_name = 'temp.svg'

            export_svgs(
                p,
                filename=os.path.join(save_dir, plot_name),
                webdriver=web_driver, timeout=3600
                )

            # import svg and assign label
            drawing = svg2rlg(os.path.join(save_dir, plot_name))

            # delete temp.svg
            os.remove('histograms/temp.svg')

            # append to reportlab_graphics dict
            reportlab_graphics[title] = drawing
        print()

        # generate an ordered dict to ensure lexicographic order
        # and format pdf pages with histograms
        print(f'Formatting pdf pages for {channel}.')
        od = collections.OrderedDict(sorted(reportlab_graphics.items()))

        # initialize plot ticker
        ticker = 0

        # loop over reportlab_graphics dict
        for k, v in od.items():

            # advance plot ticker by 1
            ticker += 1

            # convert reportlab drawing to pdf and embed in pdf page
            renderPDF.draw(
                v, my_canvas, x_start, y_start)

            # carriage return every 5th histogram
            if ticker % 4 == 0:
                x_start = 1500
                y_start -= 690

            # advance carriage to the right only
            else:
                x_start += 900

        # generate new pdf page
        my_canvas.showPage()

    # save pdf pages to disk
    my_canvas.save()
