import os
import sys
import yaml
import pandas as pd
import flowkit as fk
from sklearn.preprocessing import MinMaxScaler

# read channel metadata
config = yaml.safe_load(open(sys.argv[1]))

channel_metadata = config['channel_metadata']
cmpvs_path = config['cmpvs_path']
zeros_path = config['zeros_path']

# read zeros tables
zeros = pd.read_csv(zeros_path)

# define path to save directory
save_dir = 'input'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# initialize dataframe to store zeroed data
df = pd.DataFrame()

# make a list of fcs files whose data is to be zeroed
fcs_files = [
    i for i in os.listdir(cmpvs_path)
    if (i.startswith('measurement'))
    & ((i.endswith('.fcs')))
    ]

# choose for development
# fcs_files = fcs_files[0:3]

# append zeroed fcs file to dataframe
for fname in fcs_files:

    print(f'Processing data for {fname}')

    sample_metadata = fname.split('.fcs')[0]
    timepoint = int(sample_metadata.split('_')[1])
    tissue = sample_metadata.split('_')[2]
    status = sample_metadata.split('_')[3]
    replicate = int(sample_metadata.split('_')[4])

    # assign fcs file as a variable
    sample = fk.Sample(os.path.join(cmpvs_path, fname))

    # initialize dataframe to store zeroed sample data
    df_temp = pd.DataFrame()

    for channel in sorted(channel_metadata.keys()):
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

            # apply logicle transformation to fcs file
            sample.apply_transform(xform)

        # select current channel to plot
        sample_channel_label = channel_metadata[channel][0]
        sample_channel_idx = sample.get_channel_index(
            sample_channel_label
            )

        # assign xform data as a variable
        if channel not in ['fsc', 'ssc']:
            sample_data = sample.get_channel_data(
                channel_index=sample_channel_idx,
                source='xform', subsample=False)
        else:
            sample_data = sample.get_channel_data(
                channel_index=sample_channel_idx,
                source='raw', subsample=False)
            sample_data = (
                MinMaxScaler(feature_range=(0, 1), copy=True)
                .fit_transform(sample_data.reshape(-1, 1))
                )
            sample_data = sample_data[:, 0]

        # generate gate annoatation and add to plot
        gate = zeros['bias'][
            (zeros['timepoint'] == timepoint)
            & (zeros['tissue'] == tissue)
            & (zeros['status'] == status)
            & (zeros['replicate'] == replicate)
            & (zeros['channel'] == channel)
            ].iloc[0]

        # subtract gate value from sample data
        sample_data_zeroed = sample_data - gate

        df_temp[channel] = sample_data_zeroed
        df_temp['replicate'] = replicate
        df_temp['status'] = status
        df_temp['tissue'] = tissue
        df_temp['timepoint'] = timepoint

    df = df.append(df_temp, ignore_index=True)

# rearrange columns
cols = (
    ['timepoint', 'tissue', 'status', 'replicate']
    + sorted(channel_metadata.keys())
    )
df = df[cols]

# save df as parquet file
df.to_parquet(os.path.join(save_dir, 'sylaras_input.parquet'), index=False)
