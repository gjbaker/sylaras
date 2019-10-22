import os
import shutil

dpi_dict = {
    'dpi_7': '/Users/gjbaker/Desktop/day_07',
    'dpi_14': '/Users/gjbaker/Desktop/day_14',
    'dpi_23': '/Users/gjbaker/Desktop/day_23',
    'dpi_31': '/Users/gjbaker/Desktop/31dpi_(1mouse)_2_19_2016_p.37-38(G1)',
    'dpi_35': '/Users/gjbaker/Desktop/35dpi_ran_in_FACS_tubes',
    'dpi_36': '/Users/gjbaker/Desktop/36dpi_ran_in_FACS_tubes'}

plate_map = {'A1': '1', 'A2': '2', 'A3': '3', 'A4': '4', 'A5': '1',
             'A6': '2', 'A7': '3', 'A8': '4', 'A9': '1', 'A10': '2',
             'A11': '3', 'A12': '4', 'B1': '5', 'B2': '6', 'B3': '7',
             'B4': '8', 'B5': '5', 'B6': '6', 'B7': '7', 'B8': '8',
             'B9': '5', 'B10': '6', 'B11': '7', 'B12': '8', 'C1': '1',
             'C2': '2', 'C3': '3', 'C4': '4', 'C5': '1', 'C6': '2',
             'C7': '3', 'C8': '4', 'C9': '1', 'C10': '2', 'C11': '3',
             'C12': '4', 'D1': '5', 'D2': '6', 'D3': '7', 'D4': '8',
             'D5': '5', 'D6': '6', 'D7': '7', 'D8': '8', 'D9': '5',
             'D10': '6', 'D11': '7', 'D12': '8', 'E1': '1', 'E2': '2',
             'E3': '3', 'E4': '4', 'E5': '1', 'E6': '2', 'E7': '3',
             'E8': '4', 'E9': '1', 'E10': '2', 'E11': '3', 'E12': '4',
             'F1': '5', 'F2': '6', 'F3': '7', 'F4': '8', 'F5': '5',
             'F6': '6', 'F7': '7', 'F8': '8', 'F9': '5', 'F10': '6',
             'F11': '7', 'F12': '8', 'G1': '1', 'G2': '2', 'G3': '3',
             'G4': '4', 'G5': '1', 'G6': '2', 'G7': '3', 'G8': '4',
             'G9': '1', 'G10': '2', 'G11': '3', 'G12': '4', 'H1': '5',
             'H2': '6', 'H3': '7', 'H4': '8', 'H5': '5', 'H6': '6',
             'H7': '7', 'H8': '8', 'H9': '5', 'H10': '6', 'H11': '7',
             'H12': '8'}

for dpi, directory in dpi_dict.items():
    if dpi in ['dpi_7', 'dpi_14', 'dpi_23']:
        os.chdir(directory)
        for fname in os.listdir():
            if fname != '.DS_Store':
                print('ORIGINAL: ' + fname)
                tag = fname.split('_', 4)[2]
                replicate = plate_map[tag]
                if tag[0] in ['A', 'B', 'E', 'F']:
                    status = 'naive'
                    if (tag[0] in ['A', 'B']) and (
                      tag[1:] in ['1', '2', '3', '4']):
                        tissue = 'blood'
                    elif (tag[0] in ['A', 'B']) and (
                      tag[1:] in ['5', '6', '7', '8']):
                        tissue = 'thymus'
                    elif (tag[0] in ['A', 'B']) and (
                      tag[1:] in ['9', '10', '11', '12']):
                        tissue = 'marrow'
                    elif (tag[0] in ['E', 'F']) and (
                      tag[1:] in ['1', '2', '3', '4']):
                        tissue = 'spleen'
                    elif (tag[0] in ['E', 'F']) and (
                      tag[1:] in ['5', '6', '7', '8']):
                        tissue = 'nodes'
                    elif (tag[0] in ['E', 'F']) and (
                      tag[1:] in ['9', '10', '11', '12']):
                        tissue = 'comp'

                else:
                    status = 'gl261'
                    if (tag[0] in ['C', 'D']) and (
                      tag[1:] in ['1', '2', '3', '4']):
                        tissue = 'blood'
                    elif (tag[0] in ['C', 'D']) and (
                      tag[1:] in ['5', '6', '7', '8']):
                        tissue = 'thymus'
                    elif (tag[0] in ['C', 'D']) and (
                      tag[1:] in ['9', '10', '11', '12']):
                        tissue = 'marrow'
                    elif (tag[0] in ['G', 'H']) and (
                      tag[1:] in ['1', '2', '3', '4']):
                        tissue = 'spleen'
                    elif (tag[0] in ['G', 'H']) and (
                      tag[1:] in ['5', '6', '7', '8']):
                        tissue = 'nodes'
                    elif (tag[0] in ['G', 'H']) and (
                      tag[1:] in ['9', '10', '11', '12']):
                        tissue = 'comp'

                print('UPDATED: ' + dpi + '_' + status +
                      '_' + tissue + '_' + replicate + '_' + fname)
                print()

                os.rename(fname, dpi + '_' + status +
                          '_' + tissue + '_' + replicate + '_' + fname)

    elif dpi in ['dpi_31', 'dpi_35', 'dpi_36']:
        os.chdir(directory)
        for fname in os.listdir():
            if '.DS_Store' not in fname:
                if 'Beads' not in fname:
                    print('ORIGINAL: ' + fname)
                    status = 'gl261'
                    # cross-referencing postbot.py replicate numbers
                    if dpi == 'dpi_31':
                        replicate = '2'

                    elif dpi == 'dpi_23':
                        tag = fname.split('_', 4)[2]
                        replicate = plate_map[tag]

                    elif dpi == 'dpi_35':
                        if 'M1' in fname:
                            replicate = '4'
                        elif 'M2' in fname:
                            replicate = '5'
                        else:
                            replicate = ''

                    elif dpi == 'dpi_36':
                        if 'M1' in fname:
                            replicate = '7'
                        elif 'M2' in fname:
                            replicate = '8'
                        else:
                            replicate = ''

                    if 'Blood' in fname:
                        tissue = 'blood'
                    elif 'BM' in fname:
                        tissue = 'marrow'
                    elif 'CLN' in fname:
                        tissue = 'nodes'
                    elif 'Cervical LN' in fname:
                        tissue = 'nodes'
                    elif 'Spleen' in fname:
                        tissue = 'spleen'
                    elif 'Thymus' in fname:
                        tissue = 'thymus'
                    elif 'comp' in fname:
                        tissue = 'comp'
                    elif 'Viabile' in fname:
                        tissue = 'comp'
                    elif 'Viability' in fname:
                        tissue = 'comp'
                    elif 'unstained' in fname:
                        tissue = 'comp'
                    elif 'Unstained' in fname:
                        tissue = 'comp'

                    print('UPDATED: ' + dpi + '_' + status +
                          '_' + tissue + '_' + replicate + '_' + fname)
                    print()

                    os.rename(fname, dpi + '_' + status +
                              '_' + tissue + '_' + replicate + '_' + fname)

for dpi, directory in dpi_dict.items():
    print(dpi)
    new_directory_path = os.path.join(
        '/Users/gjbaker/Desktop', dpi + '_update')
    os.makedirs(new_directory_path)
    for fname in os.listdir(directory):
        print(fname)
        if fname.startswith('dpi_'):
            shutil.move(
                directory + '/' + fname, new_directory_path + '/' + fname)
