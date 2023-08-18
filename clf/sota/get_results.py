import sys

from csv import reader
from pandas import read_csv, Series

sys.path.insert(1, '../../aux')
from clfsettings import train_datasets
from pathutils import jnt, lst, mkd

train_name = '_'.join(train_datasets)
folds_dir = jnt('..', 'folds', train_name, checkparts=True)

dataset = 'DermNet' # edit here
#dataset = 'MClass-ND' # edit here

# edit here
#src_dir = jnt('.', 'ARL', 'Results') # edit here
#src_dir = jnt('.', 'my-thesis', 'benchmarks', 'pad', 'results', 'results', 'LewDir', 'mahalanobis', 'avg')
src_dir = jnt('.', 'IW', 'Results')
#src_dir = jnt('.', 'SLA', 'Results')
#src_dir = jnt('.', 'dermatologists', 'Results')

if 'ARL' in src_dir:
    fnames = [dataset + '_ARL.csv']
elif 'my-thesis' in src_dir:
    fnames = [dataset + '_DW.csv']
elif 'IW' in src_dir:
    fnames = [dataset + '_IW.csv']
elif 'SLA' in src_dir:
    fnames = [dataset + '_SLA.csv']
elif 'dermatologists':
    fnames = lst(src_dir)
    dataset = 'MClass-ND'

for fname in fnames:

    df1 = read_csv(jnt(src_dir, fname), header='infer')

    if 'my-thesis' in src_dir:
        #df1.MEL = df1.MEL/(df1.MEL + df1.NEV)
        df1.MEL = df1.MEL/(df1.ACK + df1.BCC + df1.MEL + df1.NEV + df1.SCC + df1.SEK)

    df1 = df1.rename(columns = {'image': 'img_name'})
    df1 = df1.rename(columns = {'MEL': 'mel_score'})

    dst_dir = jnt('.', 'results'); mkd(dst_dir) # edit here

    df2 = read_csv(jnt(folds_dir, dataset + '.csv'), header=None)
    df2.columns = ['dataset', 'label', 'img']

    scores = []
    with open(jnt(folds_dir, dataset + '.csv')) as csvfile:
        csvreader = reader(csvfile)
        for row in csvreader:
            dataset, label, img = row
            sel_row = df1[df1['img_name'] == img]
            scores.append(sel_row['mel_score'].values[0])

    df2['melscore'] = Series(scores, index=df2.index)

    if fname.startswith('derm'):
        fname = dataset + '_derm' + fname.split('.')[0].split('derm')[-1] + '.csv'

    df2.to_csv(jnt(dst_dir, fname), index=False)
