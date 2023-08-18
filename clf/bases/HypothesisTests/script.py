import sys

from matplotlib.pyplot import rc, rcParams
import numpy as np
from pandas import DataFrame, read_csv
from scipy.stats import friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt, lst

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams.update({'font.size': 10})

aucs = DataFrame()

proposed_base_dir = jnt('..', '..', 'train', '_'.join(train_datasets), checkparts=True)
df = read_csv(jnt(proposed_base_dir, 'results_' + '_'.join(train_datasets) + '.csv'), header=None, index_col=0, usecols=range(11))

aucs['método proposto'] = df.loc['resnet50_5_0.8'].reset_index(drop=True)
aucs['SRC'] = df.loc['resnet50_5_1'].reset_index(drop=True)

bases_base_dir = jnt('..', 'Results')
for fname in lst(bases_base_dir):
    method = fname.split('.csv')[0]
    if method == 'SVM':
        method = 'SVM c/ \\textit{kernel} linear'
    if method == 'RBF':
        method = 'SVM c/ \\textit{kernel} RBF'
    if method == 'FT':
        method = '\\textit{fine-tuning}'
    df = read_csv(jnt(bases_base_dir, fname), header=None)
    aucs[method] = df

pfs = aucs.values
methods = aucs.columns

friedmanchisquare(*pfs.T)

ranks = np.array([rankdata(-p) for p in pfs])
avg_ranks = np.mean(ranks, axis=0)

print('\n'.join(f'{m} average rank: {r}' for m, r in zip(methods, avg_ranks)))

alpha = 0.05
cd = compute_CD(avg_ranks, pfs.shape[1], str(alpha), 'bonferroni-dunn')

graph_ranks(avg_ranks, lowv=1, names=methods, cd=cd, textspace=1.6, cdmethod=avg_ranks.argmin(), filename='postest_cls_ms.pdf')


